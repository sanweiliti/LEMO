import torchgeometry as tgm
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import datetime
import os, json, sys
import numpy as np
from utils.Quaternions import Quaternions
from utils.Pivots import Pivots
import scipy.ndimage.filters as filters
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(logdir):
    logger = logging.getLogger('emotion')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    image_paths = []
    for looproot, _, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith(suffix):
                image_paths.append(os.path.join(looproot, filename))
    return image_paths


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def decode(module_input):
        reshaped_input = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def matrot2aa(pose_matrot):  # input: [bs, 3, 3]
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])  # [bs, 3, 4], float
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous()
        return pose_body_matrot



def convert_to_6D_rot(x_batch):
    '''
    input: [transl, rotation, local params]
    convert global rotation from Eular angle to 6D continuous representation
    '''

    xt = x_batch[:,:3]
    xr = x_batch[:,3:6]
    xb = x_batch[:, 6:]

    xr_mat = ContinousRotReprDecoder.aa2matrot(xr) # return [:,3,3]
    xr_repr =  xr_mat[:,:,:-1].reshape([-1,6])

    return torch.cat([xt, xr_repr, xb], dim=-1)



def convert_to_3D_rot(x_batch):
    '''
    input: [transl, 6d rotation, local params]
    convert global rotation from 6D continuous representation to Eular angle
    '''
    xt = x_batch[:,:3]   # (reconstructed) normalized global translation
    xr = x_batch[:,3:9]  # (reconstructed) 6D rotation vector
    xb = x_batch[:,9:]   # pose $ shape parameters

    xr_mat = ContinousRotReprDecoder.decode(xr)  # [bs,3,3]
    xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat) # return [:,3]

    return torch.cat([xt, xr_aa, xb], dim=-1)



def convert_to_6D_all(x_batch):
    xr_mat = ContinousRotReprDecoder.aa2matrot(x_batch)  # return [:,3,3]
    xr_repr = xr_mat[:, :, :-1].reshape([-1, 6])
    return xr_repr


def convert_to_3D_all(x_batch):
    # x_batch: [bs, 6]
    xr_mat = ContinousRotReprDecoder.decode(x_batch)  # [bs,3,3]
    xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat)  # return [:,3]
    return xr_aa



def gen_body_mesh_v1(body_params, smplx_model, vposer_model):
    # body_params: [T, 3+6+10+32/126 (+180:hands)]
    bs = body_params.shape[0]
    body_params_dict = {}
    body_params_dict['transl'] = body_params[:, 0:3]  # [T, 3]
    body_params_dict['global_orient'] = body_params[:, 3:6]  # [T, 3]
    body_params_dict['betas'] = body_params[:, 6:16]
    body_params_dict['body_pose'] = vposer_model.decode(body_params[:, 16:48], output_type='aa').view(bs, -1)
    body_params_dict['left_hand_pose'] = body_params[:, 48:60]
    body_params_dict['right_hand_pose'] = body_params[:, 60:]

    smplx_output = smplx_model(return_verts=True, **body_params_dict)  # generated human body mesh
    body_verts = smplx_output.vertices  # [bs, n_body_vert, 3]
    return body_verts

def gen_body_joints_v1(body_params, smplx_model, vposer_model):
    # body_params: [T, 3+6+10+32/126 (+180:hands)]
    bs = body_params.shape[0]  # T=120 frames
    body_params_dict = {}
    body_params_dict['transl'] = body_params[:, 0:3]  # [T, 3]
    body_params_dict['global_orient'] = body_params[:, 3:6]  # [T, 3]
    body_params_dict['betas'] = body_params[:, 6:16]
    body_params_dict['body_pose'] = vposer_model.decode(body_params[:, 16:48], output_type='aa').view(bs, -1)
    body_params_dict['left_hand_pose'] = body_params[:, 48:60]
    body_params_dict['right_hand_pose'] = body_params[:, 60:]

    smplx_output = smplx_model(return_verts=True, **body_params_dict)  # generated human body mesh
    body_joints = smplx_output.joints  # [bs, n_body_vert, 3]
    return body_joints




def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)



def reconstruct_global_body(body_joints_input, rot_0_pivot):
    root_traj = body_joints_input[:, -1]  # [T, 3]
    root_r, root_x, root_z = root_traj[:, 2], root_traj[:, 0], root_traj[:, 1]  # [T]
    body_joints_input = body_joints_input[:, 0:-1]  # [T, 25+1, 3]
    body_joints_input[:, :, [1, 2]] = body_joints_input[:, :, [2, 1]]
    rotation = Quaternions.id(1)
    # offsets = []
    translation = np.array([[0, 0, 0]])
    for i in range(len(body_joints_input)):
        if i == 0:
            rotation = Quaternions.from_angle_axis(-rot_0_pivot, np.array([0, 1, 0])) * rotation   # t=0
        body_joints_input[i, :, :] = rotation * body_joints_input[i]
        body_joints_input[i, :, 0] = body_joints_input[i, :, 0] + translation[0, 0]
        body_joints_input[i, :, 2] = body_joints_input[i, :, 2] + translation[0, 2]
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
        translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])

    body_joints_input[:, :, [1, 2]] = body_joints_input[:, :, [2, 1]]
    body_joints_input = body_joints_input[:, 1:, :]
    return body_joints_input





def get_local_markers_4chan(cur_body, contact_lbls):
    # cur_body: numpy, [T, 1+67, 3]
    # contact_lbls: numpy, [T, 4]
    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]  # swap y/z axis  --> in (x,z,y)

    """ Put on Floor """
    cur_body[:, :, 1] = cur_body[:, :, 1] - cur_body[:, :, 1].min()

    """ Add Reference Joint """
    reference = cur_body[:, 0] * np.array([1, 0, 1])  # [T, 3], (x,y,0)
    cur_body = np.concatenate([reference[:, np.newaxis], cur_body], axis=1)  # [T, 1+25, 3]

    """ Get Root Velocity in floor plane """
    velocity = (cur_body[1:, 0:1] - cur_body[0:-1, 0:1]).copy()  # [T-1, 3] ([:, 1]==0)

    """ To local coordinates """
    cur_body[:, :, 0] = cur_body[:, :, 0] - cur_body[:, 0:1, 0]  # [T, 1+25, 3]
    cur_body[:, :, 2] = cur_body[:, :, 2] - cur_body[:, 0:1, 2]

    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 26 + 1 + 1, 56 + 1 + 1, 27 + 1 + 1, 57 + 1 + 1  # +1+1: [0]: reference, [1]: pelvis
    across1 = cur_body[:, hip_r] - cur_body[:, hip_l]
    across0 = cur_body[:, sdr_r] - cur_body[:, sdr_l]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    cur_body = rotation * cur_body  # [T, 1+25, 3]

    """ Get Root Rotation """
    velocity = rotation[1:] * velocity  # [T-1, 1, 3]
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps  # [T-1, 1]

    rot_0_pivot = Pivots.from_quaternions(rotation[0]).ps  # [T-1, 1]

    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]
    cur_body = cur_body[0:-1, 1:, :]  # [T-1, 25, 3]  exclude reference joint
    cur_body = cur_body.reshape(len(cur_body), -1)  # [T-1, 75]


    channel_local = np.concatenate([cur_body, contact_lbls[0:-1]], axis=-1)[np.newaxis, :, :]  # [1, T-1, d=75+4]
    T, d = channel_local.shape[1], channel_local.shape[-1]
    global_x, global_y = velocity[:, :, 0], velocity[:, :, 2]  # [T-1, 1]
    channel_global_x = np.repeat(global_x, d).reshape(1, T, d)  # [1, T-1, d]
    channel_global_y = np.repeat(global_y, d).reshape(1, T, d)  # [1, T-1, d]
    channel_global_r = np.repeat(rvelocity, d).reshape(1, T, d)  # [1, T-1, d]

    cur_body = np.concatenate([channel_local, channel_global_x, channel_global_y, channel_global_r],
                               axis=0)  # [4, T-1, d]
    return cur_body, rot_0_pivot



JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplhf',
    'right_eye_smplhf'] # first 25 joints in smplx

LIMBS_BODY = [(23, 15),
         (24, 15),
         (15, 22),
         (22, 12),
         # left arm
         (12, 13),
         (13, 16),
         (16, 18),
         (18, 20),
         # right arm
         (12, 14),
         (14, 17),
         (17, 19),
         (19, 21),
         # spline
         (12, 9),
         (9, 6),
         (6, 3),
         (3, 0),
         # left leg
         (0, 1),
         (1, 4),
         (4, 7),
         (7, 10),
         # right leg
         (0, 2),
         (2, 5),
         (5, 8),
         (8, 11)]


LIMBS_BODY_SMPL = [(15, 12),
         # left arm
         (12, 13),
         (13, 16),
         (16, 18),
         (18, 20),
         (20, 22),
         # right arm
         (12, 14),
         (14, 17),
         (17, 19),
         (19, 21),
         (21, 23),
         # spline
         (12, 9),
         (9, 6),
         (6, 3),
         (3, 0),
         # left leg
         (0, 1),
         (1, 4),
         (4, 7),
         (7, 10),
         # right leg
         (0, 2),
         (2, 5),
         (5, 8),
         (8, 11),]



LIMBS_HAND = [(20, 25),
              (25, 26),
              (26, 27),
              (20, 28),
              (28, 29),
              (29, 30),
              (20, 31),
              (31, 32),
              (32, 33),
              (20, 34),
              (34, 35),
              (35, 36),
              (20, 37),
              (37, 38),
              (38, 39),
              # right hand
              (21, 40),
              (40, 41),
              (41, 42),
              (21, 43),
              (43, 44),
              (44, 45),
              (21, 46),
              (46, 47),
              (47, 48),
              (21, 49),
              (49, 50),
              (50, 51),
              (21, 52),
              (52, 53),
              (53, 54)]


openpose2smplx = [(1, 12),
                  (2, 17),
                  (3, 19),
                  (4, 21),
                  (5, 16),
                  (6, 18),
                  (7, 20),
                  (8, 0),
                  (9, 2),
                  (10, 5),
                  (11, 8),
                  (12, 1),
                  (13, 4),
                  (14, 7),
                  (11, 11),
                  (14, 10)]  # (1, 12): openpose joint 1 = smplx joint 12

LIMBS_MARKER_SSM2 = [(65, 63),
                     (65, 39),
                     (63, 9),
                     (39, 9),
                     (63, 64),
                     (65, 66),
                     (39, 56),
                     (9, 26),
                     (56, 1),
                     (26, 1),
                     (1, 61),
                     (61, 38),
                     (61, 8),
                     (38, 52),
                     (8, 22),
                     (52, 33),
                     (22, 3),
                     (33, 31),
                     (3, 31),
                     (33, 57),
                     (3, 27),
                     (57, 45),
                     (27, 14),
                     (45, 48),
                     (14, 18),
                     (48, 59),
                     (18, 29),
                     (59, 32),
                     (29, 2),
                     (32, 51),
                     (2, 21),
                     # arm
                     (56, 40),
                     (40, 43),
                     (43, 53),
                     (53, 42),
                     (26, 5),
                     (5, 10),
                     (10, 13),
                     (13, 23),
                     (23, 12),
                     # # back
                     # (64, 0),
                     # (66, 0),
                     # (0, 4),
                     # (0, 34),
                     # (4, 6),
                     # (34, 36),
                     # (6, 62),
                     # (36, 62),
                     # (62, 24),
                     # (62, 54),
                     # (24, 7),
                     # (54, 37),
                     # (7, 16),
                     # (37, 47)
                     ]

