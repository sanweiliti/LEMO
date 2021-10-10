import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm import tqdm
import glob
import smplx
from utils.utils import *
import scipy.ndimage.filters as filters
from utils.Quaternions import Quaternions
from utils.Pivots import Pivots



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainLoader(data.Dataset):
    def __init__(self, clip_seconds=8, clip_fps=30, normalize=False, split='train', mode=None):
        self.clip_seconds = clip_seconds
        self.clip_len = clip_seconds * clip_fps  # T frames for each clip
        self.data_dict_list = []
        self.normalize = normalize
        self.clip_fps = clip_fps
        self.split = split  # train/test
        self.mode = mode


    def divide_clip(self, dataset_name='HumanEva', amass_dir=None):
        npz_fnames = glob.glob(os.path.join(amass_dir, dataset_name, '*/*_poses.npz'))  # name list of all npz sequence files in current dataset
        npz_fnames.sort()
        fps_list = []
        # print('sequence #: ', len(npz_fnames))
        cnt_sub_clip = 0
        # print('reading sequences in %s...' % (dataset_name))
        for npz_fname in npz_fnames:
            cdata = np.load(npz_fname)
            fps = int(cdata['mocap_framerate'])  # check fps of current sequence
            clip_len = self.clip_seconds * fps
            fps_list.append(fps)
            if fps == 150:
                sample_rate = 5
            elif fps == 120:
                sample_rate = 4
            elif fps == 60:
                sample_rate = 2
            else:
                continue

            N = len(cdata['poses'])  # total frame number of the current sequence
            if N >= clip_len:
                num_valid_clip = int(N/clip_len)
                seq_trans = cdata['trans']
                seq_poses = cdata['poses']
                seq_dmpls = cdata['dmpls']
                seq_betas = cdata['betas']
                seq_gender = str(cdata['gender'])
                seq_fps = int(cdata['mocap_framerate'])

                for i in range(num_valid_clip):
                    # pose: 156-d (3 global rotation + 63 body pose + 45 left hand pose + 45 right hand pose)
                    # dmpls: 8-d, trans: 3-d, betas: 16-d
                    data_dict = {}
                    data_dict['trans'] = seq_trans[(clip_len * i):clip_len * (i + 1)][::sample_rate, ]  # [T, 3]
                    data_dict['poses'] = seq_poses[(clip_len * i):clip_len * (i + 1)][::sample_rate, ]  # [T, 156]
                    data_dict['betas'] = seq_betas  # [10]
                    data_dict['gender'] = seq_gender  # male/female
                    data_dict['mocap_framerate'] = seq_fps
                    self.data_dict_list.append(data_dict)
                    cnt_sub_clip += 1
            else:
                continue

        # print('get {} sub clips from dataset {}'.format(cnt_sub_clip, dataset_name))
        # print('fps range:', min(fps_list), max(fps_list), '\n')


    def read_data(self, amass_datasets, amass_dir):
        for dataset_name in tqdm(amass_datasets):
            self.divide_clip(dataset_name, amass_dir)
        self.n_samples = len(self.data_dict_list)
        print('[INFO] get {} sub clips in total.'.format(self.n_samples))


    def create_body_repr(self, with_hand=False, smplx_model_path=None):
        print('[INFO] create motion clip imgs by {}...'.format(self.mode))

        smplx_model_male = smplx.create(smplx_model_path, model_type='smplx', gender='male', ext='npz',
                                        use_pca=False,  flat_hand_mean=True, # different here, true: mean hand pose is a flat hand
                                        create_global_orient=True, create_body_pose=True, create_betas=True,
                                        create_left_hand_pose=True, create_right_hand_pose=True, create_expression=True,
                                        create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True, create_transl=True,
                                        batch_size=self.clip_len).to(device)
        smplx_model_female = smplx.create(smplx_model_path, model_type='smplx', gender='female', ext='npz',
                                          use_pca=False, flat_hand_mean=True, # different here, true: mean hand pose is a flat hand
                                          create_global_orient=True, create_body_pose=True, create_betas=True,
                                          create_left_hand_pose=True, create_right_hand_pose=True, create_expression=True,
                                          create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True, create_transl=True,
                                          batch_size=self.clip_len).to(device)

        self.clip_img_list = []
        self.beta_list = []
        self.rot_0_pivot_list = []
        self.transf_matrix_smplx_list = []
        self.smplx_params_gt_list = []
        for i in tqdm(range(self.n_samples)):
            ####################### set smplx params (gpu tensor) for each motion clip ##################
            body_param_ = {}
            body_param_['transl'] = self.data_dict_list[i]['trans']  # [T, 3]
            body_param_['global_orient'] = self.data_dict_list[i]['poses'][:, 0:3]
            body_param_['body_pose'] = self.data_dict_list[i]['poses'][:, 3:66]  # [T, 63]
            body_param_['left_hand_pose'] = self.data_dict_list[i]['poses'][:, 66:111]  # [T, 45]
            body_param_['right_hand_pose'] = self.data_dict_list[i]['poses'][:, 111:]  # [T, 45]
            body_param_['betas'] = np.tile(self.data_dict_list[i]['betas'][0:10], (len(body_param_['transl']), 1))  # [T, 10]

            for param_name in body_param_:
                body_param_[param_name] = torch.from_numpy(body_param_[param_name]).float().to(device)


            ######################## set  body representations (global joints for body/hand) #####################
            if self.data_dict_list[i]['gender'] == 'male':
                smplx_output = smplx_model_male(return_verts=True, **body_param_)  # generated human body mesh
            elif self.data_dict_list[i]['gender'] == 'female':
                smplx_output = smplx_model_female(return_verts=True, **body_param_)  # generated human body mesh
            joints = smplx_output.joints  # [T, 127, 3]

            if self.mode in ['local_markers', 'local_markers_4chan']:
                with open('loader/SSM2.json') as f:
                    marker_ids = list(json.load(f)['markersets'][0]['indices'].values())
                markers = smplx_output.vertices[:, marker_ids, :]  # # [T(/bs), 41, 3]

            ##### transfrom to pelvis at origin, face y axis
            joints_frame0 = joints[0].detach()  # [N, 3] joints of first frame
            x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]
            x_axis[-1] = 0
            x_axis = x_axis / torch.norm(x_axis)
            z_axis = torch.tensor([0, 0, 1]).float().to(device)
            y_axis = torch.cross(z_axis, x_axis)
            y_axis = y_axis / torch.norm(y_axis)
            transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)  # [3, 3]
            joints = torch.matmul(joints - joints_frame0[0], transf_rotmat)  # [T(/bs), 25, 3]
            transl_1 = - joints_frame0[0]
            if self.mode in ['local_markers', 'local_markers_4chan']:
                markers = torch.matmul(markers - joints_frame0[0], transf_rotmat)  # [T(/bs), n_marker, 3]

            ######## obtain binary contact labels
            if self.mode in ['local_joints', 'local_joints_4chan']:
                # left_heel_id, right_heel_id = 7, 8, left_toe, right_toe = 10, 11
                # velocity criteria
                left_heel_vel = torch.norm((joints[1:, 7:8] - joints[0:-1, 7:8]) * self.clip_fps, dim=-1)  # [T-1, 1]
                right_heel_vel = torch.norm((joints[1:, 8:9] - joints[0:-1, 8:9]) * self.clip_fps, dim=-1)
                left_toe_vel = torch.norm((joints[1:, 10:11] - joints[0:-1, 10:11]) * self.clip_fps, dim=-1)
                right_toe_vel = torch.norm((joints[1:, 11:12] - joints[0:-1, 11:12]) * self.clip_fps, dim=-1)

                foot_joints_vel = torch.cat([left_heel_vel, right_heel_vel, left_toe_vel, right_toe_vel],
                                            dim=-1)  # [T-1, 4]

                is_contact = torch.abs(foot_joints_vel) < 0.22
                contact_lbls = torch.zeros([joints.shape[0], 4]).to(device)  # all -1, [T, 4]
                contact_lbls[0:-1, :][is_contact == True] = 1.0  # 0/1, 1 if contact for first T-1 frames

                # z height criteria
                z_thres = torch.min(joints[:, :, -1]) + 0.10
                foot_joints = torch.cat([joints[:, 7:8], joints[:, 8:9], joints[:, 10:11], joints[:, 11:12]],
                                        dim=-2)  # [T, 4, 3]
                thres_lbls = (foot_joints[:, :, 2] < z_thres).float()  # 0/1, [T, 4]

                # combine 2 criterias
                contact_lbls = contact_lbls * thres_lbls
                contact_lbls[-1, :] = thres_lbls[-1, :]  # last frame contact lbl: only look at z height
                # contact_lbls[contact_lbls == 0] = -1.0  # tensor, 1 for contact, -1 for no contact
                contact_lbls = contact_lbls.detach().cpu().numpy()

            if self.mode in ['local_markers', 'local_markers_4chan']:
                # left foot markers: 25, 30, 16, right_foot_markers: 55, 60, 47
                # velocity criteria
                left_heel_vel = torch.norm((markers[1:, 16:17] - markers[0:-1, 16:17]) * self.clip_fps,
                                           dim=-1)  # [T-1, 1]
                right_heel_vel = torch.norm((markers[1:, 47:48] - markers[0:-1, 47:48]) * self.clip_fps, dim=-1)
                left_toe_vel = torch.norm((markers[1:, 30:31] - markers[0:-1, 30:31]) * self.clip_fps, dim=-1)
                right_toe_vel = torch.norm((markers[1:, 60:61] - markers[0:-1, 60:61]) * self.clip_fps, dim=-1)

                foot_markers_vel = torch.cat([left_heel_vel, right_heel_vel, left_toe_vel, right_toe_vel], dim=-1)  # [T-1, 4]

                is_contact = torch.abs(foot_markers_vel) < 0.22
                contact_lbls = torch.zeros([markers.shape[0], 4]).to(device)  # all -1, [T, 4]
                contact_lbls[0:-1, :][is_contact == True] = 1.0  # 0/1, 1 if contact for first T-1 frames

                # z height criteria
                z_thres = torch.min(markers[:, :, -1]) + 0.10
                foot_markers = torch.cat([markers[:, 16:17], markers[:, 47:48], markers[:, 30:31], markers[:, 60:61]], dim=-2)  # [T, 4, 3]
                thres_lbls = (foot_markers[:, :, 2] < z_thres).float()  # 0/1, [T, 4]

                # combine 2 criterias
                contact_lbls = contact_lbls * thres_lbls
                contact_lbls[-1, :] = thres_lbls[-1, :]  # last frame contact lbl: only look at z height
                contact_lbls = contact_lbls.detach().cpu().numpy()

            ######## get body representation
            body_joints = joints[:, 0:25]  # [T, 25, 3]  root(1) + body(21) + jaw/leye/reye(3)
            hand_joints = joints[:, 25:55]  # [T, 30, 3]

            if self.mode in ['local_joints', 'local_joints_4chan']:
                if not with_hand:
                    cur_body = body_joints  # [T, 25, 3]
                else:
                    cur_body = torch.cat([body_joints, hand_joints], axis=1)  # [T, 55, 3]

            if self.mode in ['local_markers', 'local_markers_4chan']:
                cur_body = torch.cat([body_joints[:, 0:1], markers], dim=1)  # first row: pelvis joint

            ############################# local repre from Holten ###############################
            cur_body = cur_body.detach().cpu().numpy()  # numpy, [T, 25, 3], in (x,y,z)
            cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]  # swap y/z axis  --> in (x,z,y)

            """ Put on Floor """
            z_transl = cur_body[:, :, 1].min()
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
            if self.mode in ['local_joints', 'local_joints_4chan']:
                sdr_l, sdr_r, hip_l, hip_r = 16 + 1, 17 + 1, 1 + 1, 2 + 1  # +1: [0]: reference
            if self.mode in ['local_markers', 'local_markers_4chan']:
                sdr_l, sdr_r, hip_l, hip_r = 26 + 1 + 1, 56 + 1 + 1, 27 + 1 + 1, 57 + 1 + 1  # +1+1: [0]: reference, [1]: pelvis
            across1 = cur_body[:, hip_r] - cur_body[:, hip_l]
            across0 = cur_body[:, sdr_r] - cur_body[:, sdr_l]
            across = across0 + across1
            across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

            direction_filterwidth = 20
            forward = np.cross(across, np.array([[0, 1, 0]]))
            # forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
            forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

            """ Remove Y Rotation """
            target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
            rotation = Quaternions.between(forward, target)[:, np.newaxis]
            cur_body = rotation * cur_body  # [T, 1+25, 3]

            """ Get Root Rotation """
            velocity = rotation[1:] * velocity  # [T-1, 1, 3]
            rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps  # [T-1, 1]

            # return rotation of 1st frame too
            rot_0_pivot = Pivots.from_quaternions(rotation[0]).ps  # [T-1, 1]

            cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]
            cur_body = cur_body[0:-1, 1:, :]  # [T-1, 25, 3]
            cur_body = cur_body.reshape(len(cur_body), -1)  # [T-1, 75]

            # global_vel = np.concatenate([velocity[:, :, 0], velocity[:, :, 2], rvelocity], axis=-1)

            if self.mode in ['local_joints', 'local_markers']:
                # global vel (3) + local pose (25*3) + contact label (4)
                global_vel = np.concatenate([velocity[:, :, 0], velocity[:, :, 2], rvelocity], axis=-1)
                cur_body = np.concatenate([global_vel, cur_body, contact_lbls[0:-1]], axis=-1)  # [T-1, d=3+75+4]

            elif self.mode in ['local_joints_4chan', 'local_markers_4chan']:
                channel_local = np.concatenate([cur_body, contact_lbls[0:-1]], axis=-1)[np.newaxis, :,
                                :]  # [1, T-1, d=75+4]
                T, d = channel_local.shape[1], channel_local.shape[-1]
                global_x, global_y = velocity[:, :, 0], velocity[:, :, 2]  # [T-1, 1]
                channel_global_x = np.repeat(global_x, d).reshape(1, T, d)  # [1, T-1, d]
                channel_global_y = np.repeat(global_y, d).reshape(1, T, d)  # [1, T-1, d]
                channel_global_r = np.repeat(rvelocity, d).reshape(1, T, d)  # [1, T-1, d]

                cur_body = np.concatenate([channel_local, channel_global_x, channel_global_y, channel_global_r],
                                          axis=0)  # [4, T-1, d]

            ############## get ground truth smplx body transform
            # + transl_1, *transf_rotmat, -z_transl(in z axis)
            transf_matrix_1 = torch.tensor([[1, 0, 0, transl_1[0]],
                                            [0, 1, 0, transl_1[1]],
                                            [0, 0, 1, transl_1[2]],
                                            [0, 0, 0, 1]])
            transf_matrix_2 = torch.zeros(4, 4)
            transf_matrix_2[0:3, 0:3] = transf_rotmat.T
            transf_matrix_2[-1, -1] = 1

            transf_matrix_3 = torch.tensor([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, -z_transl],
                                            [0, 0, 0, 1]])

            transf_matrix_smplx = torch.matmul(transf_matrix_3,
                                               torch.matmul(transf_matrix_2, transf_matrix_1)).detach()  # [4, 4]
            smplx_params_gt = torch.cat([body_param_['transl'], body_param_['global_orient'],
                                         body_param_['betas'], body_param_['body_pose'],
                                         body_param_['left_hand_pose'],
                                         body_param_['right_hand_pose']], dim=-1).detach()  # [T, 3+3+10+63+45+45]

            self.clip_img_list.append(cur_body)
            self.rot_0_pivot_list.append(rot_0_pivot)
            self.transf_matrix_smplx_list.append(transf_matrix_smplx)
            self.smplx_params_gt_list.append(smplx_params_gt)

        self.clip_img_list = np.asarray(self.clip_img_list)  # [N, T-1, d] / [N, 4, T-1, d]


        if self.normalize:
            prefix = 'preprocess_stats_infill'
            if with_hand:
                prefix += '_withHand'

            if self.mode in ['local_joints', 'local_markers']:
                Xmean = self.clip_img_list.mean(axis=1).mean(axis=0)  # [1, 1, d]
                Xmean[-4:] = 0.0

                Xstd = np.ones(self.clip_img_list.shape[-1])
                Xstd[0:2] = self.clip_img_list[:, :, 0:2].std()  # global traj vel x/y
                Xstd[2] = self.clip_img_list[:, :, 2].std()  # rotation vel
                Xstd[3:-4] = self.clip_img_list[:, :, 3:-4].std()  # local joints
                Xstd[-4:] = 1.0

                if self.split == 'train':
                    np.savez_compressed('preprocess_stats/{}_{}.npz'.format(prefix, self.mode), Xmean=Xmean, Xstd=Xstd)
                    self.clip_img_list = (self.clip_img_list - Xmean) / Xstd
                elif self.split == 'test':
                    stats = np.load('preprocess_stats/{}_{}.npz'.format(prefix, self.mode))
                    self.clip_img_list = (self.clip_img_list - stats['Xmean']) / stats['Xstd']

            elif self.mode in ['local_joints_4chan', 'local_markers_4chan']:
                d = self.clip_img_list.shape[-1]
                Xmean_local = self.clip_img_list[:, 0].mean(axis=1).mean(axis=0)  # [d]
                Xmean_local[-4:] = 0.0
                Xstd_local = np.ones(d)
                Xstd_local[0:] = self.clip_img_list[:, 0].std()  # [d]
                Xstd_local[-4:] = 1.0

                Xmean_global_xy = self.clip_img_list[:, 1:3].mean()  # scalar
                Xstd_global_xy = self.clip_img_list[:, 1:3].std()  # scalar

                Xmean_global_r = self.clip_img_list[:, 3].mean()  # scalar
                Xstd_global_r = self.clip_img_list[:, 3].std()  # scalar

                if self.split == 'train':
                    np.savez_compressed('preprocess_stats/{}_{}.npz'.format(prefix, self.mode),
                                        Xmean_local=Xmean_local, Xstd_local=Xstd_local,
                                        Xmean_global_xy=Xmean_global_xy, Xstd_global_xy=Xstd_global_xy,
                                        Xmean_global_r=Xmean_global_r, Xstd_global_r=Xstd_global_r)
                    self.clip_img_list[:, 0] = (self.clip_img_list[:, 0] - Xmean_local) / Xstd_local
                    self.clip_img_list[:, 1:3] = (self.clip_img_list[:, 1:3] - Xmean_global_xy) / Xstd_global_xy
                    self.clip_img_list[:, 3] = (self.clip_img_list[:, 3] - Xmean_global_r) / Xstd_global_r
                elif self.split == 'test':
                    stats = np.load('preprocess_stats/{}_{}.npz'.format(prefix, self.mode))
                    self.clip_img_list[:, 0] = (self.clip_img_list[:, 0] - stats['Xmean_local']) / stats['Xstd_local']
                    self.clip_img_list[:, 1:3] = (self.clip_img_list[:, 1:3] - stats['Xmean_global_xy']) / stats['Xstd_global_xy']
                    self.clip_img_list[:, 3] = (self.clip_img_list[:, 3] - stats['Xmean_global_r']) / stats['Xstd_global_r']


        print('[INFO] motion clip imgs created.')


    def __len__(self):
        return self.n_samples


    def __getitem__(self, index):
        if self.mode in ['local_joints', 'local_markers']:
            clip_img = self.clip_img_list[index]  # [T, d] d dims of body representation
            clip_img = torch.from_numpy(clip_img).float().permute(1, 0).unsqueeze(0)  # [1, d, T]
        elif self.mode in ['local_joints_4chan', 'local_markers_4chan']:
            clip_img = self.clip_img_list[index]  # [4, T, d]
            clip_img = torch.from_numpy(clip_img).float().permute(0, 2, 1)  # [4, d, T]
        smplx_beta = torch.from_numpy(self.data_dict_list[index]['betas'][0:10]).float()  # [10]
        gender = self.data_dict_list[index]['gender']
        rot_0_pivot = self.rot_0_pivot_list[index]
        transf_matrix_smplx = self.transf_matrix_smplx_list[index]
        smplx_params_gt = self.smplx_params_gt_list[index]

        if gender == 'female':
            gender = 0
        elif gender == 'male':
            gender = 1
        return [clip_img, smplx_beta, gender, rot_0_pivot, transf_matrix_smplx, smplx_params_gt]


