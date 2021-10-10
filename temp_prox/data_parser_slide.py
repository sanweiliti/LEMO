# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp

import json
import pickle

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


from temp_prox.misc_utils import smpl_to_openpose
from temp_prox.projection_utils import Projection


Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)   # data: dict, key: version/people

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):  # data['people']: list of n dicts (detect n people)
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)  # [75]: [x1,y1,c1, x2,y2,c2,...]
        body_keypoints = body_keypoints.reshape([-1, 3])  # [25, 3]
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])   # [21, 3]
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])   # [21, 3]

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)  # [67, 3]
        if use_face:
            # 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]   # [51, 3]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)   # [118, 3]

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)   # list of [128, 3] (each item: one person in the current image)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)



def read_prox_pkl(pkl_path):
    body_params_dict = {}
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        # data keys:
        # transl, global_orient, betas, body_pose, pose_embedding
        # left_hand_pose, right_hand_pose,
        # jaw_pose, leye_pose, reye_pose, expression
        body_params_dict['transl'] = data['transl'][0]
        body_params_dict['global_orient'] = data['global_orient'][0]
        body_params_dict['betas'] = data['betas'][0]
        body_params_dict['body_pose'] = data['body_pose'][0]  # array, [63,]
        body_params_dict['pose_embedding'] = data['pose_embedding'][0]

        body_params_dict['left_hand_pose'] = data['left_hand_pose'][0]
        body_params_dict['right_hand_pose'] = data['right_hand_pose'][0]
        body_params_dict['jaw_pose'] = data['jaw_pose'][0]
        body_params_dict['leye_pose'] = data['leye_pose'][0]
        body_params_dict['reye_pose'] = data['reye_pose'][0]
        body_params_dict['expression'] = data['expression'][0]
    return body_params_dict



class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 calib_dir='',
                 prox_params_dir='',
                 output_params_dir='',
                 marker_mask_dir='',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 depth_folder='Depth',
                 mask_folder='BodyIndex',
                 mask_color_folder='BodyIndexColor',
                 read_depth=False,
                 read_mask=False,
                 mask_on_color=False,
                 depth_scale=1e-3,
                 flip=False,
                 start=0,
                 step=1,
                 scale_factor=1,
                 frame_ids=None,
                 init_mode='sk',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands  # True
        self.use_face = use_face  # True
        self.model_type = model_type    # smplx
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign   # [1,9,12]
        self.use_face_contour = use_face_contour    # False
        self.batch_size = kwargs.get('batch_size')

        self.openpose_format = openpose_format   # coco25

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)  # body_joints=25, hand_joints=20, num_joints=65
        self.img_folder = osp.join(data_folder, img_folder)   # e.x. '/mnt/hdd/PROX/recordings/N3OpenArea_00157_01/Color'
        self.keyp_folder = osp.join(keyp_folder)
        self.depth_folder = os.path.join(data_folder, depth_folder)   # '.../Depth'
        self.mask_folder = os.path.join(data_folder, mask_folder)     # '.../BodyIndex'
        self.mask_color_folder = os.path.join(data_folder, mask_color_folder)     # '.../BodyIndexColor'

        self.img_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)
                          if img_fn.endswith('.png') or
                          img_fn.endswith('.jpg') and
                          not img_fn.startswith('.')]
        self.img_paths = sorted(self.img_paths)   # list of all image paths in 'Color' folder, sorted
        if frame_ids is None:
            self.img_paths = self.img_paths[start::step]
        else:
            self.img_paths = [self.img_paths[id -1] for id in frame_ids]

        self.prox_params_dir = prox_params_dir
        # self.seq_joint_mask = np.load('{}/mask_joint.npy'.format(joint_mask_dir))  # [seq_len, 25]
        self.seq_marker_mask = np.load('{}/mask_markers.npy'.format(marker_mask_dir))  # [seq_len, 67]


        #### repeat img paths, marker masks
        self.current_params_dir = output_params_dir
        # slide_window_size = 70  # todo: set slide window length as 70
        slide_window_size = int(self.batch_size * 0.7)  # set slide window length as 0.7*batch_size
        seq_n = (len(self.img_paths) - self.batch_size) - slide_window_size

        self.img_paths_slide = self.img_paths[0:self.batch_size]
        self.seq_marker_mask_slide = self.seq_marker_mask[0:self.batch_size]
        # self.seq_joint_mask_slide = self.seq_joint_mask[0:self.batch_size]
        for i in range(int(seq_n) + 1):
            start = slide_window_size * (i + 1)
            end = min(start + self.batch_size, len(self.img_paths))
            self.img_paths_slide += self.img_paths[start:end]
            self.seq_marker_mask_slide = np.append(self.seq_marker_mask_slide, self.seq_marker_mask[start:end], axis=0)
            # self.seq_joint_mask_slide = np.append(self.seq_joint_mask_slide, self.seq_joint_mask[start:end], axis=0)



        self.cnt = 0
        self.depth_scale = depth_scale   # 0.001
        self.flip = flip   # True
        self.read_depth = read_depth   # True
        self.read_mask = read_mask     # True
        self.scale_factor = scale_factor   # 1
        self.init_mode = init_mode    # 'scan'
        self.mask_on_color = mask_on_color    # True
        self.projection = Projection(calib_dir)    # self.projection.color_cam/depth_cam: dict, extrinsic/intrinsic parameters

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)  # return a mapping relation

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:    # self.joints_to_ign: [1,9,12]
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)


    def __len__(self):
        # print(len(self.img_paths))
        return len(self.img_paths_slide)

    def __getitem__(self, idx):
        img_path = self.img_paths_slide[idx]
        # joint_mask = self.seq_joint_mask_slide[idx]
        marker_mask = self.seq_marker_mask_slide[idx]
        return self.read_item(img_path, marker_mask)

    def read_item(self, img_path, marker_mask):
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0    # [1080, 1920, 3]
        if self.flip:   # True
            img = cv2.flip(img, 1)  # flip horizontally
        img_fn = osp.split(img_path)[1]
        img_fn, _ = osp.splitext(osp.split(img_path)[1])   # e.x.  's001_frame_00001__00.00.00.033'
        (H, W, _) = img.shape

        # e.x. '.../PROX/keypoints/N3OpenArea_00157_01/s001_frame_00001__00.00.00.033_keypoints.json'
        keypoint_fn = osp.join(self.keyp_folder, img_fn + '_keypoints.json')
        keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)

        if len(keyp_tuple.keypoints) < 1:
            return {}
        keypoints = np.stack(keyp_tuple.keypoints)   # [n_person, 118, 3]
        keypoints = np.expand_dims(keypoints[0], axis=0)  # [1,118,3]
        # keypoints = torch.from_numpy(keypoints).float()

        depth_im = None
        if self.read_depth:
            depth_im = cv2.imread(os.path.join(self.depth_folder, img_fn + '.png'), flags=-1).astype(float)  # [424, 512]
            depth_im = depth_im / 8.
            depth_im = depth_im * self.depth_scale   # self.depth_scale=0.001
            if self.flip:
                depth_im = cv2.flip(depth_im, 1)


        mask = None
        if self.read_mask:
            if self.mask_on_color:  # True
                mask = cv2.imread(os.path.join(self.mask_color_folder, img_fn + '.png'), cv2.IMREAD_GRAYSCALE)  # [1080, 1920]
            else:
                mask = cv2.imread(os.path.join(self.mask_folder, img_fn + '.png'), cv2.IMREAD_GRAYSCALE)
                mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1]
            if self.flip:
                mask = cv2.flip(mask, 1)

        scan_dict = None
        init_trans = None
        if depth_im is not None and mask is not None:
            scan_dict = self.projection.create_scan(mask, depth_im, mask_on_color=self.mask_on_color)  # 'points'/'colors': [num_valid_pts, 3]
            init_trans = np.mean(scan_dict.get('points'), axis=0)   # initial translation of the body: [3]
        # init_trans = torch.from_numpy(init_trans).float()

        img = torch.from_numpy(img).float()
        depth_im = torch.from_numpy(depth_im).float()
        # scan_dict['points'] = torch.from_numpy(scan_dict['points']).float()  # [num_valid_pts, 3]
        # scan_dict['colors'] = torch.from_numpy(scan_dict['colors']).float()
        init_trans = torch.from_numpy(init_trans).float()
        # joint_mask = torch.from_numpy(joint_mask).float()
        marker_mask = torch.from_numpy(marker_mask).float()

        scan = torch.from_numpy(scan_dict['points']).float()  # [num_valid_pts, 3]
        scan_point_num = scan.shape[0]
        if scan_point_num < 20000:
            pad = torch.zeros(20000 - scan_point_num, 3)
            scan = torch.cat([scan, pad], dim=0)  # [20000, 3]
        else:
            scan = scan[0:20000]  # get 20000 scan pts at most

        ################# load prox optimized data ################
        # e.x., prox_params_dir='/mnt/hdd/PROX/PROXD/N3OpenArea_00157_01'
        # read optimized results from last sequence for first 30 frames in current sequence
        # make sure current save dir is empty before starting!
        pkl_path = osp.join(self.current_params_dir, 'results', img_fn, '000.pkl')
        if not os.path.exists(pkl_path):
            pkl_path = osp.join(self.prox_params_dir, 'results', img_fn, '000.pkl')

        body_params_dict = read_prox_pkl(pkl_path)

        output_dict = {'fn': img_fn,
                       'img_path': img_path,
                       'keypoints': keypoints,
                       'img': img,
                       'init_trans': init_trans,
                       'depth_im': depth_im,
                       'mask': mask,
                       'marker_mask': marker_mask,
                       'scan_point_num': scan_point_num,
                       'scan': scan,
                       }
        return output_dict, body_params_dict

