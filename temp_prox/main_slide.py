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

import sys
import os

rootPath = '../'
sys.path.append(rootPath)

import os.path as osp

import time
import yaml
import scipy.io as sio
import open3d as o3d
import torch

import smplx


from temp_prox.misc_utils import JointMapper
from temp_prox.cmd_parser import parse_config
from temp_prox.data_parser_slide import *
from temp_prox.fit_temp_loadprox_slide import fit_temp_loadprox_slide

from temp_prox.camera import create_camera
from temp_prox.prior import create_prior

from models.AE import AE as AE_infill
from models.AE_sep import Enc

torch.backends.cudnn.enabled = False



def main(**args):
    gpu_id = args.get('gpu_id')
    torch.cuda.set_device(gpu_id)
    print('gpu id:', gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ################ read/creat paths ########################
    data_folder = args.get('recording_dir')
    recording_name = osp.basename(args.get('recording_dir'))   # e.x. 'N3OpenArea_00157_01'
    scene_name = recording_name.split("_")[0]   # e.x. 'N3OpenArea'
    base_dir = os.path.abspath(osp.join(args.get('recording_dir'), os.pardir, os.pardir))   # '/mnt/hdd/PROX'
    keyp_dir = osp.join(base_dir, 'keypoints')
    keyp_folder = osp.join(keyp_dir, recording_name)
    cam2world_dir = osp.join(base_dir, 'cam2world')
    scene_dir = osp.join(base_dir, 'scenes')
    calib_dir = osp.join(base_dir, 'calibration')
    sdf_dir = osp.join(base_dir, 'scenes_sdf')
    body_segments_dir = '../body_segments'
    marker_mask_dir = osp.join('../mask_markers', recording_name)
    prox_params_dir = osp.join(base_dir, 'PROXD', recording_name)
    # prox_params_dir = osp.join(base_dir, 'PROXD_filled', recording_name)
    if args.get('use_motion_infill'):
        prox_params_dir = osp.join('../fit_results_S2/', recording_name)  # TODO: to set


    output_folder = args.get('output_folder')
    output_folder = osp.expandvars(output_folder)
    output_folder = osp.join(output_folder, recording_name)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)
    # remove 'output_folder' from args list
    args.pop('output_folder')

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    out_rendering_dir = os.path.join(output_folder, 'renderings')
    if not osp.exists(out_rendering_dir):
        os.mkdir(out_rendering_dir)

    tensorboard_log_dir = os.path.join(output_folder, 'tensorboard_log')
    if not osp.exists(tensorboard_log_dir):
        os.mkdir(tensorboard_log_dir)

    input_gender = args.pop('gender', 'neutral')  # male
    dtype = torch.float32


    ################################## load motion prior model #############################
    if args.get('use_motion_smooth_prior'):
        motion_smooth_model = Enc(downsample=False, z_channel=64).to(device)
        weights = torch.load(args.get('AE_Enc_path'), map_location=lambda storage, loc: storage)
        motion_smooth_model.load_state_dict(weights)
        motion_smooth_model.eval()
        for param in motion_smooth_model.parameters():
            param.requires_grad = False
    else:
        motion_smooth_model = None

    ################################### load motion infilling model ###########################
    if args.get('use_motion_infill_prior'):
        motion_infill_model = AE_infill(downsample=True, in_channel=4, kernel=args.get('conv_kernel')).to(device)
        infill_pretrain_weights = torch.load(args.get('AE_infill_path'), map_location=lambda storage, loc: storage)
        motion_infill_model.load_state_dict(infill_pretrain_weights)
    else:
        motion_infill_model = None
        infill_pretrain_weights = None


    ####################### create data loader / joint mapper / joint weights ########################
    img_folder = args.pop('img_folder', 'Color')
    dataset_obj = OpenPose(img_folder=img_folder, data_folder=data_folder, keyp_folder=keyp_folder, calib_dir=calib_dir,
                           prox_params_dir=prox_params_dir,
                           output_params_dir=output_folder,
                           marker_mask_dir=marker_mask_dir, **args)
    data_loader = torch.utils.data.DataLoader(dataset=dataset_obj,
                                              batch_size=args.get('batch_size'),
                                              shuffle=False,
                                              num_workers=0, drop_last=True)
    # map smplx joints to openpose, 118=25body+21hand*2*51face
    joint_mapper = JointMapper(dataset_obj.get_model2data())

    # A weight for each joint of the model, 1 for each joint, 0 for joint 1,9,12
    joint_weights = dataset_obj.get_joint_weights().to(device=device, dtype=dtype)  # tensor, [118]
    joint_weights.unsqueeze_(dim=0)  # [1, 118]


    ####################### init smplx model ########################
    start = time.time()
    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        dtype=dtype,
                        **args)

    male_model = smplx.create(gender='male', **model_params)
    # SMPL-H has no gender-neutral model
    if args.get('model_type') != 'smplh':
        neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)


    ####################### create camera object ########################
    camera_center = None \
        if args.get('camera_center_x') is None or args.get('camera_center_y') is None \
        else torch.tensor([args.get('camera_center_x'), args.get('camera_center_y')], dtype=dtype).view(-1, 2)  # tensor, [1,2]
    camera = create_camera(focal_length_x=args.get('focal_length_x'),
                           focal_length_y=args.get('focal_length_y'),
                           center= camera_center,
                           batch_size=args.get('batch_size'),
                           dtype=dtype)

    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False


    ####################### creat prior type ########################
    use_hands = args.get('use_hands', True)  # True
    use_face = args.get('use_face', True)    # True

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')   # 12
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    ##################### objects to cuda #######################
    camera = camera.to(device=device)
    female_model = female_model.to(device=device)
    male_model = male_model.to(device=device)
    if args.get('model_type') != 'smplh':
        neutral_model = neutral_model.to(device=device)
    body_pose_prior = body_pose_prior.to(device=device)
    angle_prior = angle_prior.to(device=device)
    shape_prior = shape_prior.to(device=device)
    if use_face:
        expr_prior = expr_prior.to(device=device)
        jaw_prior = jaw_prior.to(device=device)
    if use_hands:
        left_hand_prior = left_hand_prior.to(device=device)
        right_hand_prior = right_hand_prior.to(device=device)


    ######################### start fitting ########################
    for idx, data in enumerate(data_loader):
        input_dict, prox_params_dict = data[0], data[1]  # ex. prox_params_dict[transl]: tensor, [bs, 3]
        for param_name in prox_params_dict:
            prox_params_dict[param_name] = prox_params_dict[param_name].to(device)

        ##################### read input img/keypoint/scan/... ###############
        img = input_dict['img'].to(device)   # tensor, [bs, 1080, 1920, 3]
        fn = input_dict['fn']                 # list, ['s001_frame_00001__00.00.00.033', ...]
        keypoints = input_dict['keypoints'].to(device)      # [bs, num_person, 118, 3]
        marker_mask = input_dict['marker_mask'].to(device)  # [bs, 67]
        init_trans = input_dict['init_trans'].to(device).view(-1,3)  # [bs, 3]

        scan_point_num = input_dict['scan_point_num']  # [bs], valid number of scan pts from depth img
        scan = input_dict['scan']  # [bs, 20000, 3], pad 0 for number_pts < 20000

        # todo: do not load depth info if you don't use depth in optimization terms
        # if args.get('batch_size') > 1:
        #     scan = None

        print('Processing: {} to {}'.format(input_dict['img_path'][0], input_dict['img_path'][-1]))  # 'points'/'colors': [num_valid_pts, 3]
        sys.stdout.flush()

        # TODO: won't work for multiple persons
        person_id = 0
        ####################### set save paths #########################
        curr_result_fn_list = []
        curr_mesh_fn_list = []
        curr_rendering_fn_list = []
        out_img_fn_list = []

        # path to save logs
        start_frame = idx * args.get('batch_size') + 1
        end_frame = start_frame + args.get('batch_size') - 1
        cur_log_folder = osp.join(tensorboard_log_dir, 'frame{}_to_frame{}'.format(start_frame, end_frame))
        if not osp.exists(cur_log_folder):
            os.makedirs(cur_log_folder)

        for i in range(len(fn)):
            # path to save images
            out_img_fn = osp.join(out_img_folder, fn[i] + '.png')
            out_img_fn_list.append(out_img_fn)

            # path to save rendered imgs
            curr_rendering_fn = osp.join(out_rendering_dir, fn[i] + '.png')
            curr_rendering_fn_list.append(curr_rendering_fn)

            # path to save optimized smplx params
            curr_result_folder = osp.join(result_folder, fn[i])
            if not osp.exists(curr_result_folder):
                os.makedirs(curr_result_folder)
            curr_result_fn = osp.join(curr_result_folder, '{:03d}.pkl'.format(person_id))
            curr_result_fn_list.append(curr_result_fn)

            # path to save optimized mesh
            curr_mesh_folder = osp.join(mesh_folder, fn[i])
            if not osp.exists(curr_mesh_folder):
                os.makedirs(curr_mesh_folder)
            curr_mesh_fn = osp.join(curr_mesh_folder, '{:03d}.ply'.format(person_id))
            curr_mesh_fn_list.append(curr_mesh_fn)


        gender = input_gender  # male
        if gender == 'neutral':
            body_model = neutral_model
        elif gender == 'female':
            body_model = female_model
        elif gender == 'male':
            body_model = male_model

        ########################## fitting #########################
        if idx == 0:
            first_batch_flag = 1  # if it's the 1st motion clip
        else:
            first_batch_flag = 0

        fit_temp_loadprox_slide(img=img,
                                keypoints=keypoints[:, person_id,],
                                marker_mask=marker_mask,
                                init_trans=init_trans,
                                scan_point_num=scan_point_num,
                                scan=scan,
                                cam2world_dir=cam2world_dir,
                                scene_dir=scene_dir,
                                sdf_dir=sdf_dir,
                                body_segments_dir=body_segments_dir,
                                scene_name=scene_name,
                                body_model=body_model,
                                camera=camera,
                                joint_weights=joint_weights,
                                dtype=dtype,
                                output_folder=output_folder,
                                out_img_fn_list=out_img_fn_list,
                                result_fn_list=curr_result_fn_list,
                                mesh_fn_list=curr_mesh_fn_list,
                                log_folder=cur_log_folder,
                                rendering_fn_list=curr_rendering_fn_list,
                                shape_prior=shape_prior,
                                expr_prior=expr_prior,
                                body_pose_prior=body_pose_prior,
                                left_hand_prior=left_hand_prior,
                                right_hand_prior=right_hand_prior,
                                jaw_prior=jaw_prior,
                                angle_prior=angle_prior,
                                prox_params_dict=prox_params_dict,
                                motion_smooth_model=motion_smooth_model,
                                motion_infill_model=motion_infill_model,
                                infill_pretrain_weights=infill_pretrain_weights,
                                device=device,
                                first_batch_flag=first_batch_flag,
                                **args)


    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))
    sys.stdout.flush()


if __name__ == "__main__":
    args = parse_config()
    main(**args)
