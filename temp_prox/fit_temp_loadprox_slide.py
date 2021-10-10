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


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp
from tensorboardX import SummaryWriter

import numpy as np
import torch
import open3d as o3d

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img
from PIL import ImageDraw
import json
from temp_prox.optimizers import optim_factory

import temp_prox.fitting_temp_slide as fitting
from human_body_prior.tools.model_loader import load_vposer
import scipy.sparse as sparse


def fit_temp_loadprox_slide(img,
                            keypoints,
                            marker_mask,
                            scan,
                            scan_point_num,
                            scene_name,
                            body_model,
                            camera,
                            joint_weights,
                            body_pose_prior,
                            jaw_prior,
                            left_hand_prior,
                            right_hand_prior,
                            shape_prior,
                            expr_prior,
                            angle_prior,
                            result_fn_list,
                            mesh_fn_list,
                            out_img_fn_list,
                            log_folder,
                            loss_type='smplify',
                            use_face=True,
                            use_hands=True,
                            data_weights=None,
                            body_pose_prior_weights=None,
                            hand_pose_prior_weights=None,
                            jaw_pose_prior_weights=None,
                            shape_weights=None,
                            expr_weights=None,
                            hand_joints_weights=None,
                            face_joints_weights=None,
                            interpenetration=True,
                            coll_loss_weights=None,
                            df_cone_height=0.5,
                            penalize_outside=True,
                            max_collisions=8,
                            point2plane=False,
                            part_segm_fn='',
                            rho=100,
                            vposer_latent_dim=32,
                            vposer_ckpt='',
                            use_joints_conf=False,
                            interactive=True,
                            visualize=False,
                            save_meshes=True,
                            dtype=torch.float32,
                            ign_part_pairs=None,
                            ####################
                            ### PROX
                            render_results=True,
                            ## Depth
                            s2m=False,
                            s2m_weights=None,
                            m2s=False,
                            m2s_weights=None,
                            rho_s2m=1,
                            rho_m2s=1,
                            #penetration
                            sdf_penetration=False,
                            sdf_penetration_weights=0.0,
                            sdf_dir=None,
                            cam2world_dir=None,
                            #contact
                            contact=False,
                            contact_loss_weights=None,
                            contact_body_parts=None,
                            body_segments_dir=None,
                            load_scene=False,
                            scene_dir=None,
                            prox_params_dict=None,
                            # smooth acceleration term
                            smooth_acc=False,
                            smooth_acc_weights=None,
                            # smooth velocity term
                            smooth_vel=False,
                            smooth_vel_weights=None,
                            # motion smoothness prior term
                            use_motion_smooth_prior=False,
                            motion_prior_smooth_weights=None,
                            motion_smooth_model=None,
                            # friction term
                            use_friction=None,
                            friction_normal_weights=None,
                            friction_tangent_weights=None,
                            # motion infilling prior term
                            use_motion_infill_prior=False,
                            motion_infill_rec_weights=None,
                            motion_infill_contact_weights=None,
                            motion_infill_model=None,
                            infill_pretrain_weights=None,
                            device=None,
                            first_batch_flag=None,
                            **kwargs):
    batch_size = len(img)
    body_model.reset_params()  # set all params as 0


    if data_weights is None:
        data_weights = [1, ] * 5

    ############################## set loss weights if not predifined ############################
    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]
    msg = ('Number of Body pose prior weights does not match the number of data term weights')
    assert (len(data_weights) == len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) == len(body_pose_prior_weights)), msg

        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) == len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of Shape prior weights')
    assert (len(shape_weights) == len(body_pose_prior_weights)), msg

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) == len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of Expression prior weights')
        assert (len(expr_weights) == len(body_pose_prior_weights)), msg

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) == len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) == len(body_pose_prior_weights)), msg

    if smooth_acc:
        if smooth_acc_weights is None:
            smooth_acc_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of smooth loss weights')
        assert (len(smooth_acc_weights) == len(body_pose_prior_weights)), msg

    if smooth_vel:
        if smooth_vel_weights is None:
            smooth_vel_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of smooth vel loss weights')
        assert (len(smooth_vel_weights) == len(body_pose_prior_weights)), msg


    if use_motion_smooth_prior:
        if motion_prior_smooth_weights is None:
            motion_prior_smooth_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of motion prior smooth loss weights')
        assert (len(motion_prior_smooth_weights) == len(body_pose_prior_weights)), msg

    if use_friction:
        if friction_normal_weights is None:
            friction_normal_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of friction normal loss weights')
        assert (len(friction_normal_weights) == len(body_pose_prior_weights)), msg

        if friction_tangent_weights is None:
            friction_tangent_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of friction tangent loss weights')
        assert (len(friction_tangent_weights) == len(body_pose_prior_weights)), msg

    if use_motion_infill_prior:
        if motion_infill_rec_weights is None:
            motion_infill_rec_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of motion infill rec loss weights')
        assert (len(motion_infill_rec_weights) == len(body_pose_prior_weights)), msg

        if motion_infill_contact_weights is None:
            motion_infill_contact_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of motion infill contact loss weights')
        assert (len(motion_infill_contact_weights) == len(body_pose_prior_weights)), msg


    ######################## init vpose embedding from 0 / load vpose model ##########################
    use_vposer = kwargs.get('use_vposer', True)   # True
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)   # all 0, [bs, 32]

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()


    ######################## keypoints/joint_conf/scan pointclouds to device ##########################
    gt_joints = keypoints[:, :, :2]    # [bs, 118, 2]
    if use_joints_conf:   # user 2D keypoint condifence score
        joints_conf = keypoints[:, :, 2].reshape(keypoints.shape[0], -1)   # [bs, 118]
    scan_tensor = None
    if scan is not None:
        scan_tensor = scan.to(device)  # [bs, 20000, 3]
        scan_point_num = scan_point_num.to(device)  # [bs]


    ######################## load sdf/normals/cam2world of the current scene  ##########################
    sdf = None
    sdf_normals = None
    grid_min = None
    grid_max = None
    voxel_size = None
    if sdf_penetration or use_friction:  # True
        with open(osp.join(sdf_dir, scene_name + '.json'), 'r') as f:
            sdf_data = json.load(f)
            grid_min = torch.tensor(np.array(sdf_data['min']), dtype=dtype, device=device)  # [3]
            grid_max = torch.tensor(np.array(sdf_data['max']), dtype=dtype, device=device)
            grid_dim = sdf_data['dim']  # 256
        voxel_size = (grid_max - grid_min) / grid_dim    # [3]
        sdf = np.load(osp.join(sdf_dir, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        sdf = torch.tensor(sdf, dtype=dtype, device=device)   # [256, 256, 256]

        # for bs>1
        grid_min = grid_min.repeat(gt_joints.shape[0], 1).unsqueeze(1)  # [bs, 1, 3]
        grid_max = grid_max.repeat(gt_joints.shape[0], 1).unsqueeze(1)  # [bs, 1, 3]
        sdf = sdf.repeat(gt_joints.shape[0], 1, 1, 1).unsqueeze(1)  # [bs, 256, 256, 256]

        if osp.exists(osp.join(sdf_dir, scene_name + '_normals.npy')):
            sdf_normals = np.load(osp.join(sdf_dir, scene_name + '_normals.npy')).reshape(grid_dim, grid_dim, grid_dim, 3)  # [256, 256, 256, 3]
            sdf_normals = torch.tensor(sdf_normals, dtype=dtype, device=device)
        else:
            print("Normals not found...")

    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        cam2world = np.array(json.load(f))  # [4, 4] last row: [0,0,0,1]
        R = torch.tensor(cam2world[:3, :3].reshape(3, 3), dtype=dtype, device=device)
        t = torch.tensor(cam2world[:3, 3].reshape(1, 3), dtype=dtype, device=device)



    ######################## Create the search tree (for self interpenetration) ##########################
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:  # True
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used with CUDA'

        search_tree = BVH(max_collisions=max_collisions)  # max_collisions = 128

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:  # True
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']  # [20908] each face belongs to which body part??
            faces_parents = face_segm_data['parents']  # [20908]
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)


    #################################### load vertex ids of contact parts ##################################
    contact_verts_ids = None
    contact_fric_verts_ids = []
    for part in ['L_Leg', 'R_Leg', 'gluteus']:
        with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_fric_verts_ids.append(list(set(data["verts_ind"])))
    contact_fric_verts_ids = np.concatenate(contact_fric_verts_ids)

    if contact:
        contact_verts_ids = []
        for part in contact_body_parts:  # ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
            with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                data = json.load(f)
                contact_verts_ids.append(list(set(data["verts_ind"])))
        contact_verts_ids = np.concatenate(contact_verts_ids)  # [1121]


    ############################################  load scene mesh    ##################################
    scene_v = None
    if contact:
        if scene_name is not None:
            if load_scene:  # True
                from psbody.mesh import Mesh
                scene = Mesh(filename=os.path.join(scene_dir, scene_name + '.ply'))
                scene.vn = scene.estimate_vertex_normals()
                scene_v = torch.tensor(scene.v[np.newaxis, :], dtype=dtype, device=device).contiguous()   # [1, num_scene_verts, 3]


    ######################################## loss weights   ###################################
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights
    if s2m:
        opt_weights_dict['s2m_weight'] = s2m_weights
    if m2s:
        opt_weights_dict['m2s_weight'] = m2s_weights
    if sdf_penetration:
        opt_weights_dict['sdf_penetration_weight'] = sdf_penetration_weights
    if contact:
        opt_weights_dict['contact_loss_weight'] = contact_loss_weights
    if smooth_acc:
        opt_weights_dict['smooth_acc_weight'] = smooth_acc_weights
    if smooth_vel:
        opt_weights_dict['smooth_vel_weight'] = smooth_vel_weights
    if use_motion_smooth_prior:
        opt_weights_dict['motion_prior_smooth_weight'] = motion_prior_smooth_weights
    if use_friction:
        opt_weights_dict['friction_normal_weight'] = friction_normal_weights
        opt_weights_dict['friction_tangent_weight'] = friction_tangent_weights
    if use_motion_infill_prior:
        opt_weights_dict['motion_infill_rec_weight'] = motion_infill_rec_weights
        opt_weights_dict['motion_infill_contact_weight'] = motion_infill_contact_weights

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]

    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key], device=device, dtype=dtype)


    ########################### load indices of the head of smpl-x model ###########################
    with open(osp.join(body_segments_dir, 'body_mask.json'), 'r') as fp:
        head_indx = np.array(json.load(fp))
    N = body_model.get_num_verts()
    body_indx = np.setdiff1d(np.arange(N), head_indx)  # 1D array of values in `ar1` that are not in `ar2`
    head_mask = np.in1d(np.arange(N), head_indx)  # [10475] True/False
    body_mask = np.in1d(np.arange(N), body_indx)



    ##################################  define loss  #################################
    loss = fitting.create_loss(loss_type=loss_type,   # 'smplify'
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               s2m=s2m,
                               m2s=m2s,
                               rho_s2m=rho_s2m,
                               rho_m2s=rho_m2s,
                               head_mask=head_mask,
                               body_mask=body_mask,
                               sdf_penetration=sdf_penetration,
                               voxel_size=voxel_size,
                               grid_min=grid_min,
                               grid_max=grid_max,
                               sdf=sdf,
                               sdf_normals=sdf_normals,
                               R=R,
                               t=t,
                               contact=contact,
                               contact_verts_ids=contact_verts_ids,
                               dtype=dtype,
                               smooth_acc=smooth_acc,
                               smooth_vel=smooth_vel,
                               # motion prior
                               use_motion_smooth_prior=use_motion_smooth_prior,
                               motion_smooth_model=motion_smooth_model,
                               # friction term
                               use_friction=use_friction,
                               # batch_size=batch_size,
                               contact_fric_verts_ids=contact_fric_verts_ids,
                               # motion infill term
                               use_motion_infill_prior=use_motion_infill_prior,
                               motion_infill_model=motion_infill_model,
                               infill_pretrain_weights=infill_pretrain_weights,
                               device=device,
                               **kwargs)
    loss = loss.to(device
                   =device)


    ############################ fitting ####################################
    with fitting.FittingMonitor(**kwargs) as monitor:
        results_list = []

        #################################### optimize the whole body #######################################
        final_loss_val = 0
        opt_start = time.time()
        writer = SummaryWriter(log_dir=log_folder)

        ########### init from optimized prox body params
        for param_name in prox_params_dict:
            prox_params_dict[param_name] = prox_params_dict[param_name].detach().cpu().numpy()  # each param: array, [bs, xx]
        mean_betas = np.mean(prox_params_dict['betas'], axis=0)
        prox_params_dict['betas'] = np.repeat(np.expand_dims(mean_betas, axis=0), gt_joints.shape[0], axis=0)
        body_model.reset_params(**prox_params_dict)  # check requires_grad


        if use_vposer:
            with torch.no_grad():
                pose_embedding = torch.from_numpy(prox_params_dict['pose_embedding']).float().to(device)  # pose_embedding.requires_grad=True
                pose_embedding.requires_grad = True

        for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):

                ############################ define parameters to optimize ########################
                body_model.betas.requires_grad = False  # todo

                # body_model.parameters(): betas(bs, 10), global_orient(bs, 3), transl(bs, 3), left/right_hand_pose(bs, 12/12), jaw/leye/reye_pose(bs, 3/3/3), expression(bs, 10)
                body_params = list(body_model.parameters())
                final_params = list(filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)  # [1, 32]

                body_optimizer, body_create_graph = optim_factory.create_optimizer(final_params, **kwargs)
                body_optimizer.zero_grad()

                ########################### update weights ############################
                curr_weights['bending_prior_weight'] = (3.17 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 25:76] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 76:] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)


                ############################ fitting ###################################
                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    marker_mask=marker_mask,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    scan_tensor=scan_tensor,
                    scan_point_num=scan_point_num,
                    scene_v=scene_v,
                    return_verts=True, return_full_pose=True,
                    writer=writer,
                    first_batch_flag=first_batch_flag)

                if interactive:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()

                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)

                if interactive:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(opt_idx, elapsed))

        if interactive:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - opt_start
            tqdm.write('Body fitting done after {:.4f} seconds'.format(elapsed))
            tqdm.write('Body final loss val = {:.5f}'.format(final_loss_val))


        ########################### set results pkl formation #############################
        for i in range(batch_size):
            result = {'camera_' + str(key): val[i].unsqueeze(0).detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val[i].unsqueeze(0).detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
            if use_vposer:
                result['pose_embedding'] = pose_embedding[i].unsqueeze(0).detach().cpu().numpy()
                body_pose = vposer.decode(
                    pose_embedding,
                    output_type='aa').view(pose_embedding.shape[0], -1) if use_vposer else None
                result['body_pose'] = body_pose[i].unsqueeze(0).detach().cpu().numpy()
            results_list.append(result)


        ########################### save results pkl file #############################
        for i in range(len(results_list)):
            with open(result_fn_list[i], 'wb') as result_file:
                pickle.dump(results_list[i], result_file, protocol=2)


    #################################### save mesh #################################
    if save_meshes or render_results:
        body_pose = vposer.decode(
            pose_embedding,
            output_type='aa').view(pose_embedding.shape[0], -1) if use_vposer else None  # [bs, 63]

        model_type = kwargs.get('model_type', 'smpl')
        append_wrists = model_type == 'smpl' and use_vposer
        if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

        model_output = body_model(return_verts=True, body_pose=body_pose)  # all elements are in [bs, ...]
        vertices = model_output.vertices.detach().cpu().numpy()
        import trimesh

        out_mesh_list = []
        for i in range(len(vertices)):
            out_mesh = trimesh.Trimesh(vertices[i], body_model.faces, process=False)
            if save_meshes:
                out_mesh.export(mesh_fn_list[i])
            out_mesh_list.append(out_mesh)

    #################################### rendering #################################
    if render_results:
        import pyrender
        # common
        camera_center = np.array([951.30, 536.77])
        camera_pose = np.eye(4)
        camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
        camera_render = pyrender.camera.IntrinsicsCamera(
            fx=1060.53, fy=1060.38,
            cx=camera_center[0], cy=camera_center[1])
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))


        _, H, W, _ = img.shape  # H,W: 1080, 1920
        for i in range(len(img)):
            ############################# redering body in img #######################
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.3, 0.3, 0.3))
            scene.add(camera_render, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            body_mesh = pyrender.Mesh.from_trimesh(out_mesh_list[i], material=material)
            scene.add(body_mesh, 'mesh')
            r = pyrender.OffscreenRenderer(viewport_width=W,
                                           viewport_height=H,
                                           point_size=1.0)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]

            input_img = img[i].detach().cpu().numpy()
            output_img = color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img
            output_img = (output_img * 255).astype(np.uint8)

            # visualize 2d joints
            # openpose gt points
            projected_joints = gt_joints  # [bs, 118, 2]
            projected_joints = projected_joints[i].detach().cpu().numpy()  # [118, 2]
            projected_joints = projected_joints.astype(int)
            body_joints = projected_joints[0:25, :]  # [25, 2]
            draw = ImageDraw.Draw(output_img)
            for k in range(len(body_joints)):
                draw.ellipse((body_joints[k][0] - 2, body_joints[k][1] - 2,
                              body_joints[k][0] + 2, body_joints[k][1] + 2), fill=(255, 0, 0, 0))

            # optimized body
            projected_joints = camera(model_output.joints)  # [bs, 118, 2]
            projected_joints = projected_joints[i].detach().cpu().numpy()  # [118, 2]
            projected_joints = projected_joints.astype(int)
            body_joints = projected_joints[0:25, :]  # [25, 2]
            draw = ImageDraw.Draw(output_img)
            for k in range(len(body_joints)):
                draw.ellipse((body_joints[k][0] - 2, body_joints[k][1] - 2,
                              body_joints[k][0] + 2, body_joints[k][1] + 2), fill=(255, 0, 0, 0))


            cur_img = pil_img.fromarray(output_img)
            cur_img.save(out_img_fn_list[i])

            # ############################# redering body+scene #######################
            # body_mesh = pyrender.Mesh.from_trimesh(out_mesh_list[i], material=material)
            # static_scene = trimesh.load(osp.join(scene_dir, scene_name + '.ply'))
            # trans = np.linalg.inv(cam2world)
            # static_scene.apply_transform(trans)
            # static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)
            #
            # scene = pyrender.Scene()
            # scene.add(camera_render, pose=camera_pose)
            # scene.add(light, pose=camera_pose)
            #
            # scene.add(static_scene_mesh, 'mesh')
            # scene.add(body_mesh, 'mesh')
            #
            # r = pyrender.OffscreenRenderer(viewport_width=W,
            #                                viewport_height=H)
            # color, _ = r.render(scene)
            # color = color.astype(np.float32) / 255.0
            # cur_img = pil_img.fromarray((color * 255).astype(np.uint8))
            # cur_img.save(rendering_fn_list[i])


