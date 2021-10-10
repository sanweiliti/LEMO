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

import open3d as o3d
import sys



import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import time
import smplx
import torch.optim as optim
import itertools

import matplotlib.pyplot as plt

import temp_prox.misc_utils as utils
import temp_prox.dist_chamfer as ext
distChamfer = ext.chamferDist()

from utils.utils import *



def update_cam(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T)  # T is applied in the rotated coord
    cam_aux = np.array([[0, 0, 0, 1]])
    mat = np.concatenate([cam_R, cam_T], axis=-1)
    mat = np.concatenate([mat, cam_aux], axis=0)
    cam_param.extrinsic = mat
    return cam_param

@torch.no_grad()
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''

    body_pose = vposer.decode(
        pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints_3d = output.joints
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t


class FittingMonitor(object):
    def __init__(self, summary_steps=1,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol  # 1e-09  The tolerance threshold for the function
        self.gtol = gtol  # 1e-09  The tolerance threshold for the gradient

        self.summary_steps = summary_steps  # 1
        self.body_color = body_color
        self.model_type = model_type  # smplx

        # self.visualize = visualize
        # self.viz_mode = viz_mode  # o3d

    def __enter__(self):
        self.steps = 0
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        print('total steps:', self.steps)

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer  # false
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)  # closure: fitting_func()  forward, compute loss, backward
            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            # if n > 0 and prev_loss is not None and self.ftol > 0:
            #     loss_rel_change = utils.rel_change(prev_loss, loss.item())
            #
            #     if loss_rel_change <= self.ftol:
            #         break
            #
            # if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
            #         for var in params if var.grad is not None]):
            #     break

            prev_loss = loss.item()
        return prev_loss


    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               marker_mask=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               scan_tensor=None,
                               scan_point_num=None,
                               scene_v=None,
                               create_graph=False,
                               writer=None,
                               first_batch_flag=None,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)   # [62724]
        append_wrists = self.model_type == 'smpl' and use_vposer  # false

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(pose_embedding, output_type='aa').view(pose_embedding.shape[0], -1) if use_vposer else None   # [1, 63]  pose_embedding: initialize from all zeros (32-d)
            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6], dtype=body_pose.dtype, device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,  # false
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)  # false

            ########## output body joints in smplx topology
            joint_mapper = body_model.joint_mapper
            body_model.joint_mapper = None
            smplx_joints = body_model(return_verts=return_verts,
                                      body_pose=body_pose,
                                      return_full_pose=return_full_pose).joints
            body_model.joint_mapper = joint_mapper


            ########### compute loss
            loss_dict = loss(body_model=body_model,
                             body_model_output=body_model_output,
                             smplx_joints=smplx_joints,
                             camera=camera,
                             gt_joints=gt_joints,
                             body_model_faces=faces_tensor,
                             joints_conf=joints_conf,
                             marker_mask=marker_mask,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              scan_tensor=scan_tensor,
                              scan_point_num=scan_point_num,
                              scene_v=scene_v,
                              opt_step=self.steps,
                              **kwargs)  # SMPLifyLoss

            if backward:
                loss_dict['total_loss'].backward(create_graph=create_graph)

            # bs=100: erase gradient for first 15 frames
            bs = smplx_joints.shape[0]
            erase_n = int(bs*0.15)
            if not first_batch_flag:
                for body_param in body_model.parameters():
                    if body_param.grad is not None:
                        body_param.grad[0:erase_n, :] = 0
                pose_embedding.grad[0:erase_n, :] = 0



            if writer is not None:
                writer.add_scalar('optimize/total_loss', loss_dict['total_loss'].item(), self.steps)
                writer.add_scalar('optimize/joint_loss', loss_dict['joint_loss'].item(), self.steps)
                writer.add_scalar('optimize/s2m_dist', loss_dict['s2m_dist'].item(), self.steps)
                writer.add_scalar('optimize/m2s_dist', loss_dict['m2s_dist'].item(), self.steps)
                writer.add_scalar('optimize/self_penetration_loss', loss_dict['self_penetration_loss'].item(), self.steps)
                writer.add_scalar('optimize/sdf_penetration_loss', loss_dict['sdf_penetration_loss'].item(), self.steps)
                writer.add_scalar('optimize/contact_loss', loss_dict['contact_loss'].item(), self.steps)
                writer.add_scalar('optimize/smooth_acc_loss', loss_dict['smooth_acc_loss'].item(), self.steps)
                writer.add_scalar('optimize/smooth_vel_loss', loss_dict['smooth_vel_loss'].item(), self.steps)
                writer.add_scalar('optimize/motion_prior_smooth_loss', loss_dict['motion_prior_smooth_loss'].item(), self.steps)
                writer.add_scalar('optimize/loss_fric_tangent', loss_dict['loss_fric_tangent'].item(), self.steps)
                writer.add_scalar('optimize/loss_fric_normal', loss_dict['loss_fric_normal'].item(), self.steps)
                writer.add_scalar('optimize/motion_infill_loss', loss_dict['motion_infill_loss'].item(), self.steps)
                writer.add_scalar('optimize/motion_infill_contact_loss', loss_dict['motion_infill_contact_loss'].item(), self.steps)

            self.steps += 1
            # return total_loss
            return loss_dict['total_loss']

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 # rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 s2m=False,
                 m2s=False,
                 rho_s2m=1,
                 rho_m2s=1,
                 s2m_weight=0.0,
                 m2s_weight=0.0,
                 head_mask=None,
                 body_mask=None,
                 sdf_penetration=False,
                 voxel_size=None,
                 grid_min=None,
                 grid_max=None,
                 sdf=None,
                 sdf_normals=None,
                 sdf_penetration_weight=0.0,
                 R=None,
                 t=None,
                 contact=False,
                 contact_loss_weight=0.0,
                 contact_verts_ids=None,
                 ######## smooth acc
                 smooth_acc=False,
                 smooth_acc_weight=0.0,
                 ######## smooth vel
                 smooth_vel=False,
                 smooth_vel_weight=0.0,
                 ######## motion prior
                 use_motion_smooth_prior=False,
                 motion_prior_smooth_weight=0.0,
                 motion_smooth_model=None,
                 ######## friction term
                 use_friction=False,
                 friction_normal_weight=0.0,
                 friction_tangent_weight=0.0,
                 contact_fric_verts_ids=None,
                 ######## motion infill term
                 use_motion_infill_prior=False,
                 motion_infill_rec_weight=0.0,
                 motion_infill_contact_weight=0.0,
                 motion_infill_model=None,
                 infill_pretrain_weights=None,
                 device=None,
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.device = device

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.s2m = s2m
        self.m2s = m2s
        self.s2m_robustifier = utils.GMoF(rho=rho_s2m)
        self.m2s_robustifier = utils.GMoF(rho=rho_m2s)

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.body_mask = body_mask
        self.head_mask = head_mask

        self.R = R
        self.t = t

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance


        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

        self.register_buffer('s2m_weight',
                             torch.tensor(s2m_weight, dtype=dtype))
        self.register_buffer('m2s_weight',
                             torch.tensor(m2s_weight, dtype=dtype))

        self.sdf_penetration = sdf_penetration
        if self.sdf_penetration:
            self.sdf = sdf
            self.sdf_normals = sdf_normals
            self.voxel_size = voxel_size
            self.grid_min = grid_min
            self.grid_max = grid_max
            self.register_buffer('sdf_penetration_weight',
                                 torch.tensor(sdf_penetration_weight, dtype=dtype))
        self.contact = contact
        if self.contact:
            self.contact_verts_ids = contact_verts_ids
            self.register_buffer('contact_loss_weight',
                                 torch.tensor(contact_loss_weight, dtype=dtype))

        self.smooth_acc = smooth_acc
        if self.smooth_acc:
            self.register_buffer('smooth_acc_weight',
                                 torch.tensor(smooth_acc_weight, dtype=dtype))

        self.smooth_vel = smooth_vel
        if self.smooth_vel:
            self.register_buffer('smooth_vel_weight',
                                 torch.tensor(smooth_vel_weight, dtype=dtype))


        self.use_friction = use_friction
        if self.use_friction:
            self.contact_fric_verts_ids = contact_fric_verts_ids
            self.sdf = sdf
            self.sdf_normals = sdf_normals
            self.voxel_size = voxel_size
            self.grid_min = grid_min
            self.grid_max = grid_max
            self.register_buffer('friction_normal_weight',
                                 torch.tensor(friction_normal_weight, dtype=dtype))
            self.register_buffer('friction_tangent_weight',
                                 torch.tensor(friction_tangent_weight, dtype=dtype))

        self.use_motion_infill_prior = use_motion_infill_prior
        if self.use_motion_infill_prior:
            self.motion_infill_model = motion_infill_model
            self.infill_pretrain_weights = infill_pretrain_weights

            body_segments_dir = '../body_segments'
            with open(os.path.join(body_segments_dir, 'L_Leg.json'), 'r') as f:
                data = json.load(f)
                self.left_foot_verts_id = np.asarray(list(set(data["verts_ind"])))
            left_heel_verts_id = np.load('../foot_verts_id/left_heel_verts_id.npy')
            left_toe_verts_id = np.load('../foot_verts_id/left_toe_verts_id.npy')
            self.left_heel_verts_id = self.left_foot_verts_id[left_heel_verts_id]
            self.left_toe_verts_id = self.left_foot_verts_id[left_toe_verts_id]

            with open(os.path.join(body_segments_dir, 'R_Leg.json'), 'r') as f:
                data = json.load(f)
                self.right_foot_verts_id = np.asarray(list(set(data["verts_ind"])))
            right_heel_verts_id = np.load('../foot_verts_id/right_heel_verts_id.npy')
            right_toe_verts_id = np.load('../foot_verts_id/right_toe_verts_id.npy')
            self.right_heel_verts_id = self.right_foot_verts_id[right_heel_verts_id]
            self.right_toe_verts_id = self.right_foot_verts_id[right_toe_verts_id]


            self.register_buffer('motion_infill_rec_weight',
                                 torch.tensor(motion_infill_rec_weight, dtype=dtype))
            self.register_buffer('motion_infill_contact_weight',
                                 torch.tensor(motion_infill_contact_weight, dtype=dtype))



        self.use_motion_smooth_prior = use_motion_smooth_prior
        if self.use_motion_smooth_prior:
            self.motion_smooth_model = motion_smooth_model
            self.register_buffer('motion_prior_smooth_weight',
                                 torch.tensor(motion_prior_smooth_weight, dtype=dtype))

        with open('../loader/SSM2_withhand.json') as f:
            self.smooth_marker_ids = list(json.load(f)['markersets'][0]['indices'].values())  # list, [81]

        if self.use_motion_infill_prior or self.use_motion_smooth_prior:
            with open('../loader/SSM2.json') as f:
                self.infill_marker_ids = list(json.load(f)['markersets'][0]['indices'].values())  # list, [67]

            # global markers (for motion smoothness prior)
            preprocess_stats = np.load('../preprocess_stats/preprocess_stats_smooth_withHand_global_markers.npz')
            self.Xmean_global_markers = torch.from_numpy(preprocess_stats['Xmean']).float().to(device)
            self.Xstd_global_markers = torch.from_numpy(preprocess_stats['Xstd']).float().to(device)

            # local_markers_4chan (for motion infilling prior) preprocess_stats_infill_local_markers_4chan
            self.infill_stats = np.load('../preprocess_stats/preprocess_stats_infill_local_markers_4chan.npz')
            self.infill_stats_torch = {}
            self.infill_stats_torch['Xmean_local'] = torch.from_numpy(self.infill_stats['Xmean_local']).float().to(self.device)
            self.infill_stats_torch['Xstd_local'] = torch.from_numpy(self.infill_stats['Xstd_local']).float().to(self.device)
            self.infill_stats_torch['Xmean_global_xy'] = torch.from_numpy(self.infill_stats['Xmean_global_xy']).float().to(self.device)
            self.infill_stats_torch['Xstd_global_xy'] = torch.from_numpy(self.infill_stats['Xstd_global_xy']).float().to(self.device)
            self.infill_stats_torch['Xmean_global_r'] = torch.from_numpy(self.infill_stats['Xmean_global_r']).float().to(self.device)
            self.infill_stats_torch['Xstd_global_r'] = torch.from_numpy(self.infill_stats['Xstd_global_r']).float().to(self.device)


    def reset_loss_weights(self, loss_weight_dict):
            for key in loss_weight_dict:  # ex.: set self.contact_loss_weight = loss_weight_dict['loss_weight_dict']
                if hasattr(self, key):
                    weight_tensor = getattr(self, key)
                    if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                        weight_tensor = loss_weight_dict[key].clone().detach()
                    else:
                        weight_tensor = torch.tensor(loss_weight_dict[key],
                                                     dtype=weight_tensor.dtype,
                                                     device=weight_tensor.device)
                    setattr(self, key, weight_tensor)

    def forward(self, body_model, body_model_output, smplx_joints,
                camera, gt_joints, joints_conf, marker_mask,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                scan_tensor=None, scan_point_num=None,
                scene_v=None,
                opt_step=None,
                **kwargs):
        ################################ 2d keypoint loss ####################################
        projected_joints = camera(body_model_output.joints)  # [bs, 118, 2]

        # Calculate the weights for each joints
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)   # [bs, 118, 1]
        joint_diff = torch.abs(gt_joints - projected_joints)
        joint_loss = torch.mean(weights ** 2 * joint_diff) * self.data_weight


        ################################ pose/shape prior loss ####################################
        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() * self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2

        shape_loss = torch.sum(self.shape_prior(body_model_output.betas)) * self.shape_weight ** 2
        # Calculate the prior over the joint rotations. This a heuristic used to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]  # [bs, 63]
        angle_prior_loss = torch.sum(self.angle_prior(body_pose)) * self.bending_prior_weight ** 2

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss = torch.tensor(0.0).to(self.device)
        right_hand_prior_loss = torch.tensor(0.0).to(self.device)
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(self.left_hand_prior(body_model_output.left_hand_pose)) * \
                                   self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(self.right_hand_prior(body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2

        expression_loss = torch.tensor(0.0).to(self.device)
        jaw_prior_loss = torch.tensor(0.0).to(self.device)
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(body_model_output.expression)) * \
                self.expr_prior_weight ** 2

            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(self.jaw_prior(body_model_output.jaw_pose.mul(self.jaw_prior_weight)))


        ################################ self-penetration loss ####################################
        pen_loss = torch.tensor(0.0).to(self.device)
        # Calculate the loss due to interpenetration
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            batch_size = projected_joints.shape[0]  # 1
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3)  # [bs, 20908, 3, 3] xyz coordinates of 3 verts of each face

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles).detach()   # [bs, 2676224, 2]  vertex index of collision pairs?

            if self.tri_filtering_module is not None:
                for i in range(batch_size):
                    collision_idxs[i:i+1] = self.tri_filtering_module(collision_idxs[i:i+1])

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(self.coll_loss_weight * self.pen_distance(triangles, collision_idxs))

        ################################ loss between body and scan pointcloud from depth #############################
        s2m_dist = torch.tensor(0.0).to(self.device)
        m2s_dist = torch.tensor(0.0).to(self.device)
        # calculate the scan2mesh and mesh2scan loss from the sparse point cloud
        if (self.s2m or self.m2s) and (self.s2m_weight > 0 or self.m2s_weight > 0) and scan_tensor is not None:
            from psbody.mesh.visibility import visibility_compute
            from psbody.mesh import Mesh
            vertices_np = body_model_output.vertices.detach().cpu().numpy()   # [bs, 10475, 3]
            body_faces_np = body_model_faces.detach().cpu().numpy().reshape(-1, 3)  # [20908, 3]

            bs = vertices_np.shape[0]
            s2m_dist_list = []
            m2s_dist_list = []
            for i in range(bs):
                m = Mesh(v=vertices_np[i], f=body_faces_np)
                (vis, n_dot) = visibility_compute(v=m.v, f=m.f, cams=np.array([[0.0, 0.0, 0.0]]))  # vis: [1,10475], 0/1
                vis = vis.squeeze()

                cur_scan_tensor = scan_tensor[i:i+1][:, 0:scan_point_num[i]]  # [1, num_valid_pts, 3 ]
                if self.s2m and self.s2m_weight > 0 and vis.sum() > 0:
                    s2m_dist, _, _, _ = distChamfer(cur_scan_tensor,
                                                    body_model_output.vertices[:, np.where(vis > 0)[0], :])
                    s2m_dist = self.s2m_robustifier(s2m_dist.sqrt())  # [1, num_valid_pts]
                    s2m_dist_list.append(s2m_dist.mean())
                if self.m2s and self.m2s_weight > 0 and vis.sum() > 0:
                    _, m2s_dist, _, _ = distChamfer(cur_scan_tensor,
                                                    # body_mask: [1, 10475], true/false, mask for body without head
                                                    body_model_output.vertices[:,
                                                    np.where(np.logical_and(vis > 0, self.body_mask))[0], :])
                    m2s_dist = self.m2s_robustifier(m2s_dist.sqrt())
                    m2s_dist_list.append(m2s_dist.mean())

            s2m_dist = sum(s2m_dist_list) / len(s2m_dist_list) * self.s2m_weight
            m2s_dist = sum(m2s_dist_list) / len(m2s_dist_list) * self.m2s_weight



        ################################ to world coordinates ####################################
        if self.R is not None and self.t is not None:   # R/t: cam2world
            vertices = body_model_output.vertices  # [bs, 10475, 3]
            nv = vertices.shape[1]

            vertices_world = torch.matmul(self.R, vertices.permute(0, 2, 1)).permute(0, 2, 1) + self.t
            smplx_joints_world = torch.matmul(self.R, smplx_joints.permute(0, 2, 1)).permute(0, 2, 1) + self.t


        ################################ human-scene penetration loss ####################################
        sdf_penetration_loss = torch.tensor(0.0).to(self.device)
        if self.sdf_penetration and self.sdf_penetration_weight > 0:
            norm_vertices = (vertices_world - self.grid_min) / (self.grid_max - self.grid_min) * 2 - 1  # [bs, 10475, 3]
            body_sdf = F.grid_sample(self.sdf,
                                     norm_vertices[:, :, [2, 1, 0]].view(-1, nv, 1, 1, 3),
                                     padding_mode='border')  # [bs, 1, 10475, 1, 1]
            if body_sdf.lt(0).sum().item() < 1:
                sdf_penetration_loss = torch.tensor(0.0, dtype=joint_loss.dtype, device=joint_loss.device)
            else:
                sdf_penetration_loss = body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs()
                sdf_penetration_loss = self.sdf_penetration_weight * sdf_penetration_loss.pow(2).sum(dim=-1).sqrt().sum()



        ############################### contact friction loss ########################################
        loss_fric_tangent = torch.tensor(0.0).to(self.device)
        loss_fric_normal = torch.tensor(0.0).to(self.device)
        if self.use_friction:
            norm_vertices = (vertices_world - self.grid_min) / (self.grid_max - self.grid_min) * 2 - 1  # [bs, 10475, 3]
            body_sdf = F.grid_sample(self.sdf,
                                     norm_vertices[:, :, [2, 1, 0]].view(-1, nv, 1, 1, 3),
                                     padding_mode='border')  # [bs, 1, 10475, 1, 1]
            # grid_dim = self.sdf.shape[-1]

            # if body verts interpenetrate with floor, find foot verts' normals n
            # velocity v: v*n>=0, abs(v*n_t1)<=sigma, abs(v*n_t2)<=sigma
            vertices_world_fric = vertices_world[:, self.contact_fric_verts_ids, :]  # [bs, 194, 3]
            velocity_fric = vertices_world_fric[1:, :, :] - vertices_world_fric[0:-1, :, :]  # [bs-1, 194, 3]
            sdf_normals_xyplane = torch.tensor([0.0, 0.0, 1.0]).repeat(vertices_world.shape[0], len(self.contact_fric_verts_ids), 1).to(self.device)  # [bs, 194, 3]

            ############# constraints on velocity
            # body_sdf: [bs, 1, 10475, 1, 1]
            body_sdf_fric = body_sdf[0:-1, :, self.contact_fric_verts_ids, :, :].squeeze()  # [bs-1, 194]
            cur_contact_fric_verts_id = torch.where(body_sdf_fric<0.01)  # tuple of 2, (1)id in batchwise, (2)id in vertex-wise
            # if interpenetration happens
            if len(cur_contact_fric_verts_id[0]) > 0:
                sdf_normals_fric_contact = sdf_normals_xyplane[cur_contact_fric_verts_id]  # [N=# of cur foot contact verts, 3]
                velocity_fric_contact = velocity_fric[cur_contact_fric_verts_id]   # [N, 3]

                v_dot_n = torch.sum(velocity_fric_contact * sdf_normals_fric_contact, dim=-1)  # [N]
                v_dot_n_repeat = v_dot_n.repeat(3, 1).permute(1,0)  # [N, 3]
                velocity_foot_normal = v_dot_n_repeat * sdf_normals_fric_contact  # v_n = (dot(v,n))*n  [N, 3]
                velocity_foot_tangent = velocity_fric_contact - velocity_foot_normal  # v_t = v - v_n    [N, 3]

                # norm(v-(dot(v,n))*n) <= sigma
                goal_tangent = torch.norm(velocity_foot_tangent, dim=-1)   # should be <=sigma(a small value)
                if (goal_tangent - 0.0001).gt(0).sum().item() < 1:
                    loss_fric_tangent = torch.tensor(0.0).to(self.device)
                else:
                    loss_fric_tangent = goal_tangent[goal_tangent > 0.0001].abs().mean() * self.friction_tangent_weight

                # dot(v,n)>=0
                if v_dot_n.lt(0).sum().item() < 1:
                    loss_fric_normal = torch.tensor(0.0).to(self.device)
                else:
                    loss_fric_normal = v_dot_n[v_dot_n<0].abs().mean() * self.friction_normal_weight



        #################################### contact loss ####################################
        contact_loss = torch.tensor(0.0).to(self.device)
        if self.contact and self.contact_loss_weight >0:
            # select contact vertices
            contact_body_vertices = vertices_world[:, self.contact_verts_ids, :]  # [bs, 1121, 3]
            batched_scene_v = scene_v.repeat(vertices_world.shape[0], 1, 1)  # [bs, num_scene_verts, 3]
            contact_dist, _, idx1, _ = distChamfer(contact_body_vertices.contiguous(), batched_scene_v.contiguous())  # contact_dist: [bs, 1121], idx1: [bs, 1121]

            # contact_dist = self.contact_robustifier(contact_dist.sqrt())
            contact_dist = torch.sqrt(contact_dist + 1e-4) / (torch.sqrt(contact_dist + 1e-4) + 1.0)
            contact_loss = self.contact_loss_weight * contact_dist.mean()


        ############################# smooth acceleration term ##################################
        smooth_acc_loss = torch.tensor(0.0).to(self.device)
        if self.smooth_acc and self.smooth_acc_weight > 0:
            # penalize L2 norm of acceleration
            markers_smooth = body_model_output.vertices[:, self.smooth_marker_ids, :]
            markers_vel = markers_smooth[1:] - markers_smooth[0:-1]
            markers_acc = markers_vel[1:] - markers_vel[0:-1]  # [bs-2, 118, 3]
            smooth_acc_loss = torch.mean(markers_acc ** 2) * self.smooth_acc_weight


        ############################# smooth velocity term ##################################
        smooth_vel_loss = torch.tensor(0.0).to(self.device)
        if self.smooth_vel and self.smooth_vel_weight > 0:
            # penalize L2 norm of velocity
            markers_smooth = body_model_output.vertices[:, self.smooth_marker_ids, :]
            markers_vel = markers_smooth[1:] - markers_smooth[0:-1]
            smooth_vel_loss = torch.mean(markers_vel ** 2) * self.smooth_vel_weight


        ################################## motion infilling prior term #########################
        motion_infill_loss = torch.tensor(0.0).to(self.device)
        motion_infill_contact_loss = torch.tensor(0.0).to(self.device)
        if self.use_motion_infill_prior:
            markers = vertices_world[:, self.infill_marker_ids, :]  # [120, 67, 3]

            ##### transfrom to pelvis at origin, face y axis
            joints_3d = smplx_joints_world[:, 0:25]  # smplx body joints in world coordinate: [T, 25, 3]
            joints_frame0 = joints_3d[0].detach()  # [25, 3], joints of first frame

            x_axis = joints_frame0[2, :] - joints_frame0[1, :] # [3]
            x_axis[-1] = 0
            x_axis = x_axis / torch.norm(x_axis)
            z_axis = torch.tensor([0, 0, 1]).float().to(self.device)
            y_axis = torch.cross(z_axis, x_axis)
            y_axis = y_axis / torch.norm(y_axis)
            transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)  # [3, 3]
            joints_3d_normed = torch.matmul(joints_3d - joints_frame0[0], transf_rotmat)  # [120, 25, 3]
            markers_normed = torch.matmul(markers - joints_frame0[0], transf_rotmat)  # [120, 67, 3]

            ######## obtain binary contact labels for input
            # left_heel_id, right_heel_id = 16, 47, left_toe, right_toe = 30, 60
            # velocity criteria
            left_heel_vel = torch.norm((markers_normed[1:, 16:17] - markers_normed[0:-1, 16:17]) * 30, dim=-1)  # [119, 1]
            right_heel_vel = torch.norm((markers_normed[1:, 47:48] - markers_normed[0:-1, 47:48]) * 30, dim=-1)
            left_toe_vel = torch.norm((markers_normed[1:, 30:31] - markers_normed[0:-1, 30:31]) * 30, dim=-1)
            right_toe_vel = torch.norm((markers_normed[1:, 60:61] - markers_normed[0:-1, 60:61]) * 30, dim=-1)

            foot_markers_vel = torch.cat([left_heel_vel, right_heel_vel, left_toe_vel, right_toe_vel], dim=-1)  # [T-1, 4]

            is_contact = torch.abs(foot_markers_vel) < 0.22
            contact_lbls = torch.zeros([markers_normed.shape[0], 4]).to(self.device)  # all -1, [T, 4]
            contact_lbls[0:-1, :][is_contact == True] = 1.0  # 0/1, 1 if contact for first T-1 frames

            # z height criteria
            z_thres = torch.min(markers_normed[:, :, -1]) + 0.10
            foot_markers = torch.cat([markers_normed[:, 16:17], markers_normed[:, 47:48],
                                      markers_normed[:, 30:31], markers_normed[:, 60:61]], dim=-2)  # [T, 4, 3]
            thres_lbls = (foot_markers[:, :, 2] < z_thres).float()  # 0/1, [T, 4]

            # combine 2 criterias
            contact_lbls = contact_lbls * thres_lbls
            contact_lbls[-1, :] = thres_lbls[-1, :]  # last frame contact lbl: only look at z height
            contact_lbls = contact_lbls.detach().cpu().numpy()  # [T, 4]

            ######## get body representation
            if opt_step == 0:
                # if self.infill_body_mode == 'local_markers_4chan':
                cur_body = torch.cat([joints_3d_normed[:, 0:1], markers_normed], dim=1)  # first row: pelvis joint
                cur_body = cur_body.detach().cpu().numpy()  # numpy, [120, 1+67, 3]
                cur_body, rot_0_pivot = get_local_markers_4chan(cur_body, contact_lbls)  # numpy, [4, 119, (1+67)*3+4]
                clip_img = torch.from_numpy(cur_body).float().unsqueeze(0).to(self.device)  # tensor, [1, 4, 119, (1+67)*3+4]
                # normalize
                clip_img[:, 0] = (clip_img[:, 0] - self.infill_stats_torch['Xmean_local']) / self.infill_stats_torch['Xstd_local']
                clip_img[:, 1:3] = (clip_img[:, 1:3] - self.infill_stats_torch['Xmean_global_xy']) / self.infill_stats_torch['Xstd_global_xy']
                clip_img[:, 3] = (clip_img[:, 3] - self.infill_stats_torch['Xmean_global_r']) / self.infill_stats_torch['Xstd_global_r']
                clip_img = clip_img.permute(0, 1, 3, 2)  # [1, 4, d, 119]

                clip_img_input = clip_img.clone()  # [1, 4, d, 119]

                ####### mask input
                # marker_mask: [T(bs, 67)]
                marker_mask_flat = marker_mask.repeat_interleave(3).reshape([marker_mask.shape[0], -1])  # [120, 67*3]
                marker_mask_flat = marker_mask_flat.permute(1, 0).unsqueeze(0).unsqueeze(0)  # [1, 1, 67*3, 120]
                # mask contact lbls if foot marker is masked
                # is_mask_left: 1: keep left foot, 0: mask left foot
                is_mask_left = (marker_mask_flat[:, :, (16 * 3):(16 * 3 + 1), :] == 1) * \
                               (marker_mask_flat[:, :, (30 * 3):(30 * 3 + 1), :] == 1)  # [bs, 1, 1, 120]
                is_mask_right = (marker_mask_flat[:, :, (47 * 3):(47 * 3 + 1), :] == 1) * \
                                (marker_mask_flat[:, :, (60 * 3):(60 * 3 + 1), :] == 1)
                append_contact_mask = torch.cat([is_mask_left, is_mask_right, is_mask_left, is_mask_right], dim=-2)  # [bs, 1, 4, 120]
                append_contact_mask = append_contact_mask.float()

                T = clip_img_input.shape[-1]  # 119
                # if self.infill_body_mode in ['local_markers_4chan']:
                append_mask = torch.ones([1, 1, 3, T]).to(self.device)  # for pelvis joint, [1, 1, 3, 99]

                mask = torch.cat([append_mask, marker_mask_flat[:, :, :, 0:T], append_contact_mask[:, :, :, 0:T]], dim=-2)  # [bs=1, 1, 208, T]
                clip_img_input[:, 0:1] = clip_img_input[:, 0:1] * mask  # clip_img_input: [bs=1, 4, d=208, 119]


                if marker_mask.shape[0] * marker_mask.shape[1] > marker_mask.sum():
                    ####### input pad
                    p2d = (8, 8, 1, 1)
                    clip_img_input = F.pad(clip_img_input, p2d, 'reflect')

                    finetune_step_total = 60
                    ########## self-supervised finetune
                    self.motion_infill_model.load_state_dict(self.infill_pretrain_weights)
                    finetune_optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                                           itertools.chain(self.motion_infill_model.parameters())),
                                                    lr=3e-6)

                    for finetine_step in range(finetune_step_total):
                        self.motion_infill_model.train()
                        finetune_optimizer.zero_grad()

                        #### forward
                        clip_img_rec, z = self.motion_infill_model(clip_img_input)  # [1,4,d+2,T+16]

                        #### finetune loss
                        res_map = clip_img_rec[:, 0] - clip_img_input[:, 0]  # [bs=1, d+2, T+16]

                        mask_finetune = mask.detach()   # [bs=1, 1, d, T]
                        mask_finetune = F.pad(mask_finetune, p2d, 'reflect')[:, 0]  # pad, [1, d+2, T+16]
                        mask_finetune[:, -5:, :] = 0  # exclude input contact lbl, -5 for pad
                        res_map_finetune = res_map[mask_finetune == 1]

                        loss_finetune = res_map_finetune.abs().mean()
                        loss_finetune.backward()
                        finetune_optimizer.step()


                    ###### encoder forward
                    self.motion_infill_model.eval()
                    with torch.no_grad():
                        clip_img_rec, _ = self.motion_infill_model(clip_img_input)  # clip_img_rec: [1, 1, d, T]
                    ###### output inpad
                    clip_img_rec = clip_img_rec[:, :, 1:-1, 8:-8]  # [1, 1, d, T]
                    clip_img_input = clip_img_input[:, :, 1:-1, 8:-8]  # [1, 4, d, T]
                    clip_img_rec = clip_img_rec.detach()
                    clip_img_input = clip_img_input.detach()


                    # sigmoid to output contact lbl
                    self.contact_lbl_rec = F.sigmoid(clip_img_rec[0, 0, -4:, :].permute(1, 0))  # [T, 4]
                    self.contact_lbl_rec[self.contact_lbl_rec > 0.5] = 1.0
                    self.contact_lbl_rec[self.contact_lbl_rec <= 0.5] = 0.0
                    self.contact_lbl_rec = self.contact_lbl_rec.detach()


                    ###### get optimize target
                    # if self.infill_body_mode in ['local_markers_4chan']:
                    T = clip_img.shape[-1]  # 119

                    body_markers_rec = clip_img_rec[0, 0, 0:-4]  # [3+67*3, T], pelvis+markers
                    global_traj = torch.cat([clip_img_input[0, 1, 0:1], clip_img_input[0, 2, 0:1], clip_img_input[0, 3, 0:1]], dim=0)  # [3, T]

                    body_markers_rec = torch.cat([global_traj, body_markers_rec], dim=0)  # [3+3+67*3, T], global_traj + local(pelvis+marker)
                    body_markers_rec = body_markers_rec.permute(1, 0).reshape(T, -1, 3)  # [T, 1+1+67, 3]
                    body_markers_rec = body_markers_rec.detach().cpu().numpy()  # [T, 1+1+67, 3]

                    ######### normalize by preprocess stats
                    body_markers_rec = np.reshape(body_markers_rec, (T, -1))
                    body_markers_rec[:, 3:] = body_markers_rec[:, 3:] * self.infill_stats['Xstd_local'][0:-4] + \
                                             self.infill_stats['Xmean_local'][0:-4]
                    body_markers_rec[:, 0:2] = body_markers_rec[:, 0:2] * self.infill_stats['Xstd_global_xy'] + \
                                              self.infill_stats['Xmean_global_xy']
                    body_markers_rec[:, 2] = body_markers_rec[:, 2] * self.infill_stats['Xstd_global_r'] + \
                                            self.infill_stats['Xmean_global_r']

                    body_markers_rec = np.reshape(body_markers_rec, (T, -1, 3))  # [T, 1+1+67, 3], global_traj + local(pelvis+marker)
                    ######### back to old format: [T, 1+67+1, 3] reference + local(pelvis+markers) + global_traj
                    pad_0 = np.zeros([T, 1, 3])
                    body_markers_rec = np.concatenate([pad_0, body_markers_rec[:, 1:], body_markers_rec[:, 0:1]], axis=1)
                    body_markers_rec = reconstruct_global_body(body_markers_rec, rot_0_pivot)  # [T, 1+67, 3]  pelvis+markers (global position)
                    body_markers_rec = body_markers_rec[:, 1:, :]  # remove first pelvis joint

                    self.body_markers_rec = torch.from_numpy(body_markers_rec).float().to(self.device)

                    # put x-y plane from floor back to pelvis
                    min_z = markers_normed[:, :, 2].min().detach()
                    self.body_markers_rec[:, :, 2] = self.body_markers_rec[:, :, 2] + min_z

                    # back to prox world coordinate
                    self.body_markers_rec = torch.matmul(self.body_markers_rec, torch.inverse(transf_rotmat)) + joints_frame0[0]
                    self.body_markers_rec = self.body_markers_rec.detach()  # [T, 67, 3]

            # if with occlusion
            if marker_mask.shape[0] * marker_mask.shape[1] > marker_mask.sum():
                marker_mask_weight = marker_mask.repeat_interleave(3).reshape([marker_mask.shape[0], -1, 3]).detach()  # [T/bs, 67, 3]
                T = self.body_markers_rec.shape[0]
                diff = (self.body_markers_rec - markers[0:T]).abs() * (1 - marker_mask_weight[0:T])  # prox world coordinate
                diff = diff[diff > 0]
                motion_infill_loss = self.motion_infill_rec_weight * torch.mean(diff)

                ###### constraint on contact label
                # left_heel, right_heel, left_toe, right_toe, self.contact_lbl_rec    # [T, 4]
                left_heel_contact = self.contact_lbl_rec[:, 0]  # [119]
                right_heel_contact = self.contact_lbl_rec[:, 1]
                left_toe_contact = self.contact_lbl_rec[:, 2]
                right_toe_contact = self.contact_lbl_rec[:, 3]

                body_verts_opt_vel = (vertices_world[1:] - vertices_world[0:-1]) * 30  # [119, 10475, 3]

                left_heel_verts_vel = body_verts_opt_vel[:, self.left_heel_verts_id, :][left_heel_contact == 1]  # [t, n, 3]
                left_heel_verts_vel = torch.norm(left_heel_verts_vel, dim=-1)  # [t, n]

                right_heel_verts_vel = body_verts_opt_vel[:, self.right_heel_verts_id, :][right_heel_contact == 1]
                right_heel_verts_vel = torch.norm(right_heel_verts_vel, dim=-1)

                left_toe_verts_vel = body_verts_opt_vel[:, self.left_toe_verts_id, :][left_toe_contact == 1]
                left_toe_verts_vel = torch.norm(left_toe_verts_vel, dim=-1)

                right_toe_verts_vel = body_verts_opt_vel[:, self.right_toe_verts_id, :][right_toe_contact == 1]
                right_toe_verts_vel = torch.norm(right_toe_verts_vel, dim=-1)

                vel_thres = 0.1
                loss_contact_vel_left_heel = torch.tensor(0.0).to(self.device)
                if (left_heel_verts_vel - vel_thres).gt(0).sum().item() >= 1:
                    loss_contact_vel_left_heel = left_heel_verts_vel[left_heel_verts_vel > vel_thres].abs().mean()

                loss_contact_vel_right_heel = torch.tensor(0.0).to(self.device)
                if (right_heel_verts_vel - vel_thres).gt(0).sum().item() >= 1:
                    loss_contact_vel_right_heel = right_heel_verts_vel[
                        right_heel_verts_vel > vel_thres].abs().mean()

                loss_contact_vel_left_toe = torch.tensor(0.0).to(self.device)
                if (left_toe_verts_vel - vel_thres).gt(0).sum().item() >= 1:
                    loss_contact_vel_left_toe = left_toe_verts_vel[left_toe_verts_vel > vel_thres].abs().mean()

                loss_contact_vel_right_toe = torch.tensor(0.0).to(self.device)
                if (right_toe_verts_vel - vel_thres).gt(0).sum().item() >= 1:
                    loss_contact_vel_right_toe = right_toe_verts_vel[right_toe_verts_vel > vel_thres].abs().mean()

                motion_infill_contact_loss = loss_contact_vel_left_heel + loss_contact_vel_right_heel + \
                                             loss_contact_vel_left_toe + loss_contact_vel_right_toe
                motion_infill_contact_loss = self.motion_infill_contact_weight * motion_infill_contact_loss



        ############################# motion smoothness prior term #############################
        motion_prior_smooth_loss = torch.tensor(0.0).to(self.device)
        if self.use_motion_smooth_prior:
            markers_smooth = vertices_world[:, self.smooth_marker_ids, :]  # [T, 67, 3]
            joints_3d = smplx_joints_world[:, 0:75]  # in world coordinate: [T, 75, 3]


            ##### transfrom to pelvis at origin, face y axis
            joints_frame0 = joints_3d[0].detach()  # [25, 3], joints of first frame
            x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]
            x_axis[-1] = 0
            x_axis = x_axis / torch.norm(x_axis)
            z_axis = torch.tensor([0, 0, 1]).float().to(self.device)
            y_axis = torch.cross(z_axis, x_axis)
            y_axis = y_axis / torch.norm(y_axis)
            transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)  # [3, 3]
            joints_3d = torch.matmul(joints_3d - joints_frame0[0], transf_rotmat)  # [T(/bs), 25, 3]
            markers_frame0 = markers_smooth[0].detach()
            markers_smooth = torch.matmul(markers_smooth - markers_frame0[0], transf_rotmat)  # [T(/bs), 67, 3]
            clip_img = markers_smooth.reshape(markers_smooth.shape[0], -1).unsqueeze(0)
            clip_img = (clip_img - self.Xmean_global_markers) / self.Xstd_global_markers
            clip_img = clip_img.permute(0, 2, 1).unsqueeze(1)  # [1, 1, d, T]

            ####### input res, encoder forward
            T = clip_img.shape[-1]
            clip_img_v = clip_img[:, :, :, 1:] - clip_img[:, :, :, 0:-1]
            ### input padding
            p2d = (8, 8, 1, 1)
            clip_img_v = F.pad(clip_img_v, p2d, 'reflect')
            ### forward
            motion_z, _, _, _, _, _ = self.motion_smooth_model(clip_img_v)


            ####### constraints on latent z
            motion_z_v = motion_z[:, :, :, 1:] - motion_z[:, :, :, 0:-1]
            motion_prior_smooth_loss = torch.mean(motion_z_v ** 2) * self.motion_prior_smooth_weight



        ################################## total loss #####################################
        total_loss = (joint_loss + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss + m2s_dist + s2m_dist
                      + sdf_penetration_loss + contact_loss +
                      smooth_acc_loss + smooth_vel_loss +
                      motion_prior_smooth_loss +
                      loss_fric_tangent + loss_fric_normal +
                      motion_infill_loss + motion_infill_contact_loss)

        # return total_loss
        loss_dict = {}
        loss_dict['total_loss'] = total_loss
        loss_dict['joint_loss'] = joint_loss
        loss_dict['s2m_dist'] = s2m_dist
        loss_dict['m2s_dist'] = m2s_dist
        loss_dict['self_penetration_loss'] = pen_loss
        loss_dict['sdf_penetration_loss'] = sdf_penetration_loss
        loss_dict['contact_loss'] = contact_loss
        loss_dict['smooth_acc_loss'] = smooth_acc_loss
        loss_dict['smooth_vel_loss'] = smooth_vel_loss
        loss_dict['motion_prior_smooth_loss'] = motion_prior_smooth_loss
        loss_dict['loss_fric_tangent'] = loss_fric_tangent
        loss_dict['loss_fric_normal'] = loss_fric_normal
        loss_dict['motion_infill_loss'] = motion_infill_loss
        loss_dict['motion_infill_contact_loss'] = motion_infill_contact_loss
        return loss_dict



class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2,
                 camera_mode='moving',
                 dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype
        self.camera_mode = camera_mode  # 'fixed'

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key], dtype=weight_tensor.dtype, device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, body_model,
                **kwargs):
        # camera_mode == 'fixed': optimize params: body_model.transl, body_model.global_orient
        projected_joints = camera(body_model_output.joints)  # [bs, 118, 2], Convert points to homogeneous coordinates, in openpose topology?

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs), 2)   # [bs, 4, 2]
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2  # joint error for joint [2,5,9,12] between detected keypoint and current model joint

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not None):
            if self.camera_mode == 'moving':
                depth_loss = self.depth_loss_weight ** 2 * torch.sum((camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))
            elif self.camera_mode == 'fixed':
                depth_loss = self.depth_loss_weight ** 2 * torch.sum((body_model.transl[:, 2] - self.trans_estimation[:, 2]).pow(2))

        # return joint_loss + depth_loss
        loss_dict = {}
        loss_dict['total_loss'] = joint_loss + depth_loss
        loss_dict['joint_loss'] = joint_loss
        loss_dict['depth_loss'] = depth_loss
        return loss_dict
