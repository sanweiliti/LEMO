import smplx
import numpy as np
import os
import torch
import open3d as o3d
from tqdm import tqdm
import argparse
import torch.optim as optim
import itertools
from tensorboardX import SummaryWriter
from human_body_prior.tools.model_loader import load_vposer

from loader.optimize_loader_amass_new import TrainLoader
from models.AE import AE
from models.AE_sep import Enc
from utils.utils import *

parser = argparse.ArgumentParser()

# path to amass and smplx body model
parser.add_argument('--amass_dir', type=str, default='/local/home/szhang/AMASS/amass', help='path to AMASS dataset')
parser.add_argument('--body_model_path', type=str, default='/mnt/hdd/PROX/body_models', help='path to smplx body models')

# settings for body representation
parser.add_argument("--clip_seconds", default=4, type=int, help='length (seconds) of each motion sequence')
# settings for motion infilling prior
parser.add_argument('--body_mode', type=str, default='local_markers_4chan',
                    choices=['local_markers', 'local_markers_4chan'], help='which body representation to use')
parser.add_argument('--infill_model_path', type=str, default='runs/59547/AE_last_model.pkl', help='path to pretrained infilling prior')
parser.add_argument("--conv_k", default=3, type=int, help='conv kernel size')
# settings for motion smooth prior
parser.add_argument('--smooth_model_path', type=str, default='runs/15217/Enc_last_model.pkl', help='path to pretrained smoothness prior')


parser.add_argument("--start", default=0, type=int, help='from which sequence to start')
parser.add_argument("--end", default=100, type=int, help='until which sequence to end')
parser.add_argument("--step", default=20, type=int, help='optimize 1 sequence every [step] sequences')
parser.add_argument('--dataset_name', type=str, default='TotalCapture', help='which dataset in AMASS to optimize')
# amass_train_datasets = ['HumanEva', 'MPI_HDM05', 'MPI_mosh', 'Transitions_mocap',
#                         'ACCAD', 'BMLhandball', 'BMLmovi', 'BioMotionLab_NTroje', 'CMU',
#                         'DFaust_67', 'Eyes_Japan_Dataset', 'MPI_Limits']
# amass_test_datasets = ['TCD_handMocap', 'TotalCapture', 'SFU']

parser.add_argument('--perframe_res_dir', type=str, default='res_opt_amass_perframe', help='path to body params optimized per frame')
parser.add_argument('--save_dir', type=str, default='res_opt_amass_temp', help='path to save optimized body params')

parser.add_argument('--weight_loss_rec_markers', type=float, default=1.0, help='weight for marker reconstruction loss (motion infilling prior)')
parser.add_argument('--weight_loss_contact_vel', type=float, default=0.03, help='weight for foot contact friction loss')
parser.add_argument('--weight_loss_smooth', type=float, default=1e6, help='weight for smoothness loss (motion smoothness prior)')
parser.add_argument('--weight_loss_vposer', type=float, default=0.02, help='weight for vposer prior loss')
parser.add_argument('--weight_loss_shape', type=float, default=0.01, help='weight for body shape prior loss')
parser.add_argument('--weight_loss_hand', type=float, default=0.01, help='weight for hand pose prior loss')

args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('gpu id:', torch.cuda.current_device())


def optimize():
    # amass_dir = '/local/home/szhang/AMASS/amass'
    # body_model_path = '/mnt/hdd/PROX/body_models'

    smplx_model_path = os.path.join(args.body_model_path, 'smplx_model')
    vposer_model_path = os.path.join(args.body_model_path, 'vposer_v1_0')

    vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
    vposer_model = vposer_model.to(device)
    print('[INFO] vposer model loaded')

    smplx_model_male = smplx.create(smplx_model_path, model_type='smplx', gender='male', ext='npz',
                                    num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                    create_betas=True, create_left_hand_pose=True,
                                    create_right_hand_pose=True, create_expression=True, create_jaw_pose=True,
                                    create_leye_pose=True, create_reye_pose=True, create_transl=True,
                                    batch_size=args.clip_seconds*30-1
                                    ).to(device)

    smplx_model_female = smplx.create(smplx_model_path, model_type='smplx', gender='female', ext='npz',
                                      num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                      create_betas=True, create_left_hand_pose=True,
                                      create_right_hand_pose=True, create_expression=True, create_jaw_pose=True,
                                      create_leye_pose=True, create_reye_pose=True, create_transl=True,
                                      batch_size=args.clip_seconds*30-1
                                      ).to(device)  # note amass do not use pca for hands

    # stats for infill prior
    stats = np.load('preprocess_stats/preprocess_stats_infill_{}.npz'.format(args.body_mode))

    # stats for smooth prior
    preprocess_stats_smooth = np.load('preprocess_stats/preprocess_stats_smooth_withHand_global_markers.npz')
    Xmean_global_markers = torch.from_numpy(preprocess_stats_smooth['Xmean']).float().to(device)
    Xstd_global_markers = torch.from_numpy(preprocess_stats_smooth['Xstd']).float().to(device)


    body_segments_dir = 'body_segments'
    with open(os.path.join(body_segments_dir, 'L_Leg.json'), 'r') as f:
        data = json.load(f)
        left_foot_verts_id = np.asarray(list(set(data["verts_ind"])))
    left_heel_verts_id = np.load('foot_verts_id/left_heel_verts_id.npy')
    left_toe_verts_id = np.load('foot_verts_id/left_toe_verts_id.npy')
    left_heel_verts_id = left_foot_verts_id[left_heel_verts_id]
    left_toe_verts_id = left_foot_verts_id[left_toe_verts_id]

    with open(os.path.join(body_segments_dir, 'R_Leg.json'), 'r') as f:
        data = json.load(f)
        right_foot_verts_id = np.asarray(list(set(data["verts_ind"])))
    right_heel_verts_id = np.load('foot_verts_id/right_heel_verts_id.npy')
    right_toe_verts_id = np.load('foot_verts_id/right_toe_verts_id.npy')
    right_heel_verts_id = right_foot_verts_id[right_heel_verts_id]
    right_toe_verts_id = right_foot_verts_id[right_toe_verts_id]


    ################################### set dataloaders ######################################
    print('[INFO] reading test data from datasets {}...'.format(args.dataset_name))
    dataset = TrainLoader(clip_seconds=args.clip_seconds, clip_fps=30, normalize=True,
                          split='test', mode=args.body_mode)
    dataset.read_data([args.dataset_name], args.amass_dir)
    dataset.create_body_repr(with_hand=False, smplx_model_path=smplx_model_path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                                             num_workers=0, drop_last=True)


    ################################## set tes configs ######################################
    # load infill prior model
    if args.body_mode == 'local_markers':
        in_channel = 1
    elif args.body_mode == 'local_markers_4chan':
        in_channel = 4
    infill_model = AE(downsample=True, in_channel=in_channel, kernel=args.conv_k).to(device)
    weights = torch.load(args.infill_model_path, map_location=lambda storage, loc: storage)
    infill_model.load_state_dict(weights)

    # load smooth prior model
    smooth_encoder = Enc(downsample=False, z_channel=64).to(device)
    smooth_weights = torch.load(args.smooth_model_path, map_location=lambda storage, loc: storage)
    smooth_encoder.load_state_dict(smooth_weights)
    smooth_encoder.eval()
    for param in smooth_encoder.parameters():
        param.requires_grad = False


    ###################### NN: get infilled sequences with self-supervised finetune ###################
    clip_img_rec_list = []
    clip_img_list = []
    rot_0_pivot_list = []
    beta_list = []
    gender_list = []

    finetine_step_total = 60
    print('[INFO] inference stage (with self-supervised finetuning)')
    for step, data in tqdm(enumerate(dataloader)):
        if step == args.end:
            break
        [clip_img, smplx_beta, gender,
         rot_0_pivot, _, _] = [item.to(device) for item in data]
        T = clip_img.shape[-1]
        infill_model.load_state_dict(weights)

        finetune_optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                               itertools.chain(infill_model.parameters())),
                                        lr=3e-6)  # to set

        ###### mask input
        clip_img_input = clip_img.clone()  # [bs, 1/4, d, T]
        bs = clip_img.shape[0]

        # mask out upper body markers
        mask_marker_id = np.array(
            [14, 15, 18, 19, 29, 2, 20, 21, 30, 25, 16, 45, 46, 48, 49, 59, 32, 50, 51, 55, 60, 47])
        mask_row_id1 = mask_marker_id * 3
        if args.body_mode in ['local_markers']:  # for global traj and pelvis joint
            mask_row_id1 = mask_row_id1 + 3 + 3
        if args.body_mode in ['local_markers_4chan']:  # for pelvis joint
            mask_row_id1 = mask_row_id1 + 3
        mask_row_id2 = mask_row_id1 + 1
        mask_row_id3 = mask_row_id2 + 1
        clip_img_input[:, 0, mask_row_id1, :] = 0.
        clip_img_input[:, 0, mask_row_id2, :] = 0.
        clip_img_input[:, 0, mask_row_id3, :] = 0.
        # mask contact lbls/distance if foot joint is masked
        clip_img_input[:, 0, -4:, :] = 0.

        # input padding
        p2d = (8, 8, 1, 1)
        clip_img_input = F.pad(clip_img_input, p2d, 'reflect')  # [bs, 1, 77, 119+8+8]

        ########## finetune
        for finetine_step in range(finetine_step_total):
            infill_model.train()
            finetune_optimizer.zero_grad()

            #### forward
            clip_img_rec, z = infill_model(clip_img_input)

            #### finetune loss
            res_map = clip_img_rec[:, 0] - clip_img_input[:, 0]  # [1, d, T]

            mask_row = np.concatenate([mask_row_id1, mask_row_id2, mask_row_id3]) + 1  # for pad at 1st row
            all_row = list(range(clip_img_rec.shape[-2]))
            upper_body_row = list(set(all_row) - set(mask_row))
            res_map_finetune = res_map[:, upper_body_row][:, 0:-5]

            loss = res_map_finetune.abs().mean()
            loss.backward()
            finetune_optimizer.step()

        #### forward
        infill_model.eval()
        with torch.no_grad():
            clip_img_rec, z = infill_model(clip_img_input)
            clip_img_rec = clip_img_rec[:, :, 1:-1, 8:-8]  # [bs, 1, d, T]

        clip_img_rec_list.append(clip_img_rec)
        clip_img_list.append(clip_img)
        rot_0_pivot_list.append(rot_0_pivot)
        beta_list.append(smplx_beta)
        gender_list.append(gender)


    clip_img_rec_list = torch.cat(clip_img_rec_list, dim=0)  # [n, 1, d, T]  n clips
    clip_img_list = torch.cat(clip_img_list, dim=0)  # [n, 1/4, d, T]
    rot_0_pivot_list = torch.cat(rot_0_pivot_list, dim=0)  # [n, 1]
    beta_list = torch.cat(beta_list, dim=0)  # [n, 10]
    gender_list = torch.cat(gender_list, dim=0)  # [n, 1]


    #################################### get infilled sequences ############################
    LIMBS = LIMBS_MARKER_SSM2

    color_input = np.zeros([len(LIMBS), 3])
    color_input[:, 0] = 1.0  # red
    color_rec = np.zeros([len(LIMBS), 3])
    color_rec[:, 2] = 1.0  # blue

    save_folder = os.path.join(args.save_dir, args.dataset_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save('{}/gender_list.npy'.format(save_folder), gender_list.detach().cpu().numpy())


    with open('loader/SSM2.json') as f:
        infill_marker_ids = list(json.load(f)['markersets'][0]['indices'].values())

    with open('loader/SSM2_withhand.json') as f:
        smooth_marker_ids = list(json.load(f)['markersets'][0]['indices'].values())

    print('[INFO] temporal optimizing ...')
    for i in range(args.start, args.end, args.step):
        print('current clip:', i)
        init_path = os.path.join(args.perframe_res_dir, args.dataset_name, 'body_params_opt_clip_{}.npy'.format(i))
        init_params = np.load(init_path)  # [T, 72]

        clip_img_rec = clip_img_rec_list[i, 0]  # [d, T]
        clip_img_input = clip_img_list[i]   # [1/4, d, T], to give global traj
        T = clip_img_rec.shape[-1]

        rot_0_pivot = rot_0_pivot_list[i].detach().cpu().numpy()

        # beta_gt = beta_list[i]  # [10]
        gender = gender_list[i]
        if gender == 0:
            gender = 'female'
            smplx_model = smplx_model_female
        elif gender == 1:
            gender = 'male'
            smplx_model = smplx_model_male


        ######### get contact labels, infilled global markers
        contact_lbl_rec = F.sigmoid(clip_img_rec[-4:, :].permute(1, 0))  # [T, 4]
        contact_lbl_rec[contact_lbl_rec > 0.5] = 1.0
        contact_lbl_rec[contact_lbl_rec <= 0.5] = 0.0
        np.save('{}/contact_lbl_rec_clip_{}.npy'.format(save_folder, i), contact_lbl_rec.detach().cpu().numpy())


        if args.body_mode == 'local_markers_4chan':
            body_joints_rec = clip_img_rec[0:-4, :]  # [3+67*3, T], pelvis+markers
            body_joints_input = clip_img_input[0, 0:-4, :]
            global_traj = torch.cat([clip_img_input[1, 0:1], clip_img_input[2, 0:1], clip_img_input[3, 0:1]], dim=0)  # [3, T]

        if args.body_mode == 'local_markers':
            body_joints_rec = clip_img_rec[3:-4, :]  # [3+67*3, T], pelvis+markers
            body_joints_input = clip_img_input[0, 3:-4, :]
            global_traj = clip_img_input[0:3, :]  # [3, T]

        body_joints_rec = torch.cat([global_traj, body_joints_rec], dim=0)  # [3+3+67*3, T], global_traj + local(pelvis+marker)
        body_joints_rec = body_joints_rec.permute(1, 0).reshape(T, -1, 3)  # [T, 1+1+67, 3]
        body_joints_rec = body_joints_rec.detach().cpu().numpy()

        body_joints_input = torch.cat([global_traj, body_joints_input], dim=0)  # [3+3+67*3, T], global_traj + local(pelvis+marker)
        body_joints_input = body_joints_input.permute(1, 0).reshape(T, -1, 3)  # [T, 1+1+67, 3]
        body_joints_input = body_joints_input.detach().cpu().numpy()

        ######### normalize by preprocess stats
        body_joints_rec = np.reshape(body_joints_rec, (T, -1))
        body_joints_input = np.reshape(body_joints_input, (T, -1))

        if args.body_mode == 'local_markers':
            body_joints_rec = body_joints_rec * stats['Xstd'][0:-4] + stats['Xmean'][0:-4]
            body_joints_input = body_joints_input * stats['Xstd'][0:-4] + stats['Xmean'][0:-4]

        if args.body_mode == 'local_markers_4chan':
            body_joints_rec[:, 3:] = body_joints_rec[:, 3:] * stats['Xstd_local'][0:-4] + stats['Xmean_local'][0:-4]
            body_joints_rec[:, 0:2] = body_joints_rec[:, 0:2] * stats['Xstd_global_xy'] + stats['Xmean_global_xy']
            body_joints_rec[:, 2] = body_joints_rec[:, 2] * stats['Xstd_global_r'] + stats['Xmean_global_r']

            body_joints_input[:, 3:] = body_joints_input[:, 3:] * stats['Xstd_local'][0:-4] + stats['Xmean_local'][0:-4]
            body_joints_input[:, 0:2] = body_joints_input[:, 0:2] * stats['Xstd_global_xy'] + stats['Xmean_global_xy']
            body_joints_input[:, 2] = body_joints_input[:, 2] * stats['Xstd_global_r'] + stats['Xmean_global_r']

        body_joints_rec = np.reshape(body_joints_rec, (T, -1, 3))  # [T, 1+1+67, 3], global_traj + local(pelvis+marker)
        body_joints_input = np.reshape(body_joints_input, (T, -1, 3))

        ######### back to old format: [T, 1+67+1, 3] reference + local(pelvis+markers) + global_traj
        pad_0 = np.zeros([T, 1, 3])
        body_joints_rec = np.concatenate([pad_0, body_joints_rec[:, 1:], body_joints_rec[:, 0:1]], axis=1)
        body_joints_rec = reconstruct_global_body(body_joints_rec, rot_0_pivot)  # [T, 1+67, 3]  pelvis+markers (global position)
        body_joints_rec = body_joints_rec[:, 1:, :]  # remove first pelvis joint

        body_joints_input = np.concatenate([pad_0, body_joints_input[:, 1:], body_joints_input[:, 0:1]], axis=1)
        body_joints_input = reconstruct_global_body(body_joints_input, rot_0_pivot)  # [T, 1+67, 3]  pelvis+markers (global position)
        body_joints_input = body_joints_input[:, 1:, :]


        #################################### optimize ############################################3
        markers_rec_t = torch.from_numpy(body_joints_rec).float().to(device)  # np, [T, 67, 3]

        ############### init opt params ##################
        transl_opt_t = torch.from_numpy(init_params[:, 0:3]).float().to(device)
        rot_opt_t = torch.from_numpy(init_params[:, 3:6]).float().to(device)
        rot_6d_opt_t = convert_to_6D_all(rot_opt_t)
        shape_t = torch.from_numpy(init_params[:, 6:16]).float().to(device)  # fixed
        other_params_opt_t = torch.from_numpy(init_params[:, 16:]).float().to(device)

        transl_opt_t.requires_grad = True
        rot_6d_opt_t.requires_grad = True
        other_params_opt_t.requires_grad = True

        final_params = [transl_opt_t, rot_6d_opt_t, other_params_opt_t]

        init_lr = 0.01
        optimizer = optim.Adam(final_params, lr=init_lr)

        # fitting iteration
        total_steps = 100
        for step in range(total_steps):
            if step > 60:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.005
            optimizer.zero_grad()

            body_params_opt_t = torch.cat([transl_opt_t, rot_6d_opt_t, shape_t, other_params_opt_t], dim=-1)  # [T, 75]
            body_params_opt_t_72 = convert_to_3D_rot(body_params_opt_t)  # tensor, [T, 72]
            body_verts_opt_t = gen_body_mesh_v1(body_params=body_params_opt_t_72, smplx_model=smplx_model,
                                                vposer_model=vposer_model)  # tensor [T, 10475, 3]
            markers_opt_t = body_verts_opt_t[:, infill_marker_ids, :]  # [T, 67, 3]



            ####################### smooth prios, global markers with hands
            joints_3d = gen_body_joints_v1(body_params=body_params_opt_t_72, smplx_model=smplx_model,
                                           vposer_model=vposer_model)  # [T, 25, 3]
            markers_smooth = body_verts_opt_t[:, smooth_marker_ids, :]  # [T, 67+..., 3]
            # transfrom to pelvis at origin, face y axis
            joints_frame0 = joints_3d[0].detach()  # [25, 3], joints of first frame
            x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]
            x_axis[-1] = 0
            x_axis = x_axis / torch.norm(x_axis)
            z_axis = torch.tensor([0, 0, 1]).float().to(device)
            y_axis = torch.cross(z_axis, x_axis)
            y_axis = y_axis / torch.norm(y_axis)
            transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)  # [3, 3]
            markers_frame0 = markers_smooth[0].detach()
            global_markers_smooth_opt = torch.matmul(markers_smooth - markers_frame0[0], transf_rotmat)  # [T(/bs), 67, 3]

            clip_img_smooth = global_markers_smooth_opt.reshape(global_markers_smooth_opt.shape[0], -1).unsqueeze(0)  # [1, T, d]
            clip_img_smooth = (clip_img_smooth - Xmean_global_markers) / Xstd_global_markers
            clip_img_smooth = clip_img_smooth.permute(0, 2, 1).unsqueeze(1)  # [1, 1, d, T]

            # input res, encoder forward
            clip_img_smooth_v = clip_img_smooth[:, :, :, 1:] - clip_img_smooth[:, :, :, 0:-1]
            ### input padding
            p2d = (8, 8, 1, 1)
            clip_img_smooth_v = F.pad(clip_img_smooth_v, p2d, 'reflect')
            ### forward
            motion_z, _, _, _, _, _ = smooth_encoder(clip_img_smooth_v)
            motion_z_v = motion_z[:, :, :, 1:] - motion_z[:, :, :, 0:-1]
            loss_smooth = torch.mean(motion_z_v ** 2)


            ### marker rec loss
            loss_marker = F.l1_loss(markers_opt_t, markers_rec_t)
            ### vposer loss
            vposer_pose = body_params_opt_t_72[:, 16:48]
            loss_vposer = torch.mean(vposer_pose ** 2)
            ### shape prior loss
            shape_params = body_params_opt_t_72[:, 6:16]
            loss_shape = torch.mean(shape_params ** 2)
            ### hand pose prior loss
            hand_params = body_params_opt_t_72[:, 48:]
            loss_hand = torch.mean(hand_params ** 2)

            ### contact friction loss for foot
            loss_contact_vel = torch.tensor(0.0).to(device)
            if args.weight_loss_contact_vel > 0:
                left_heel_contact = contact_lbl_rec[:, 0]  # [T]
                right_heel_contact = contact_lbl_rec[:, 1]
                left_toe_contact = contact_lbl_rec[:, 2]
                right_toe_contact = contact_lbl_rec[:, 3]

                body_verts_opt_vel = (body_verts_opt_t[1:] - body_verts_opt_t[0:-1]) * 30  # [T-1, 10475, 3]
                # x/y vel
                left_heel_verts_vel = body_verts_opt_vel[:, left_heel_verts_id, :][left_heel_contact[0:-1] == 1]
                left_heel_verts_vel = torch.norm(left_heel_verts_vel, dim=-1)  # [t, n]

                right_heel_verts_vel = body_verts_opt_vel[:, right_heel_verts_id, :][right_heel_contact[0:-1] == 1]
                right_heel_verts_vel = torch.norm(right_heel_verts_vel, dim=-1)

                left_toe_verts_vel = body_verts_opt_vel[:, left_toe_verts_id, :][left_toe_contact[0:-1] == 1]
                left_toe_verts_vel = torch.norm(left_toe_verts_vel, dim=-1)

                right_toe_verts_vel = body_verts_opt_vel[:, right_toe_verts_id, :][right_toe_contact[0:-1] == 1]
                right_toe_verts_vel = torch.norm(right_toe_verts_vel, dim=-1)


                vel_thres = 0.1
                loss_contact_vel_left_heel = torch.tensor(0.0).to(device)
                if (left_heel_verts_vel - vel_thres).gt(0).sum().item() >= 1:
                    loss_contact_vel_left_heel = left_heel_verts_vel[left_heel_verts_vel > vel_thres].abs().mean()

                loss_contact_vel_right_heel = torch.tensor(0.0).to(device)
                if (right_heel_verts_vel - vel_thres).gt(0).sum().item() >= 1:
                    loss_contact_vel_right_heel = right_heel_verts_vel[right_heel_verts_vel > vel_thres].abs().mean()

                loss_contact_vel_left_toe = torch.tensor(0.0).to(device)
                if (left_toe_verts_vel - vel_thres).gt(0).sum().item() >= 1:
                    loss_contact_vel_left_toe = left_toe_verts_vel[left_toe_verts_vel > vel_thres].abs().mean()

                loss_contact_vel_right_toe = torch.tensor(0.0).to(device)
                if (right_toe_verts_vel - vel_thres).gt(0).sum().item() >= 1:
                    loss_contact_vel_right_toe = right_toe_verts_vel[right_toe_verts_vel > vel_thres].abs().mean()

                loss_contact_vel = loss_contact_vel_left_heel + loss_contact_vel_right_heel + \
                                   loss_contact_vel_left_toe + loss_contact_vel_right_toe

            loss = args.weight_loss_rec_markers * loss_marker + \
                   args.weight_loss_vposer * loss_vposer + \
                   args.weight_loss_shape * loss_shape + args.weight_loss_hand * loss_hand + \
                   args.weight_loss_contact_vel * loss_contact_vel + \
                   args.weight_loss_smooth * loss_smooth
            loss.backward(retain_graph=True)
            optimizer.step()

        body_params_opt_t_72 = body_params_opt_t_72.detach().cpu().numpy()  # [T, 72]
        np.save('{}/body_params_opt_clip_{}.npy'.format(save_folder, i), body_params_opt_t_72)



if __name__ == '__main__':
    optimize()














