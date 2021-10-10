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
parser.add_argument("--conv_k", default=3, type=int, help='conv kernel size')
parser.add_argument('--infill_model_path', type=str, default='runs/59547/AE_last_model.pkl', help='path to pretrained infilling prior')

parser.add_argument("--start", default=0, type=int, help='from which sequence to start')
parser.add_argument("--end", default=100, type=int, help='until which sequence to end')
parser.add_argument("--step", default=20, type=int, help='optimize 1 sequence every [step] sequences')
parser.add_argument('--dataset_name', type=str, default='TotalCapture', help='which dataset in AMASS to optimize')
# amass_train_datasets = ['HumanEva', 'MPI_HDM05', 'MPI_mosh', 'Transitions_mocap',
#                         'ACCAD', 'BMLhandball', 'BMLmovi', 'BioMotionLab_NTroje', 'CMU',
#                         'DFaust_67', 'Eyes_Japan_Dataset', 'MPI_Limits']
# amass_test_datasets = ['TCD_handMocap', 'TotalCapture', 'SFU']

parser.add_argument('--save_dir', type=str, default='res_opt_amass_perframe', help='path to save optimized body params')

parser.add_argument('--weight_loss_rec_markers', type=float, default=1.0, help='weight for marker reconstruction loss (motion infilling prior)')
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
                                    batch_size=1
                                    ).to(device)

    smplx_model_female = smplx.create(smplx_model_path, model_type='smplx', gender='female', ext='npz',
                                      num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                      create_betas=True, create_left_hand_pose=True,
                                      create_right_hand_pose=True, create_expression=True, create_jaw_pose=True,
                                      create_leye_pose=True, create_reye_pose=True, create_transl=True,
                                      batch_size=1
                                      ).to(device)

    stats = np.load('preprocess_stats/preprocess_stats_infill_{}.npz'.format(args.body_mode))

    with open('loader/SSM2.json') as f:
        marker_ids = list(json.load(f)['markersets'][0]['indices'].values())

    ################################### set dataloaders ######################################
    print('[INFO] reading test data from datasets {}...'.format(args.dataset_name))
    dataset = TrainLoader(clip_seconds=args.clip_seconds, clip_fps=30, normalize=True,
                          split='test', mode=args.body_mode)
    dataset.read_data([args.dataset_name], args.amass_dir)
    dataset.create_body_repr(with_hand=False, smplx_model_path=smplx_model_path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                                             num_workers=0, drop_last=True)

    ################################## set test configs ######################################
    if args.body_mode == 'local_markers':
        in_channel = 1
    elif args.body_mode == 'local_markers_4chan':
        in_channel = 4
    infill_model = AE(downsample=True, in_channel=in_channel, kernel=args.conv_k).to(device)

    weights = torch.load(args.infill_model_path, map_location=lambda storage, loc: storage)
    infill_model.load_state_dict(weights)


    ###################### inference: get infilled clip imgs, per-instance self-supervised finetune ##################
    clip_img_rec_list = []
    clip_img_list = []
    rot_0_pivot_list = []
    # transf_matrix_smplx_list / smplx_params_gt_list: for evaluation of 3D accuracy
    transf_matrix_smplx_list = []
    smplx_params_gt_list = []
    beta_list = []
    gender_list = []

    finetune_step_total = 60
    print('[INFO] inference stage (with self-supervised finetuning)')
    for step, data in tqdm(enumerate(dataloader)):
        if step == args.end:
            break
        [clip_img, smplx_beta, gender,
         rot_0_pivot, transf_matrix_smplx, smplx_params_gt] = [item.to(device) for item in data]
        # T = clip_img.shape[-1]
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
        for finetine_step in range(finetune_step_total):
            infill_model.train()
            finetune_optimizer.zero_grad()

            #### forward
            clip_img_rec, z = infill_model(clip_img_input)

            #### finetune loss
            res_map = clip_img_rec[:, 0] - clip_img_input[:, 0]  # [bs=1, d, T]

            mask_row = np.concatenate([mask_row_id1, mask_row_id2, mask_row_id3]) + 1  # for pad at 1st row
            all_row = list(range(clip_img_rec.shape[-2]))
            upper_body_row = list(set(all_row) - set(mask_row))
            res_map_finetune = res_map[:, upper_body_row][:, 0:-5]

            loss = res_map_finetune.abs().mean()
            loss.backward()
            finetune_optimizer.step()
            # writer.add_scalar('finetune/loss', loss.item(), finetine_step)

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
        # transf_matrix_smplx_list.append(transf_matrix_smplx)
        # smplx_params_gt_list.append(smplx_params_gt)


    clip_img_rec_list = torch.cat(clip_img_rec_list, dim=0)  # [n, 1, d, T]  n clips
    clip_img_list = torch.cat(clip_img_list, dim=0)  # [n, 1/4, d, T]
    rot_0_pivot_list = torch.cat(rot_0_pivot_list, dim=0)  # [n, 1]
    beta_list = torch.cat(beta_list, dim=0)  # [n, 10]
    gender_list = torch.cat(gender_list, dim=0)  # [n, 1]

    # transf_matrix_smplx_list = torch.cat(transf_matrix_smplx_list, dim=0)  # [n, 4, 4]
    # transf_matrix_smplx_list = transf_matrix_smplx_list.detach().cpu().numpy()
    # smplx_params_gt_list = torch.cat(smplx_params_gt_list, dim=0)  # [n, T, 3+3+10+63+45+45] (T=120)
    # smplx_params_gt_list = smplx_params_gt_list.detach().cpu().numpy()


    #################################### get infilled motions ############################
    LIMBS = LIMBS_MARKER_SSM2

    color_input = np.zeros([len(LIMBS), 3])
    color_input[:, 0] = 1.0  # red
    color_rec = np.zeros([len(LIMBS), 3])
    color_rec[:, 2] = 1.0  # blue

    save_folder = os.path.join(args.save_dir, args.dataset_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save('{}/gender_list.npy'.format(save_folder), gender_list.detach().cpu().numpy())

    print('[INFO] optimizing per frame...')
    for i in range(args.start, args.end, args.step):
        print('current clip:', i)
        clip_img_rec = clip_img_rec_list[i, 0]  # [d, T]
        clip_img_input = clip_img_list[i]   # [1/4, d, T], to give global traj
        T = clip_img_rec.shape[-1]

        rot_0_pivot = rot_0_pivot_list[i].detach().cpu().numpy()

        beta_gt = beta_list[i]  # [10]
        gender = gender_list[i]
        if gender == 0:
            smplx_model = smplx_model_female
        elif gender == 1:
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
        body_joints_rec = body_joints_rec[:, 1:, :]  # remove first pelvis joint  todo: don't? use pelvis in optimize?

        body_joints_input = np.concatenate([pad_0, body_joints_input[:, 1:], body_joints_input[:, 0:1]], axis=1)
        body_joints_input = reconstruct_global_body(body_joints_input, rot_0_pivot)  # [T, 1+67, 3]  pelvis+markers (global position)
        body_joints_input = body_joints_input[:, 1:, :]


        #################################### optimize ############################################3
        body_params_opt_cur_clip = []
        body_verts_opt_t_1 = None
        for t in tqdm(range(T)):
            markers_rec_t = torch.from_numpy(body_joints_rec[t:t + 1, :]).float().to(device)  # np, [1, 67, 3]
            shape_t = beta_gt.unsqueeze(0)  # fixed shape  [1, 10]

            ############### init opt params ##################
            if t == 0:
                transl_opt_t = torch.zeros(1, 3).to(device)
                rot_opt_t = torch.zeros(1, 3).to(device)
                transl_opt_t[:, 1] = 0.4
                transl_opt_t[:, 2] = 1.0
                # rot_opt_t[:, 0] = 0.
                rot_opt_t[:, 1] = 1.6
                rot_opt_t[:, 2] = 3.14

                rot_6d_opt_t = convert_to_6D_all(rot_opt_t)
                other_params_opt_t = torch.zeros(1, 56).to(device)  # other params except transl/rot/shape

                transl_opt_t.requires_grad = True
                rot_6d_opt_t.requires_grad = True
                other_params_opt_t.requires_grad = True

            final_params = [transl_opt_t, rot_6d_opt_t, other_params_opt_t]
            if t == 0:
                init_lr = 0.1
            else:
                init_lr = 0.01
            optimizer = optim.Adam(final_params, lr=init_lr)


            # fitting iteration
            total_steps = 100
            for step in range(total_steps):
                if step > 60:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.01
                if step > 80:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.003
                optimizer.zero_grad()

                body_params_opt_t = torch.cat([transl_opt_t, rot_6d_opt_t, shape_t, other_params_opt_t], dim=-1)  # [1, 75]
                body_params_opt_t_72 = convert_to_3D_rot(body_params_opt_t)  # tensor, [bs=1, 72]
                body_verts_opt_t = gen_body_mesh_v1(body_params=body_params_opt_t_72, smplx_model=smplx_model,
                                                    vposer_model=vposer_model)  # tensor [1, 10475, 3]
                markers_opt_t = body_verts_opt_t[:, marker_ids, :]  # [1, 67, 3]

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

                loss = args.weight_loss_rec_markers * loss_marker + \
                       args.weight_loss_vposer * loss_vposer + \
                       args.weight_loss_shape * loss_shape + args.weight_loss_hand * loss_hand
                loss.backward(retain_graph=True)
                optimizer.step()


                if step == total_steps - 1:
                    body_params_opt_cur_clip.append(body_params_opt_t_72[0].detach().cpu().numpy())

            body_verts_opt_t_1 = body_verts_opt_t.detach()  # save opt results from last frame

        body_params_opt_cur_clip = np.asarray(body_params_opt_cur_clip)
        np.save('{}/body_params_opt_clip_{}.npy'.format(save_folder, i), body_params_opt_cur_clip)



if __name__ == '__main__':
    optimize()














