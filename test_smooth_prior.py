import argparse
import torch
from torch.utils import data
import time
from tqdm import tqdm
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


from loader.train_loader_smooth import TrainLoader
from models.AE_sep import Enc, Dec
from utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default='0')

parser.add_argument('--save_dir', type=str, default='runs_try', help='path to save train logs and models')
parser.add_argument('--batch_size', type=int, default=20, help='input batch size')


# path to amass and smplx body model
parser.add_argument('--amass_dir', type=str, default='/local/home/szhang/AMASS/amass', help='path to AMASS dataset')
parser.add_argument('--body_model_path', type=str, default='/mnt/hdd/PROX/body_models', help='path to smplx body models')


# settings for body representation
parser.add_argument("--clip_seconds", default=4, type=int, help='length (seconds) of each motion sequence')
parser.add_argument('--body_mode', type=str, default='global_markers',
                    choices=['global_markers', 'global_joints'], help='which body representation to use')
parser.add_argument('--with_hand', default='True', type=lambda x: x.lower() in ['true', '1'], help='include hand or not')
parser.add_argument('--normalize', default='True', type=lambda x: x.lower() in ['true', '1'], help='normalize input motion representation or not')
parser.add_argument('--input_padding', default='True', type=lambda x: x.lower() in ['true', '1'], help='pad input motion representation or not')

# settings for network
parser.add_argument('--downsample', default='False', type=lambda x: x.lower() in ['true', '1'], help='downsample latent space or not')
parser.add_argument("--z_channel", default=64, type=int, help='channel # of latent space z')
parser.add_argument('--Enc_path', type=str, default='runs/15217/Enc_last_model.pkl', help='path to pretrained motion smoothness prior encoder')
parser.add_argument('--Dec_path', type=str, default='runs/15217/Dec_last_model.pkl', help='path to pretrained motion smoothness prior decoder')

parser.add_argument('--dataset_name', type=str, default='TotalCapture', help='which dataset in amass')


# amass_train_datasets = ['HumanEva', 'MPI_HDM05', 'MPI_mosh', 'Transitions_mocap',
#                         'ACCAD', 'BMLhandball', 'BMLmovi', 'BioMotionLab_NTroje', 'CMU',
#                         'DFaust_67', 'Eyes_Japan_Dataset', 'MPI_Limits']
# amass_test_datasets = ['TCD_handMocap', 'TotalCapture', 'SFU']


args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('gpu id:', torch.cuda.current_device())



def update_cam(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T)  # T is applied in the rotated coord
    cam_aux = np.array([[0, 0, 0, 1]])
    mat = np.concatenate([cam_R, cam_T], axis=-1)
    mat = np.concatenate([mat, cam_aux], axis=0)
    cam_param.extrinsic = mat
    return cam_param



def test():
    # amass_dir = '/local/home/szhang/AMASS/amass'
    # body_model_path = '/mnt/hdd/PROX/body_models'

    smplx_model_path = os.path.join(args.body_model_path, 'smplx_model')

    ################################### set dataloaders ######################################
    print('[INFO] reading test data from datasets {}...'.format(args.dataset_name))
    dataset = TrainLoader(clip_seconds=args.clip_seconds, clip_fps=30, normalize=args.normalize,
                          split='test', mode=args.body_mode)
    dataset.read_data([args.dataset_name], args.amass_dir)
    dataset.create_body_repr(with_hand=args.with_hand,
                             smplx_model_path=smplx_model_path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=0, drop_last=True)

    ################################## set tes configs ######################################
    encoder = Enc(downsample=args.downsample, z_channel=args.z_channel).to(device)
    decoder = Dec(downsample=args.downsample, z_channel=args.z_channel).to(device)

    weights = torch.load(args.Enc_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(weights)
    encoder.eval()

    weights = torch.load(args.Dec_path, map_location=lambda storage, loc: storage)
    decoder.load_state_dict(weights)
    decoder.eval()


    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh_frame)

    trans = np.eye(4)
    trans[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]])
    trans[:3, -1] = np.array([5, 2, 0])
    # # top view
    # trans[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    # trans[:3, -1] = np.array([0, 0, 4])

    ################################## test #########################################
    with torch.no_grad():
        for step, data in tqdm(enumerate(dataloader)):
            [clip_img] = [item.to(device) for item in data]
            T = clip_img.shape[-1]  # T=120frames

            # netowrk input/output residuals (velocity)
            clip_img_v = clip_img[:, :, :, 1:] - clip_img[:, :, :, 0:-1]

            if args.input_padding:
                p2d = (8, 8, 1, 1)
                clip_img_v = F.pad(clip_img_v, p2d, 'reflect')

            z, input_size, x_down1_size, x_down2_size, x_down3_size, x_down4_size = encoder(clip_img_v)
            clip_img_v_rec = decoder(z, input_size, x_down1_size, x_down2_size, x_down3_size, x_down4_size)

            clip_img_rec = clip_img_v_rec.clone()
            if args.input_padding:
                clip_img_rec = clip_img_rec[:, :, 1:-1, 8:-8]  # [75, T=119]
                clip_img_rec = torch.cat([clip_img[:, :, :, 0:1], clip_img_rec], dim=-1)  # [75, T=120]

            for i in range(1, T):
                clip_img_rec[:, :, :, i] = clip_img_rec[:, :, :, i] + clip_img_rec[:, :, :, i - 1]


            # visualize latent z
            # cur_z = z[0].detach().cpu().numpy()  # [256, d, T]
            # max_z, min_z = np.max(cur_z), np.min(cur_z)
            # print('max/min of z:', max_z, min_z)
            # for i in [30, 60, 90, 120, 150, 180, 210, 240]:
            # # for i in [5,10,15,20,30,40,50,60]:
            #     z_map = cur_z[i].astype(np.float32)
            #     z_map = z_map[1:-1, 8:-8]
            #     c_max_z, c_min_z = np.max(z_map), np.min(z_map)
            #     print('max/min of z[]:'.format(i), c_max_z, c_min_z)
            #     fig = plt.imshow(z_map, cmap='viridis', vmin=c_min_z, vmax=c_max_z)
            #     plt.axis('off')
            #     fig.axes.get_xaxis().set_visible(False)
            #     fig.axes.get_yaxis().set_visible(False)
            #     # plt.imsave('1.jpg', z_map, cmap='viridis')
            #     plt.show()


            body_joints_input = clip_img[0][0].permute(1, 0).reshape(T, -1, 3)  # [T, 25/55, 3]
            body_joints_rec = clip_img_rec[0][0].permute(1, 0).reshape(T, -1, 3)
            body_joints_input = body_joints_input.detach().cpu().numpy()  # [T, 25/55, 3]
            body_joints_rec = body_joints_rec.detach().cpu().numpy()

            if args.normalize:
                if not args.with_hand:
                    preprocess_stats = np.load('preprocess_stats/preprocess_stats_smooth_{}.npz'.format(args.body_mode))
                else:
                    preprocess_stats = np.load('preprocess_stats/preprocess_stats_smooth_withHand_{}.npz'.format(args.body_mode))
                body_joints_input = np.reshape(body_joints_input, (T, -1))
                body_joints_rec = np.reshape(body_joints_rec, (T, -1))
                body_joints_input = body_joints_input * preprocess_stats['Xstd'] + preprocess_stats['Xmean']
                body_joints_rec = body_joints_rec * preprocess_stats['Xstd'] + preprocess_stats['Xmean']
                body_joints_input = np.reshape(body_joints_input, (T, -1, 3))
                body_joints_rec = np.reshape(body_joints_rec, (T, -1, 3))


            ##################### visualization ###############################
            if args.body_mode in ['global_markers']:
                LIMBS = LIMBS_MARKER_SSM2
            elif args.body_mode in ['global_joints']:
                LIMBS = LIMBS_BODY
            color_input = np.zeros([len(LIMBS), 3])
            color_input[:, 0] = 1.0
            color_rec = np.zeros([len(LIMBS), 3])
            color_rec[:, 2] = 1.0
            for t in range(0, T):
                skeleton_input = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(body_joints_input[t]),
                    lines=o3d.utility.Vector2iVector(LIMBS))
                skeleton_input.colors = o3d.utility.Vector3dVector(color_input)

                skeleton_rec = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(body_joints_rec[t]),
                    lines=o3d.utility.Vector2iVector(LIMBS))
                skeleton_rec.colors = o3d.utility.Vector3dVector(color_rec)

                # if t in [0, 50, 100, 119]:
                #     o3d.visualization.draw_geometries([skeleton_input, skeleton_rec, mesh_frame])
                # print(body_joints_input[t][0])

                vis.add_geometry(skeleton_input)
                vis.add_geometry(skeleton_rec)

                ctr = vis.get_view_control()
                cam_param = ctr.convert_to_pinhole_camera_parameters()
                cam_param = update_cam(cam_param, trans)
                ctr.convert_from_pinhole_camera_parameters(cam_param)

                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.05)
                vis.remove_geometry(skeleton_input)
                vis.remove_geometry(skeleton_rec)





if __name__ == '__main__':
    test()


