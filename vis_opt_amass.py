import smplx
import torch
import numpy as np
import os
from utils.utils import *
import open3d as o3d
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from human_body_prior.tools.model_loader import load_vposer


parser = argparse.ArgumentParser()


parser.add_argument('--body_model_path', type=str, default='/mnt/hdd/PROX/body_models', help='path to smplx body models')

parser.add_argument("--start", default=0, type=int, help='from which sequence to start')
parser.add_argument("--end", default=100, type=int, help='until which sequence to end')
parser.add_argument("--step", default=20, type=int, help='visualize 1 sequence every [step] sequences')
parser.add_argument('--dataset_name', type=str, default='TotalCapture', help='which dataset in AMASS')

# res_opt_amass_perframe / res_opt_amass_temp
parser.add_argument('--load_dir', type=str, default='res_opt_amass_temp', help='path to save optimized body params')

# visualize option: visualize in static poses on animation
parser.add_argument('--vis_option', type=str, default='animate', choices=['static', 'animate'])

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_cam(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T)  # T is applied in the rotated coord
    cam_aux = np.array([[0, 0, 0, 1]])
    mat = np.concatenate([cam_R, cam_T], axis=-1)
    mat = np.concatenate([mat, cam_aux], axis=0)
    cam_param.extrinsic = mat
    return cam_param

def capture_image(vis, outfilename=None):
    image = vis.capture_screen_float_buffer(do_render=True)
    if outfilename is None:
        plt.imshow(np.asarray(image))
        plt.show()
    else:
        plt.imsave(outfilename, np.asarray(image))
        print('-- output image to:' + outfilename)
    return False


if __name__ == '__main__':
    body_model_path = '/mnt/hdd/PROX/body_models'
    body_segments_dir = 'body_segments'

    smplx_model_path = os.path.join(body_model_path, 'smplx_model')
    vposer_model_path = os.path.join(body_model_path, 'vposer_v1_0')

    vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
    vposer_model = vposer_model.to(device)
    print('[INFO] vposer model loaded')

    smplx_model_male = smplx.create(smplx_model_path, model_type='smplx', gender='male', ext='npz',
                                   num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                   create_betas=True, create_left_hand_pose=True,
                                   create_right_hand_pose=True, create_expression=True, create_jaw_pose=True,
                                   create_leye_pose=True, create_reye_pose=True, create_transl=True,
                                   batch_size=119
                                   ).to(device)

    smplx_model_female = smplx.create(smplx_model_path, model_type='smplx', gender='female', ext='npz',
                                   num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                   create_betas=True, create_left_hand_pose=True,
                                   create_right_hand_pose=True, create_expression=True, create_jaw_pose=True,
                                   create_leye_pose=True, create_reye_pose=True, create_transl=True,
                                   batch_size=119
                                   ).to(device)

    smplx_model_amass_male = smplx.create(smplx_model_path, model_type='smplx', gender='male', ext='npz',
                                              use_pca=False, flat_hand_mean=True,
                                              # different here, true: mean hand pose is a flat hand
                                              create_global_orient=True, create_body_pose=True, create_betas=True,
                                              create_left_hand_pose=True, create_right_hand_pose=True,
                                              create_expression=True,
                                              create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True,
                                              create_transl=True,
                                              batch_size=119).to(device)

    smplx_model_amass_female = smplx.create(smplx_model_path, model_type='smplx', gender='female', ext='npz',
                                            use_pca=False, flat_hand_mean=True,
                                            # different here, true: mean hand pose is a flat hand
                                            create_global_orient=True, create_body_pose=True, create_betas=True,
                                            create_left_hand_pose=True, create_right_hand_pose=True,
                                            create_expression=True,
                                            create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True,
                                            create_transl=True,
                                            batch_size=119).to(device)



    with open('loader/SSM2.json') as f:
        marker_ids = list(json.load(f)['markersets'][0]['indices'].values())

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

    foot_verts_id = list(left_foot_verts_id) + list(right_foot_verts_id)


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)
    trans = np.eye(4)
    trans[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]])
    trans[:3, -1] = np.array([5, 2, 0])


    gender_list = np.load('{}/{}/gender_list.npy'.format(args.load_dir, args.dataset_name))
    for i in tqdm(range(args.start, args.end, args.step)):
        contact_lbl_rec = np.load('{}/{}/contact_lbl_rec_clip_{}.npy'.format(args.load_dir, args.dataset_name, i))
        body_params_opt = np.load('{}/{}/body_params_opt_clip_{}.npy'.format(args.load_dir, args.dataset_name, i))
        body_params_opt = torch.from_numpy(body_params_opt).float().to(device)

        if gender_list[i] == 0:
            smplx_model = smplx_model_female
            smplx_model_amass = smplx_model_amass_female
        else:
            smplx_model = smplx_model_male
            smplx_model_amass = smplx_model_amass_male

        # body_params_opt[:, 0:2] = body_params_opt[:, 0:2] * 3

        body_verts_opt = gen_body_mesh_v1(body_params=body_params_opt, smplx_model=smplx_model,
                                          vposer_model=vposer_model)  # tensor [T, 10475, 3]
        markers_opt = body_verts_opt[:, marker_ids, :]
        joints_opt = gen_body_joints_v1(body_params=body_params_opt, smplx_model=smplx_model,
                                        vposer_model=vposer_model)[:, 0:25]
        joints_opt_lower = joints_opt[:, [5, 11, 8, 4, 7, 10]]
        markers_opt_lower = markers_opt[:, [14, 15, 18, 19, 29, 2, 20, 21, 30, 25, 16,
                                            45, 46, 48, 49, 59, 32, 50, 51, 55, 60, 47]]

        T = len(body_params_opt)
        body_opt_mesh_all_frame_list = []
        sphere_all_frame_list = []

        for t in range(0, T):
            body_opt_mesh_cur_frame_list = []
            sphere_cur_frame_list = []
            contact_marker_cur_frame_list = []
            ########################## visualization ###########################
            body_verts_opt_t = body_verts_opt[t].detach().cpu().numpy()
            body_opt_mesh = o3d.geometry.TriangleMesh()
            body_opt_mesh.vertices = o3d.utility.Vector3dVector(body_verts_opt_t)
            body_opt_mesh.triangles = o3d.utility.Vector3iVector(smplx_model_male.faces)
            body_opt_mesh.compute_vertex_normals()
            body_opt_mesh_cur_frame_list.append(body_opt_mesh)
            if t % 20 == 0:
                body_opt_mesh_all_frame_list.append(body_opt_mesh)

            ###### add spheres for bodymarkers
            # markers_opt: [T, 67, 3]
            for j in range(67):
                transformation = np.identity(4)
                transformation[:3, 3] = markers_opt[t][j].detach().cpu().numpy()

                if j in [14, 15, 18, 19, 29, 2, 20, 21, 25, 45, 46, 48, 49, 59, 32, 50, 51, 55]:
                    # infilled markers
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    sphere.paint_uniform_color([218 / 255, 165 / 255, 32 / 255])  # yellow, 218,165,32
                # for foot markers, visualize corresponding foot contact labels
                elif j == 16:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
                    if contact_lbl_rec[t][0] == 1:
                        sphere.paint_uniform_color([0, 128 / 255, 0])  # green, in contact
                    else:
                        sphere.paint_uniform_color([128 / 255, 0, 0])  # red, not in contact
                    contact_marker_cur_frame_list.append(sphere)
                elif j == 47:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
                    if contact_lbl_rec[t][1] == 1:
                        sphere.paint_uniform_color([0, 128 / 255, 0])  # green, in contact
                    else:
                        sphere.paint_uniform_color([128 / 255, 0, 0])  # red, not in contact
                    contact_marker_cur_frame_list.append(sphere)
                elif j == 30:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
                    if contact_lbl_rec[t][2] == 1:
                        sphere.paint_uniform_color([0, 128 / 255, 0])  # green, in contact
                    else:
                        sphere.paint_uniform_color([128 / 255, 0, 0])  # red, not in contact
                    contact_marker_cur_frame_list.append(sphere)
                elif j == 60:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
                    if contact_lbl_rec[t][3] == 1:
                        sphere.paint_uniform_color([0, 128 / 255, 0])  # green, in contact
                    else:
                        sphere.paint_uniform_color([128 / 255, 0, 0])  # red, not in contact
                    contact_marker_cur_frame_list.append(sphere)
                else:
                    # input markers
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    sphere.paint_uniform_color([70 / 255, 130 / 255, 180 / 255])  # steel blue 70,130,180

                # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                # sphere.paint_uniform_color([70 / 255, 130 / 255, 180 / 255])  # steel blue 70,130,180

                sphere.compute_vertex_normals()
                sphere.transform(transformation)
                sphere_cur_frame_list.append(sphere)
                if t % 20 == 0:
                    sphere_all_frame_list.append(sphere)

            # visualize in animation
            if args.vis_option == 'animate':
                for body_mesh in body_opt_mesh_cur_frame_list:
                    vis.add_geometry(body_mesh)
                for sphere in contact_marker_cur_frame_list:
                    vis.add_geometry(sphere)

                ctr = vis.get_view_control()
                cam_param = ctr.convert_to_pinhole_camera_parameters()
                cam_param = update_cam(cam_param, trans)
                ctr.convert_from_pinhole_camera_parameters(cam_param)

                vis.poll_events()
                vis.update_renderer()
                # time.sleep(0.01)

                for body_mesh in body_opt_mesh_cur_frame_list:
                    vis.remove_geometry(body_mesh)
                for sphere in contact_marker_cur_frame_list:
                    vis.remove_geometry(sphere)

        # visualize in static poses
        if args.vis_option == 'static':
            o3d.visualization.draw_geometries(sphere_all_frame_list + body_opt_mesh_all_frame_list)


