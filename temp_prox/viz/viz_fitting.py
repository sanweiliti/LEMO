import os
import os.path as osp
import cv2
import numpy as np
import json
import open3d as o3d
import argparse
import time

import torch
import pickle
import smplx
import matplotlib.pyplot as plt
import PIL.Image as pil_img


def update_cam(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T)  # !!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0, 0, 0, 1]])
    mat = np.concatenate([cam_R, cam_T], axis=-1)
    mat = np.concatenate([mat, cam_aux], axis=0)
    cam_param.extrinsic = mat
    return cam_param

def capture_image(vis, outfilename=None):
    image = vis.capture_screen_float_buffer()
    if outfilename is None:
        plt.imshow(np.asarray(image))
        plt.show()
    else:
        plt.imsave(outfilename, np.asarray(image))
        # print('-- output image to:' + outfilename)
    return False


def main(args):
    fitting_dir = args.fitting_dir
    recording_name = os.path.abspath(fitting_dir).split("/")[-1]
    fitting_dir = osp.join(fitting_dir, 'results')
    scene_name = recording_name.split("_")[0]
    base_dir = args.base_dir
    cam2world_dir = osp.join(base_dir, 'cam2world')
    scene_dir = osp.join(base_dir, 'scenes')

    female_subjects_ids = [162, 3452, 159, 3403]
    subject_id = int(recording_name.split('_')[1])
    if subject_id in female_subjects_ids:
        gender = 'female'
    else:
        gender = 'male'


    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        trans = np.array(json.load(f))


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)
    render_opt = vis.get_render_option().mesh_show_back_face = True

    scene = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '.ply'))
    vis.add_geometry(scene)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # vis.add_geometry(mesh_frame)


    model = smplx.create(args.model_folder, model_type='smplx',
                         gender=gender, ext='npz',
                         num_pca_comps=args.num_pca_comps,
                         create_global_orient=True,
                         create_body_pose=True,
                         create_betas=True,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=True,
                         create_jaw_pose=True,
                         create_leye_pose=True,
                         create_reye_pose=True,
                         create_transl=True
                         )

    cnt = 0

    rendering_dir = os.path.join(args.fitting_dir, 'images')
    if not os.path.exists(rendering_dir):
        os.makedirs(rendering_dir)

    for img_name in sorted(os.listdir(fitting_dir))[args.start::args.step]:
        cnt += 1
        print('viz frame {}'.format(img_name))

        if not osp.exists(osp.join(fitting_dir, img_name, '000.pkl')):
            continue

        with open(osp.join(fitting_dir, img_name, '000.pkl'), 'rb') as f:
            param = pickle.load(f)
        torch_param = {}
        for key in param.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation']:
                continue
            else:
                torch_param[key] = torch.tensor(param[key])


        output = model(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()

        body = o3d.geometry.TriangleMesh()
        body.vertices = o3d.utility.Vector3dVector(vertices)
        body.triangles = o3d.utility.Vector3iVector(model.faces)
        body.compute_vertex_normals()
        body.transform(trans)  # camera to world coordinate

        vis.add_geometry(body)

        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        cam_param = update_cam(cam_param, trans)
        ctr.convert_from_pinhole_camera_parameters(cam_param)


        vis.poll_events()
        vis.update_renderer()
        vis.remove_geometry(body)
        # time.sleep(0.1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitting_dir', type=str, default='../../fit_results_S3/N3OpenArea_00157_01',
                        help='recording dir')
    parser.add_argument('--base_dir', type=str, default='/mnt/hdd/PROX',
                        help='recording dir')
    parser.add_argument('--start', type=int, default=0, help='id of the starting frame')
    parser.add_argument('--step', type=int, default=1, help='id of the starting frame')
    parser.add_argument('--model_folder', default='/mnt/hdd/PROX/body_models/smplx_model', type=str, help='')
    parser.add_argument('--num_pca_comps', type=int, default=12,help='')
    parser.add_argument('--gender', type=str, default='male', choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL' +
                             'model')
    args = parser.parse_args()
    main(args)
