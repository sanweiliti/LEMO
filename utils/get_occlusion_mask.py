import smplx
import os
import pickle
import pyrender
import trimesh
import numpy as np
import json
import torch
from tqdm import tqdm
import PIL.Image as pil_img
from human_body_prior.tools.model_loader import load_vposer
from temp_prox.camera import create_camera

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_prox_pkl(pkl_path):
    body_params_dict = {}
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        # data keys: camera_rotation, camera_translation, (useless)
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


scene_name = 'BasementSittingBooth'
seq_name_list = ['BasementSittingBooth_00142_01', 'BasementSittingBooth_00145_01']
# scene_name = 'MPH112'
# seq_name_list = ['MPH112_00034_01','MPH112_00150_01', 'MPH112_00151_01', 'MPH112_00157_01', 'MPH112_00169_01']
# scene_name = 'MPH11'
# seq_name_list = ['MPH11_00034_01', 'MPH11_00150_01', 'MPH11_00151_01', 'MPH11_03515_01']
# scene_name = 'MPH16'
# seq_name_list = ['MPH16_00157_01']
# scene_name = 'MPH1Library'
# seq_name_list = ['MPH1Library_00034_01']
# scene_name = 'MPH8'
# seq_name_list = ['MPH8_00168_01']
# scene_name = 'N0SittingBooth'
# seq_name_list = ['N0SittingBooth_00162_01', 'N0SittingBooth_00169_01', 'N0SittingBooth_00169_02',
#                  'N0SittingBooth_03301_01', 'N0SittingBooth_03403_01']
# scene_name = 'N0Sofa'
# seq_name_list = ['N0Sofa_00034_01', 'N0Sofa_00034_02', 'N0Sofa_00141_01', 'N0Sofa_00145_01']
# scene_name = 'N3Library'
# seq_name_list = ['N3Library_00157_01', 'N3Library_00157_02', 'N3Library_03301_01', 'N3Library_03301_02',
#                  'N3Library_03375_01', 'N3Library_03375_02', 'N3Library_03403_01', 'N3Library_03403_02']
# scene_name = 'N3Office'
# seq_name_list = ['N3Office_00034_01', 'N3Office_00139_01', 'N3Office_00139_02', 'N3Office_00150_01',
#                  'N3Office_00153_01', 'N3Office_00159_01', 'N3Office_03301_01']
# scene_name = 'N3OpenArea'
# seq_name_list = ['N3OpenArea_00157_01', 'N3OpenArea_00157_02', 'N3OpenArea_00158_01', 'N3OpenArea_00158_02',
#                  'N3OpenArea_03301_01', 'N3OpenArea_03403_01']
# scene_name = 'Werkraum'
# seq_name_list = ['Werkraum_03301_01', 'Werkraum_03403_01', 'Werkraum_03516_01', 'Werkraum_03516_02']


for seq_name in seq_name_list:

    prox_params_folder = '/mnt/hdd/PROX/PROXD/{}'.format(seq_name)
    img_folder = '/mnt/hdd/PROX/recordings/{}/Color'.format(seq_name)
    scene_mesh_path = '/mnt/hdd/PROX/scenes/{}.ply'.format(scene_name)
    keyp_folder = '/mnt/hdd/PROX/keypoints/{}'.format(seq_name)
    save_img_folder = '/local/home/szhang/temp_prox/mask_render_imgs/{}'.format(seq_name)
    save_mask_folder = '/local/home/szhang/temp_prox/mask_joint/{}'.format(seq_name)
    # if not os.path.exists(save_img_folder):
    #     os.makedirs(save_img_folder)

    cam2world_dir = '/mnt/hdd/PROX/cam2world'
    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        cam2world = np.array(json.load(f))  # [4, 4] last row: [0,0,0,1]

    ########## smplx/vposer model
    vposer_model_path = '/mnt/hdd/PROX/body_models/vposer_v1_0'
    smplx_model_path = '/mnt/hdd/PROX/body_models/smplx_model'

    smplx_model = smplx.create(smplx_model_path, model_type='smplx',
                                   gender='male', ext='npz',
                                   num_pca_comps=12,
                                   create_global_orient=True,
                                   create_body_pose=True,
                                   create_betas=True,
                                   create_left_hand_pose=True,
                                   create_right_hand_pose=True,
                                   create_expression=True,
                                   create_jaw_pose=True,
                                   create_leye_pose=True,
                                   create_reye_pose=True,
                                   create_transl=True,
                                   batch_size=1
                                   ).to(device)
    print('[INFO] smplx model loaded.')

    vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
    vposer_model = vposer_model.to(device)
    print('[INFO] vposer model loaded')


    ########### render settings
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

    static_scene = trimesh.load(scene_mesh_path)
    trans = np.linalg.inv(cam2world)
    static_scene.apply_transform(trans)
    static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)



    ####################### create camera object ########################
    camera_center = torch.tensor(camera_center).float().reshape(1, 2)  # # tensor, [1,2]
    camera = create_camera(focal_length_x=1060.53,
                           focal_length_y=1060.53,
                           center= camera_center,
                           batch_size=1,)
    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False
    camera = camera.to(device)

    ############################# redering scene #######################
    scene = pyrender.Scene()
    scene.add(camera_render, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    scene.add(static_scene_mesh, 'mesh')
    r = pyrender.OffscreenRenderer(viewport_width=1920,
                                   viewport_height=1080)
    color_scene, depth_scene = r.render(scene)  # color [1080, 1920, 3], depth [1080, 1920]
    # color_scene = color_scene.astype(np.float32) / 255.0
    # img_scene = (color_scene * 255).astype(np.uint8)


    ############# render, mask
    img_list = os.listdir(img_folder)
    img_list = sorted(img_list)
    seq_mask = []
    cnt = 0
    for img_fn in tqdm(img_list):
        cnt += 1
        # if cnt == 1000:
        #     break
        if img_fn.endswith('.png') or img_fn.endswith('.jpg') and not img_fn.startswith('.'):
            mask = np.ones([25])

            ######## get smplx body mesh
            prox_params_dir = os.path.join(prox_params_folder, 'results', img_fn[0:-4], '000.pkl')
            params_dict = read_prox_pkl(prox_params_dir)
            for param_name in params_dict:
                params_dict[param_name] = np.expand_dims(params_dict[param_name], axis=0)
                params_dict[param_name] = torch.from_numpy(params_dict[param_name]).to(device)
            smplx_output = smplx_model(return_verts=True, **params_dict)  # generated human body mesh
            body_verts = smplx_output.vertices.detach().cpu().numpy()[0]
            body_mesh = trimesh.Trimesh(body_verts, smplx_model.faces, process=False)
            body_mesh = pyrender.Mesh.from_trimesh(body_mesh, material=material)


            ############################# redering body #######################
            scene = pyrender.Scene()
            scene.add(camera_render, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            # scene.add(static_scene_mesh, 'mesh')
            scene.add(body_mesh, 'mesh')
            r = pyrender.OffscreenRenderer(viewport_width=1920,
                                           viewport_height=1080)
            color_body, depth_body = r.render(scene)  # color [1080, 1920, 3], depth [1080, 1920]
            # color_body = color_body.astype(np.float32) / 255.0
            # img_body = (color_body * 255).astype(np.uint8)

            ######### body joints --> set mask
            body_joints = smplx_output.joints
            projected_joints = camera(body_joints)  # [1, n, 2]
            projected_joints = projected_joints[0][0:25].detach().cpu().numpy()  # [25, 2]
            projected_joints = projected_joints.astype(int)

            for j_id in range(25):
            # for j_id in [5, 8, 11, 4, 7, 10]:  # todo: only mask joints in legs/feet
                x_coord, y_coord = projected_joints[j_id][0], projected_joints[j_id][1]
                if 0 <= x_coord < 1920 and 0 <= y_coord < 1080:
                    if depth_body[y_coord][x_coord] - depth_scene[y_coord][x_coord] > 0.1 \
                            and depth_scene[y_coord][x_coord] != 0:  # todo: set threshold
                        mask[j_id] = 0       # occlusion happens, mask corresponding joint
            seq_mask.append(mask)



            # ############################# render body+scene (for visualization)
            # scene = pyrender.Scene()
            # scene.add(camera_render, pose=camera_pose)
            # scene.add(light, pose=camera_pose)
            # scene.add(static_scene_mesh, 'mesh')
            # scene.add(body_mesh, 'mesh')
            # r = pyrender.OffscreenRenderer(viewport_width=1920,
            #                                viewport_height=1080)
            # color, _ = r.render(scene)  # color [1080, 1920, 3], depth [1080, 1920]
            # color = color.astype(np.float32) / 255.0
            # save_img = (color * 255).astype(np.uint8)
            #
            # ########## for visualization
            # for k in range(len(projected_joints)):
            #     for p in range(max(0, projected_joints[k][0] - 3), min(1920 - 1, projected_joints[k][0] + 3)):
            #         for q in range(max(0, projected_joints[k][1] - 3), min(1080 - 1, projected_joints[k][1] + 3)):
            #             if mask[k] == 1:
            #                 save_img[q][p][0] = 255
            #                 save_img[q][p][1] = 0
            #                 save_img[q][p][2] = 0
            #             else:
            #                 save_img[q][p][0] = 0
            #                 save_img[q][p][1] = 255
            #                 save_img[q][p][2] = 0
            #
            # ######### save img (for visualization)
            # # img_scene = pil_img.fromarray(img_scene.astype(np.uint8))
            # # img_scene.save('img_scene.png')
            # # img_body = pil_img.fromarray(img_body.astype(np.uint8))
            # # img_body.save('img_body.png')
            # save_img = pil_img.fromarray(save_img.astype(np.uint8))
            # save_img.save('{}/{}'.format(save_img_folder, img_fn))

    if not os.path.exists(save_mask_folder):
        os.makedirs(save_mask_folder)
    seq_mask = np.asarray(seq_mask)
    np.save('{}/mask_joint.npy'.format(save_mask_folder), seq_mask)







