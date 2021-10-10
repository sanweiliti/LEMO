import argparse
import torch
from torch.utils import data
from tqdm import tqdm
from tensorboardX import SummaryWriter
import smplx
import torch.optim as optim
import itertools

from loader.train_loader_infill import TrainLoader
from models.AE import AE
from utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default='0')

parser.add_argument('--save_dir', type=str, default='runs_try', help='path to save train logs and models')
parser.add_argument('--batch_size', type=int, default=60, help='input batch size')
parser.add_argument('--num_workers', type=int, default=2, help='# of dataloadeer num_workers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--num_epoch', type=int, default=100000, help='# of training epochs ')
parser.add_argument("--log_step", default=500, type=int, help='log after n iters')
parser.add_argument("--save_step", default=1000, type=int, help='save models after n iters')

# path to amass and smplx body model
parser.add_argument('--amass_dir', type=str, default='/local/home/szhang/AMASS/amass', help='path to AMASS dataset')
parser.add_argument('--body_model_path', type=str, default='/mnt/hdd/PROX/body_models', help='path to smplx body models')

# settings for body representation
parser.add_argument("--clip_seconds", default=4, type=int, help='length (seconds) of each motion sequence')
parser.add_argument('--body_mode', type=str, default='local_markers_4chan',
                    choices=['local_markers', 'local_markers_4chan'], help='which body representation to use')
parser.add_argument("--conv_k", default=3, type=int)
parser.add_argument('--with_hand', default='False', type=lambda x: x.lower() in ['true', '1'], help='include hand or not')
parser.add_argument('--normalize', default='True', type=lambda x: x.lower() in ['true', '1'], help='normalize input motion representation or not')
parser.add_argument('--input_padding', default='True', type=lambda x: x.lower() in ['true', '1'], help='pad input motion representation or not')


# settings for network
parser.add_argument('--downsample', default='True', type=lambda x: x.lower() in ['true', '1'], help='downsample latent space or not')

# loss weights
parser.add_argument("--weight_loss_rec_body", default=10.0, type=float, help='weight for input reconstruction loss')
parser.add_argument("--weight_loss_rec_body_v", default=10.0, type=float, help='weight for input 1st-order reconstruction loss')
parser.add_argument("--weight_loss_rec_contact_lbl", default=1.0, type=float, help='weight for foot contact prediction loss')




args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('gpu id:', torch.cuda.current_device())




def train(writer, logger):
    # amass_dir = '/local/home/szhang/AMASS/amass'
    # body_model_path = '/mnt/hdd/PROX/body_models'

    smplx_model_path = os.path.join(args.body_model_path, 'smplx_model')

    # amass_train_datasets = ['HumanEva', 'MPI_HDM05', 'MPI_mosh', 'Transitions_mocap',
    #                         'ACCAD', 'BMLhandball', 'BMLmovi', 'BioMotionLab_NTroje', 'CMU',
    #                         'DFaust_67', 'Eyes_Japan_Dataset', 'MPI_Limits']
    # amass_test_datasets = ['TCD_handMocap', 'TotalCapture', 'SFU']
    amass_train_datasets = ['HumanEva', 'BMLmovi']
    amass_test_datasets = ['TCD_handMocap', 'TotalCapture']

    preprocess_stats_dir = 'preprocess_stats'
    if not os.path.exists(preprocess_stats_dir):
        os.makedirs(preprocess_stats_dir)

    ################################### set dataloaders ######################################
    print('[INFO] reading training data from datasets {}...'.format(amass_train_datasets))
    train_dataset = TrainLoader(clip_seconds=args.clip_seconds, clip_fps=30, normalize=args.normalize,
                                split='train', mode=args.body_mode)
    train_dataset.read_data(amass_train_datasets, args.amass_dir)
    train_dataset.create_body_repr(with_hand=args.with_hand,
                                   smplx_model_path=smplx_model_path)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, drop_last=True)

    print('[INFO] reading test data from datasets {}...'.format(amass_test_datasets))
    test_dataset = TrainLoader(clip_seconds=args.clip_seconds, clip_fps=30, normalize=args.normalize,
                               split='test', mode=args.body_mode)
    test_dataset.read_data(amass_test_datasets, args.amass_dir)
    test_dataset.create_body_repr(with_hand=args.with_hand,
                                  smplx_model_path=smplx_model_path)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=True)


    ################################## set train configs ######################################
    if args.body_mode in ['local_markers']:
        in_channel = 1
    elif args.body_mode in ['local_markers_4chan']:
        in_channel = 4
    model = AE(downsample=args.downsample, in_channel=in_channel, kernel=args.conv_k).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  itertools.chain(model.parameters())),
                           lr=args.lr)

    bce_loss = nn.BCEWithLogitsLoss().to(device)

    ################################# load prox masks ########################################
    prox_mask_dir_list = os.listdir('mask_markers')  # ['MPH11_00034_01', 'MPH11_00150_01', ...]
    prox_mask_list = []
    for dir in tqdm(prox_mask_dir_list):
        mask = np.load('mask_markers/{}/mask_markers.npy'.format(dir))  # 0: markers to mask out
        n_clip = len(mask) // 120
        for i in range(n_clip):
            mask_clip = mask[(i*120):((i+1)*120)]  # [T, 67]
            all_markers_n = mask_clip.shape[0] * mask_clip.shape[1]
            mask_markers_n = all_markers_n  - mask_clip.sum()
            mask_ratio = mask_markers_n / all_markers_n
            if mask_ratio >= 0.05:  # ignore clips with few masks
                cur_mask_clip = np.repeat(mask_clip, 3, axis=1)  # [T, 67*3]
                prox_mask_list.append(cur_mask_clip)    # [n_seq, T, 67*3]
    prox_mask_list = np.asarray(prox_mask_list)
    print('[INFO] prox masks loaded, get {} prox mask clips in total.'.format(len(prox_mask_list)))


    ################################## start training #########################################
    total_steps = 0
    for epoch in range(args.num_epoch):
        for step, data in tqdm(enumerate(train_dataloader)):
            model.train()

            total_steps += 1
            [clip_img] = [item.to(device) for item in data]  # clip_img:  # [bs, 1/4, 1, T]

            optimizer.zero_grad()

            ###### mask input
            clip_img_input = clip_img.clone()  # [bs, 1/4, d, T]
            bs = clip_img.shape[0]
            d = clip_img.shape[-2]
            T = clip_img.shape[-1]


            if epoch <= 20:
                mask_marker_n = random.randint(1, 6)
                mask_marker_id = torch.rand(bs, mask_marker_n) * 67  # all 67 markers
                mask_marker_id = mask_marker_id.long()  # [bs, mask_marker_n]
                mask_row_id1 = mask_marker_id * 3
                if args.body_mode in ['local_markers']:  # for global traj and pelvis joint
                    mask_row_id1 = mask_row_id1 + 3 + 3
                if args.body_mode in ['local_markers_4chan']:  # for pelvis joint
                    mask_row_id1 = mask_row_id1 + 3
                mask_row_id2 = mask_row_id1 + 1
                mask_row_id3 = mask_row_id2 + 1
                for i in range(bs):
                    clip_img_input[i, 0, mask_row_id1[i], :] = 0.
                    clip_img_input[i, 0, mask_row_id2[i], :] = 0.
                    clip_img_input[i, 0, mask_row_id3[i], :] = 0.
                    # mask contact lbls/distance if foot marker is masked
                    if 16 in mask_marker_id[i] or 30 in mask_marker_id[i]:
                        clip_img_input[i, 0, -4, :] = 0.
                        clip_img_input[i, 0, -2, :] = 0.
                    if 47 in mask_marker_id[i] or 60 in mask_marker_id[i]:
                        clip_img_input[i, 0, -3, :] = 0.
                        clip_img_input[i, 0, -1, :] = 0.

            else:
                # load prox masks, prox_mask_list: [n_seq, T=120, 25*3]
                np.random.shuffle(prox_mask_list)  # shuffle along n_seq axis
                mask = torch.from_numpy(prox_mask_list[0:bs]).float().permute(0, 2, 1).unsqueeze(1).to(device)  # [bs, 1, 67*3, 120]

                # mask contact lbls if foot marker is masked
                # is_mask_left: 1: keep left foot, 0: mask left foot
                is_mask_left = (mask[:, :, (16 * 3):(16 * 3 + 1), :] == 1) * (mask[:, :, (30 * 3):(30 * 3 + 1), :] == 1)  # [bs, 1, 1, 120]
                is_mask_right = (mask[:, :, (47 * 3):(47 * 3 + 1), :] == 1) * (mask[:, :, (60 * 3):(60 * 3 + 1), :] == 1)
                append_contact_mask = torch.cat([is_mask_left, is_mask_right, is_mask_left, is_mask_right], dim=-2)  # [bs, 1, 4, 120]
                append_contact_mask = append_contact_mask.float()

                if args.body_mode in ['local_markers']:
                    append_mask = torch.ones([bs, 1, 3+3, T]).to(device)  # for global traj and pelvis joint
                if args.body_mode in ['local_markers_4chan']:
                    append_mask = torch.ones([bs, 1, 3, T]).to(device)  # for pelvis joint

                mask = torch.cat([append_mask, mask[:, :, :, 0:T], append_contact_mask[:, :, :, 0:T]], dim=-2)  # [bs, 1, 208/211, T]
                clip_img_input[:, 0:1] = clip_img_input[:, 0:1] * mask    # [bs, 1/4, d, T]


            if args.input_padding:
                p2d = (8, 8, 1, 1)
                clip_img_input = F.pad(clip_img_input, p2d, 'reflect')  # masked
                clip_img = F.pad(clip_img, p2d, 'reflect')

            # forward
            clip_img_rec, z = model(clip_img_input)  # z: [bs, 256, d, T], clip_img_rec: [bs, 1, d, T]

            # loss
            clip_img_v = clip_img[:, :, :, 1:] - clip_img[:, :, :, 0:-1]
            clip_img_rec_v = clip_img_rec[:, :, :, 1:] - clip_img_rec[:, :, :, 0:-1]

            loss_rec_body = F.l1_loss(clip_img[:, 0, 0:-5], clip_img_rec[:, 0, 0:-5])   # with 1 row of pad
            loss_rec_body_v = F.l1_loss(clip_img_v[:, 0, 0:-5], clip_img_rec_v[:, 0, 0:-5])
            loss_rec_contact_lbl = bce_loss(clip_img_rec[:, 0, -5:], clip_img[:, 0, -5:])

            loss = args.weight_loss_rec_body * loss_rec_body + args.weight_loss_rec_body_v * loss_rec_body_v + \
                   args.weight_loss_rec_contact_lbl * loss_rec_contact_lbl
            loss.backward()
            optimizer.step()

            ####################### log train loss ############################
            if total_steps % args.log_step == 0:
                writer.add_scalar('train/loss_rec_body', loss_rec_body.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_body: {:.10f}'. \
                    format(step, epoch, loss_rec_body.item())
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('train/loss_rec_body_v', loss_rec_body_v.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_v: {:.10f}'. \
                    format(step, epoch, loss_rec_body_v.item())
                logger.info(print_str)
                print(print_str)


                writer.add_scalar('train/loss_rec_contact_lbl', loss_rec_contact_lbl.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_contact_lbl: {:.10f}'. \
                    format(step, epoch, loss_rec_contact_lbl.item())
                logger.info(print_str)
                print(print_str)


            ################## test loss #################################
            if total_steps % args.log_step == 0:
                loss_rec_body_test,  loss_rec_body_v_test = 0, 0
                loss_rec_contact_lbl_test = 0
                with torch.no_grad():
                    for test_step, data in tqdm(enumerate(test_dataloader)):
                        model.eval()
                        [clip_img_test] = [item.to(device) for item in data]

                        ##### mask input
                        clip_img_input_test = clip_img_test.clone()
                        bs = clip_img_test.shape[0]

                        mask_marker_n = random.randint(1, 6)
                        mask_marker_id = torch.rand(bs, mask_marker_n) * 67   # U[0, n_markers), [bs, mask_marker_n]
                        mask_marker_id = mask_marker_id.long()
                        mask_row_id1 = mask_marker_id * 3
                        if args.body_mode in ['local_markers']:  # for global traj and pelvis joint
                            mask_row_id1 = mask_row_id1 + 3 + 3
                        if args.body_mode in ['local_markers_4chan']:  # for pelvis joint
                            mask_row_id1 = mask_row_id1 + 3
                        mask_row_id2 = mask_row_id1 + 1
                        mask_row_id3 = mask_row_id2 + 1
                        for i in range(bs):
                            clip_img_input_test[i, 0, mask_row_id1[i], :] = 0.
                            clip_img_input_test[i, 0, mask_row_id2[i], :] = 0.
                            clip_img_input_test[i, 0, mask_row_id3[i], :] = 0.

                            # mask contact lbls/distance if foot marker is masked
                            if 16 in mask_marker_id[i] or 30 in mask_marker_id[i]:
                                clip_img_input_test[i, 0, -4, :] = 0.
                                clip_img_input_test[i, 0, -2, :] = 0.
                            if 47 in mask_marker_id[i] or 60 in mask_marker_id[i]:
                                clip_img_input_test[i, 0, -3, :] = 0.
                                clip_img_input_test[i, 0, -1, :] = 0.

                        if args.input_padding:
                            p2d = (8, 8, 1, 1)
                            clip_img_input_test = F.pad(clip_img_input_test, p2d, 'reflect')
                            clip_img_test = F.pad(clip_img_test, p2d, 'reflect')

                        # forward
                        clip_img_test_rec, z = model(clip_img_input_test)

                        # reconstruction loss
                        clip_img_test_v = clip_img_test[:, :, :, 1:] - clip_img_test[:, :, :, 0:-1]  # velocity
                        clip_img_test_rec_v = clip_img_test_rec[:, :, :, 1:] - clip_img_test_rec[:, :, :, 0:-1]  # velocity

                        loss_rec_body_test += F.l1_loss(clip_img_test[:, 0, 0:-5], clip_img_test_rec[:, 0, 0:-5])
                        loss_rec_body_v_test += F.l1_loss(clip_img_test_v[:, 0, 0:-5], clip_img_test_rec_v[:, 0, 0:-5])
                        loss_rec_contact_lbl_test += bce_loss(clip_img_test_rec[:, 0, -5:], clip_img_test[:, 0, -5:])

                loss_rec_body_test = loss_rec_body_test / test_step
                loss_rec_body_v_test = loss_rec_body_v_test / test_step
                loss_rec_contact_lbl_test = loss_rec_contact_lbl_test / test_step

                ####################### log test loss ############################
                writer.add_scalar('test/loss_rec_body_test', loss_rec_body_test, total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_body_test: {:.10f}'. \
                    format(step, epoch, loss_rec_body_test)
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('test/loss_rec_body_v_test', loss_rec_body_v_test, total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_body_v_test: {:.10f}'. \
                    format(step, epoch, loss_rec_body_v_test)
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('test/loss_rec_contact_lbl_test', loss_rec_contact_lbl_test, total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_contact_lbl_test: {:.10f}'. \
                    format(step, epoch, loss_rec_contact_lbl_test)
                logger.info(print_str)
                print(print_str)


            if total_steps % args.save_step == 0:
                save_path = os.path.join(writer.file_writer.get_logdir(), "AE_last_model.pkl")
                torch.save(model.state_dict(), save_path)
                logger.info('[*] last model saved\n')






if __name__ == '__main__':
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    sys.stdout.flush()

    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    train(writer, logger)


