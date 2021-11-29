import argparse
import torch
from torch.utils import data
from tqdm import tqdm
from tensorboardX import SummaryWriter
import smplx
import torch.optim as optim
import itertools

from loader.train_loader_smooth import TrainLoader
from models.AE_sep import Enc, Dec
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
parser.add_argument('--body_mode', type=str, default='global_markers',
                    choices=['global_joints', 'local_joints', 'local_markers', 'global_markers'],
                    help='which body representation to use')
parser.add_argument('--with_hand', default='True', type=lambda x: x.lower() in ['true', '1'], help='include hand or not')
parser.add_argument('--normalize', default='True', type=lambda x: x.lower() in ['true', '1'], help='normalize input motion representation or not')
parser.add_argument('--input_padding', default='True', type=lambda x: x.lower() in ['true', '1'], help='pad input motion representation or not')

# settings for network
parser.add_argument('--downsample', default='False', type=lambda x: x.lower() in ['true', '1'], help='downsample latent space or not')
parser.add_argument("--z_channel", default=64, type=int, help='channel # of latent space z')

# loss weights
parser.add_argument("--weight_loss_rec_v", default=1.0, type=float, help='weight for reconstruction loss')
parser.add_argument("--weight_loss_z_smooth", default=1000.0, type=float, help='weight for latent smoothness term')



args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('gpu id:', torch.cuda.current_device())




def train(writer, logger):
    # amass_dir = '/local/home/szhang/AMASS/amass'
    # body_model_path = '/mnt/hdd/PROX/body_models'

    smplx_model_path = os.path.join(args.body_model_path, 'smplx_model')

    amass_train_datasets = ['HumanEva', 'MPI_HDM05', 'MPI_mosh', 'Transitions_mocap',
                            'ACCAD', 'BMLhandball', 'BMLmovi', 'BioMotionLab_NTroje', 'CMU',
                            'DFaust_67', 'Eyes_Japan_Dataset', 'MPI_Limits']
    amass_test_datasets = ['TCD_handMocap', 'TotalCapture', 'SFU']
    # amass_train_datasets = ['HumanEva', 'BMLmovi']
    # amass_test_datasets = ['TCD_handMocap', 'TotalCapture']

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
    encoder = Enc(downsample=args.downsample, z_channel=args.z_channel).to(device)
    decoder = Dec(downsample=args.downsample, z_channel=args.z_channel).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  itertools.chain(encoder.parameters(), decoder.parameters())),
                           lr=args.lr)


    ################################## start training #########################################
    total_steps = 0
    for epoch in range(args.num_epoch):
        for step, data in tqdm(enumerate(train_dataloader)):
            encoder.train()
            decoder.train()

            total_steps += 1
            [clip_img] = [item.to(device) for item in data]

            optimizer.zero_grad()

            # netowrk input/output: motion velocity
            clip_img_v = clip_img[:, :, :, 1:] - clip_img[:, :, :, 0:-1]  # T=119

            if args.input_padding:
                p2d = (8, 8, 1, 1)
                clip_img_v = F.pad(clip_img_v, p2d, 'reflect')

            # forward
            z_v, input_size, x_down1_size, x_down2_size, x_down3_size, x_down4_size = encoder(clip_img_v)
            clip_img_v_rec = decoder(z_v, input_size, x_down1_size, x_down2_size, x_down3_size, x_down4_size)

            # reconstruction loss
            loss_rec_v = F.l1_loss(clip_img_v, clip_img_v_rec)
            # smooth constraints on z
            z_a = z_v[:, :, :, 1:] - z_v[:, :, :, 0:-1]
            loss_z_smooth = torch.mean(z_a ** 2)

            loss = args.weight_loss_rec_v * loss_rec_v + \
                   args.weight_loss_z_smooth * loss_z_smooth
            loss.backward()
            optimizer.step()

            ####################### log train loss ############################
            if total_steps % args.log_step == 0:
                writer.add_scalar('train/loss_rec_v', loss_rec_v.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_v: {:.10f}'. \
                    format(step, epoch, loss_rec_v.item())
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('train/loss_z_smooth', loss_z_smooth.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_z_smooth: {:.10f}'. \
                    format(step, epoch, loss_z_smooth.item())
                logger.info(print_str)
                print(print_str)



            ################## test loss #################################
            if total_steps % args.log_step == 0:
                loss_rec_test_v = 0
                loss_z_smooth_test = 0
                with torch.no_grad():
                    for test_step, data in tqdm(enumerate(test_dataloader)):
                        encoder.eval()
                        decoder.eval()
                        [clip_img_test] = [item.to(device) for item in data]

                        # netowrk input/output: velocity
                        clip_img_v_test = clip_img_test[:, :, :, 1:] - clip_img_test[:, :, :, 0:-1]

                        if args.input_padding:
                            p2d = (8, 8, 1, 1)
                            clip_img_v_test = F.pad(clip_img_v_test, p2d, 'reflect')

                        z_v, input_size, x_down1_size, x_down2_size, x_down3_size, x_down4_size = encoder(clip_img_v_test)
                        clip_img_v_test_rec = decoder(z_v, input_size, x_down1_size, x_down2_size, x_down3_size, x_down4_size)

                        # reconstruction loss
                        loss_rec_test_v += F.l1_loss(clip_img_v_test, clip_img_v_test_rec)
                        # smooth loss
                        z_a = z_v[:, :, :, 1:] - z_v[:, :, :, 0:-1]
                        loss_z_smooth_test += torch.mean(z_a ** 2)


                loss_rec_test_v = loss_rec_test_v / test_step
                loss_z_smooth_test = loss_z_smooth_test / test_step

                ####################### log test loss ############################
                writer.add_scalar('test/loss_rec_test_v', loss_rec_test_v, total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_test_v: {:.10f}'. \
                    format(step, epoch, loss_rec_test_v)
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('test/loss_z_smooth_test', loss_z_smooth_test.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_z_smooth_test: {:.10f}'. \
                    format(step, epoch, loss_z_smooth_test.item())
                logger.info(print_str)
                print(print_str)


            if total_steps % args.save_step == 0:
                save_path = os.path.join(writer.file_writer.get_logdir(), "Enc_last_model.pkl")
                torch.save(encoder.state_dict(), save_path)
                save_path = os.path.join(writer.file_writer.get_logdir(), "Dec_last_model.pkl")
                torch.save(decoder.state_dict(), save_path)
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


