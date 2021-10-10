import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncBlock(nn.Module):
    def __init__(self, nin, nout, downsample=True):
        super(EncBlock, self).__init__()
        self.downsample = downsample

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=nout, out_channels=nout, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        output = self.main(input)
        if not self.downsample:
            return output
        else:
            output = self.pooling(output)
            return output



class DecBlock(nn.Module):
    def __init__(self, nin, nout, upsample=True):
        super(DecBlock, self).__init__()
        self.upsample = upsample
        if upsample:
            deconv_stride = 2
        else:
            deconv_stride = 1

        self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=deconv_stride, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=nout, out_channels=nout, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, input, out_size):
        output = self.deconv1(input, output_size=out_size)
        output = self.leaky_relu(output)
        output = self.leaky_relu(self.deconv2(output))
        return output


class DecBlock_output(nn.Module):
    def __init__(self, nin, nout, upsample=True):
        super(DecBlock_output, self).__init__()
        self.upsample = upsample
        if upsample:
            deconv_stride = 2
        else:
            deconv_stride = 1

        self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=deconv_stride, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=nout, out_channels=nout, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, input, out_size):
        output = self.deconv1(input, output_size=out_size)
        output = self.leaky_relu(output)
        output = self.deconv2(output)
        return output




class Enc(nn.Module):
    def __init__(self, downsample=True, z_channel=64):
        super(Enc, self).__init__()
        if z_channel == 256:
            channel_2, channel_3 = 128, 256
        elif z_channel == 64:
            channel_2, channel_3 = 64, 64
        self.enc_blc1 = EncBlock(nin=1, nout=32, downsample=downsample)
        self.enc_blc2 = EncBlock(nin=32, nout=64, downsample=downsample)
        self.enc_blc3 = EncBlock(nin=64, nout=channel_2, downsample=downsample)
        self.enc_blc4 = EncBlock(nin=channel_2, nout=channel_3, downsample=downsample)
        self.enc_blc5 = EncBlock(nin=channel_3, nout=channel_3, downsample=downsample)


    def forward(self, input):
        # input: [bs, 1, d, T]
        # [bs, 1, 51/145/75, 120] (smplx_params, no hand, vposer/6d_rot)/(joints, no hand)
        x_down1 = self.enc_blc1(input)  # [bs, 32, 26, 60]
        x_down2 = self.enc_blc2(x_down1)  # [bs, 64, 13, 30]
        x_down3 = self.enc_blc3(x_down2)  # [bs, 128, 7, 15]
        x_down4 = self.enc_blc4(x_down3)  # [bs, 256, 4, 8]
        z = self.enc_blc5(x_down4)  # [bs, 256, 2/5/3, 4]  (smplx_params, no hand, vposer/6d_rot)/(joints, no hand)
        return z, input.size(), x_down1.size(), x_down2.size(), x_down3.size(), x_down4.size()


class Dec(nn.Module):
    def __init__(self, downsample=True, z_channel=64):
        super(Dec, self).__init__()
        if z_channel == 256:
            channel_2, channel_3 = 128, 256
        elif z_channel == 64:
            channel_2, channel_3 = 64, 64

        self.dec_blc1 = DecBlock(nin=channel_3, nout=channel_3, upsample=downsample)
        self.dec_blc2 = DecBlock(nin=channel_3, nout=channel_2, upsample=downsample)
        self.dec_blc3 = DecBlock(nin=channel_2, nout=64, upsample=downsample)
        self.dec_blc4 = DecBlock(nin=64, nout=32, upsample=downsample)
        self.dec_blc5 = DecBlock_output(nin=32, nout=1, upsample=downsample)


    def forward(self, z, input_size, x_down1_size, x_down2_size, x_down3_size, x_down4_size):
        x_up4 = self.dec_blc1(z, x_down4_size)  # [bs, 256, 4, 8]
        x_up3 = self.dec_blc2(x_up4, x_down3_size)   # [bs, 128, 7, 15]
        x_up2 = self.dec_blc3(x_up3, x_down2_size)   # [bs, 64, 13, 30]
        x_up1 = self.dec_blc4(x_up2, x_down1_size)   # [bs, 32, 26, 60]
        output = self.dec_blc5(x_up1, input_size)
        return output