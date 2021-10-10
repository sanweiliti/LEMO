import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncBlock(nn.Module):
    def __init__(self, nin, nout, downsample=True, kernel=3):
        super(EncBlock, self).__init__()
        self.downsample = downsample
        padding = kernel // 2

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=1, padding=padding),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=nout, out_channels=nout, kernel_size=kernel, stride=1, padding=padding),
            nn.LeakyReLU(0.2),
        )

        if self.downsample:
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pooling = nn.MaxPool2d(kernel_size=(3,3), stride=(2, 1), padding=1)

    def forward(self, input):
        output = self.main(input)
        output = self.pooling(output)
        return output



class DecBlock(nn.Module):
    def __init__(self, nin, nout, upsample=True, kernel=3):
        super(DecBlock, self).__init__()
        self.upsample = upsample

        padding = kernel // 2
        if upsample:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=2, padding=padding)
        else:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=(2, 1), padding=padding)
        self.deconv2 = nn.ConvTranspose2d(in_channels=nout, out_channels=nout, kernel_size=kernel, stride=1, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, input, out_size):
        output = self.deconv1(input, output_size=out_size)
        output = self.leaky_relu(output)
        output = self.leaky_relu(self.deconv2(output))
        return output


class DecBlock_output(nn.Module):
    def __init__(self, nin, nout, upsample=True, kernel=3):
        super(DecBlock_output, self).__init__()
        self.upsample = upsample
        padding = kernel // 2

        if upsample:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=2, padding=padding)
        else:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=(2, 1), padding=padding)
        self.deconv2 = nn.ConvTranspose2d(in_channels=nout, out_channels=nout, kernel_size=kernel, stride=1, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, input, out_size):
        output = self.deconv1(input, output_size=out_size)
        output = self.leaky_relu(output)
        output = self.deconv2(output)
        return output



class AE(nn.Module):
    def __init__(self, downsample=True, in_channel=1, kernel=3):
        super(AE, self).__init__()
        self.enc_blc1 = EncBlock(nin=in_channel, nout=32, downsample=downsample, kernel=kernel)
        self.enc_blc2 = EncBlock(nin=32, nout=64, downsample=downsample, kernel=kernel)
        self.enc_blc3 = EncBlock(nin=64, nout=128, downsample=downsample, kernel=kernel)
        self.enc_blc4 = EncBlock(nin=128, nout=256, downsample=downsample, kernel=kernel)
        self.enc_blc5 = EncBlock(nin=256, nout=256, downsample=downsample, kernel=kernel)

        self.dec_blc1 = DecBlock(nin=256, nout=256, upsample=downsample, kernel=kernel)
        self.dec_blc2 = DecBlock(nin=256, nout=128, upsample=downsample, kernel=kernel)
        self.dec_blc3 = DecBlock(nin=128, nout=64, upsample=downsample, kernel=kernel)
        self.dec_blc4 = DecBlock(nin=64, nout=32, upsample=downsample, kernel=kernel)
        self.dec_blc5 = DecBlock_output(nin=32, nout=1, upsample=downsample, kernel=kernel)

    def forward(self, input):
        # input: [bs, 1, d, T]
        # [bs, 1, 51/145/75, 120] (smplx_params, no hand, vposer/6d_rot)/(joints, no hand)
        x_down1 = self.enc_blc1(input)  # [bs, 32, 26, 60]
        x_down2 = self.enc_blc2(x_down1)  # [bs, 64, 13, 30]
        x_down3 = self.enc_blc3(x_down2)  # [bs, 128, 7, 15]
        x_down4 = self.enc_blc4(x_down3)  # [bs, 256, 4, 8]
        z = self.enc_blc5(x_down4)  # [bs, 256, 2/5/3, 4]  (smplx_params, no hand, vposer/6d_rot)/(joints, no hand)

        x_up4 = self.dec_blc1(z, x_down4.size())  # [bs, 256, 4, 8]
        x_up3 = self.dec_blc2(x_up4, x_down3.size())   # [bs, 128, 7, 15]
        x_up2 = self.dec_blc3(x_up3, x_down2.size())   # [bs, 64, 13, 30]
        x_up1 = self.dec_blc4(x_up2, x_down1.size())   # [bs, 32, 26, 60]
        output = self.dec_blc5(x_up1, input.size())

        return output, z


# #################### with BN ###########################
#
# class EncBlock_BN(nn.Module):
#     def __init__(self, nin, nout, downsample=True):
#         super(EncBlock_BN, self).__init__()
#         self.downsample = downsample
#
#         self.main = nn.Sequential(
#             nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(nout),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(in_channels=nout, out_channels=nout, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(nout),
#             nn.LeakyReLU(0.2),
#         )
#         if self.downsample:
#             self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         else:
#             self.pooling = nn.MaxPool2d(kernel_size=(3,3), stride=(2, 1), padding=1)
#
#     def forward(self, input):
#         output = self.main(input)
#         output = self.pooling(output)
#         return output
#
#
#
# class DecBlock_BN(nn.Module):
#     def __init__(self, nin, nout, upsample=True):
#         super(DecBlock_BN, self).__init__()
#         self.upsample = upsample
#         if upsample:
#             self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=2, padding=1)
#         else:
#             self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=(2, 1), padding=1)
#         self.bn1 = nn.BatchNorm2d(nout)
#         self.deconv2 = nn.ConvTranspose2d(in_channels=nout, out_channels=nout, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(nout)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#
#     def forward(self, input, out_size):
#         output = self.deconv1(input, output_size=out_size)
#         output = self.leaky_relu(self.bn1(output))
#         output = self.leaky_relu(self.bn2(self.deconv2(output)))
#         return output
#
#
# class DecBlock_output_BN(nn.Module):
#     def __init__(self, nin, nout, upsample=True):
#         super(DecBlock_output_BN, self).__init__()
#         self.upsample = upsample
#         if upsample:
#             self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=2, padding=1)
#         else:
#             self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=(2, 1), padding=1)
#         self.bn1 = nn.BatchNorm2d(nout)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.deconv2 = nn.ConvTranspose2d(in_channels=nout, out_channels=nout, kernel_size=3, stride=1, padding=1)
#
#
#     def forward(self, input, out_size):
#         output = self.deconv1(input, output_size=out_size)
#         output = self.leaky_relu(self.bn1(output))
#         output = self.deconv2(output)
#         return output


