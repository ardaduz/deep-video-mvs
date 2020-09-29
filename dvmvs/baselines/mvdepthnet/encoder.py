import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from dvmvs.utils import freeze_batchnorm


def down_conv_layer(input_channels, output_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=1,
            bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=2,
            bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU())


def conv_layer(input_channels, output_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU())


def depth_layer(input_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, 1, 3, padding=1), nn.Sigmoid())


def refine_layer(input_channels):
    return nn.Conv2d(input_channels, 1, 3, padding=1)


def up_conv_layer(input_channels, output_channels, kernel_size):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU())


def get_trainable_number(variable):
    num = 1
    shape = list(variable.shape)
    for i in shape:
        num *= i
    return num


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = down_conv_layer(67, 128, 7)
        self.conv2 = down_conv_layer(128, 256, 5)
        self.conv3 = down_conv_layer(256, 512, 3)
        self.conv4 = down_conv_layer(512, 512, 3)
        self.conv5 = down_conv_layer(512, 512, 3)

    def getVolume(self, left_image, right_image, KRKiUV_T, KT_T):
        idepth_base = 1.0 / 50.0
        idepth_step = (1.0 / 0.5 - 1.0 / 50.0) / 63.0

        costvolume = Variable(
            torch.cuda.FloatTensor(left_image.shape[0], 64,
                                   left_image.shape[2], left_image.shape[3]))

        image_height = 256
        image_width = 320
        batch_number = left_image.shape[0]

        normalize_base = torch.cuda.FloatTensor(
            [image_width / 2.0, image_height / 2.0])

        normalize_base = normalize_base.unsqueeze(0).unsqueeze(-1)

        for depth_i in range(64):
            this_depth = 1.0 / (idepth_base + depth_i * idepth_step)
            transformed = KRKiUV_T * this_depth + KT_T
            demon = transformed[:, 2, :].unsqueeze(1)
            warp_uv = transformed[:, 0: 2, :] / (demon + 1e-6)
            warp_uv = (warp_uv - normalize_base) / normalize_base
            warp_uv = warp_uv.view(
                batch_number, 2, image_width,
                image_height)

            warp_uv = Variable(warp_uv.permute(
                0, 3, 2, 1))
            warped = F.grid_sample(right_image, warp_uv)

            costvolume[:, depth_i, :, :] = torch.sum(
                torch.abs(warped - left_image), dim=1)
        return costvolume

    def forward(self, image, plane_sweep_volume):
        x = torch.cat((image, plane_sweep_volume), 1)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = self.conv5(conv4)

        return [conv5, conv4, conv3, conv2, conv1]

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(Encoder, self).train(mode)
        self.apply(freeze_batchnorm)
