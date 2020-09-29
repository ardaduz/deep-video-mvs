import torch
import torch.nn as nn
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