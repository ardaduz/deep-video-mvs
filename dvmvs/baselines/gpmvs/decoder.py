import torch
import torch.nn as nn
import torch.nn.functional as F
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
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
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


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.upconv5 = up_conv_layer(512, 512, 3)
        self.iconv5 = conv_layer(1024, 512, 3)  # input upconv5 + conv4

        self.upconv4 = up_conv_layer(512, 512, 3)
        self.iconv4 = conv_layer(1024, 512, 3)  # input upconv4 + conv3
        self.disp4 = depth_layer(512)

        self.upconv3 = up_conv_layer(512, 256, 3)
        self.iconv3 = conv_layer(
            513, 256, 3)  # input upconv3 + conv2 + disp4 = 256 + 256 + 1 = 513
        self.disp3 = depth_layer(256)

        self.upconv2 = up_conv_layer(256, 128, 3)
        self.iconv2 = conv_layer(
            257, 128, 3)  # input upconv2 + conv1 + disp3 = 128 + 128 + 1 =  257
        self.disp2 = depth_layer(128)

        self.upconv1 = up_conv_layer(128, 64, 3)
        self.iconv1 = conv_layer(65, 64,
                                 3)  # input upconv1 + disp2 = 64 + 1 = 65
        self.disp1 = depth_layer(64)

    def forward(self, conv5, conv4, conv3, conv2, conv1):

        upconv5 = self.upconv5(conv5)

        iconv5 = self.iconv5(torch.cat((upconv5, conv4), 1))

        upconv4 = self.upconv4(iconv5)

        iconv4 = self.iconv4(torch.cat((upconv4, conv3), 1))
        disp4 = 2.0 * self.disp4(iconv4)
        udisp4 = F.interpolate(disp4, scale_factor=2)

        upconv3 = self.upconv3(iconv4)

        iconv3 = self.iconv3(torch.cat((upconv3, conv2, udisp4), 1))
        disp3 = 2.0 * self.disp3(iconv3)
        udisp3 = F.interpolate(disp3, scale_factor=2)

        upconv2 = self.upconv2(iconv3)

        iconv2 = self.iconv2(torch.cat((upconv2, conv1, udisp3), 1))
        disp2 = 2.0 * self.disp2(iconv2)
        udisp2 = F.interpolate(disp2, scale_factor=2)

        upconv1 = self.upconv1(iconv2)
        iconv1 = self.iconv1(torch.cat((upconv1, udisp2), 1))
        disp1 = 2.0 * self.disp1(iconv1)

        return [disp1, disp2, disp3, disp4]

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(Decoder, self).train(mode)
        self.apply(freeze_batchnorm)
