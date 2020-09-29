import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .resnet_s2d import resnet50


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Unpool(nn.Module):
    """Unpool: 2*2 unpooling with zero padding"""

    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride))
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights.to(x.device), stride=self.stride, groups=self.num_channels)


class Gudi_UpProj_Block(nn.Module):
    """UpProjection block from CSPN paper (Cheng et.al.)"""

    def __init__(self, in_channels, out_channels, oheight=0, owidth=0, side_channels=0, do_5x5=True, interp_nearest=True):
        super(Gudi_UpProj_Block, self).__init__()

        if do_5x5:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if do_5x5:
            self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        else:
            self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        self.interp_nearest = interp_nearest

    def _up_pooling(self, x, scale):

        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)

        if self.oheight != 0 and self.owidth != 0:
            x = x[:, :, 0:self.oheight, 0:self.owidth]
        mask = torch.zeros_like(x)
        for h in range(0, self.oheight, 2):
            for w in range(0, self.owidth, 2):
                mask[:, :, h, w] = 1
        x = torch.mul(mask, x)
        return x

    def forward(self, x):
        if self.interp_nearest:
            x = self._up_pooling(x, 2)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class Gudi_UpProj_Block_Cat(nn.Module):
    """UpProjection block with concatenation from CSPN paper (Cheng et.al.)"""

    def __init__(self, in_channels, out_channels, oheight=0, owidth=0, side_channels=0, do_5x5=True, interp_nearest=True):
        super(Gudi_UpProj_Block_Cat, self).__init__()

        if do_5x5:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        out_ch = out_channels + side_channels
        self.conv1_1 = nn.Conv2d(out_ch, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if do_5x5:
            self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        else:
            self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = Unpool(in_channels)
        self.interp_nearest = interp_nearest

    def _up_pooling(self, x, scale):

        x = self._up_pool(x)

        if self.oheight != 0 and self.owidth != 0:
            x = x.narrow(2, 0, self.oheight)
            x = x.narrow(3, 0, self.owidth)
        return x

    def forward(self, x, side_input):

        if self.interp_nearest:
            if side_input.shape[2] % x.shape[2] == 0:
                x = self._up_pooling(x, 2)
            else:
                x = F.interpolate(x, size=(side_input.shape[2], side_input.shape[3]), mode='nearest')
        else:
            x = F.interpolate(x, size=(side_input.shape[2], side_input.shape[3]), mode='bilinear')

        out = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat((out, side_input), 1)

        out = self.relu(self.bn1_1(self.conv1_1(out)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))

        out += short_cut
        out = self.relu(out)
        return out


class dilated_conv3x3(nn.Module):
    """Dilated convolutions"""

    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(dilated_conv3x3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


class ASPP(nn.Module):
    """Altrous Spatial Pyramid Pooling block"""

    def __init__(self, in_channels):
        super(ASPP, self).__init__()

        out_channels = in_channels
        self.daspp_1 = dilated_conv3x3(out_channels, out_channels // 2, dilation_rate=3)
        self.relu = nn.ReLU(inplace=True)
        self.daspp_2 = dilated_conv3x3(int(1.5 * out_channels), out_channels // 2, dilation_rate=6)
        self.daspp_3 = dilated_conv3x3(int(2 * out_channels), out_channels // 2, dilation_rate=12)
        self.daspp_4 = dilated_conv3x3(int(2.5 * out_channels), out_channels // 2, dilation_rate=18)
        self.daspp_5 = dilated_conv3x3(int(3 * out_channels), out_channels // 2, dilation_rate=24)

        self.convf = nn.Conv2d(int(3.5 * out_channels), out_channels, kernel_size=3, padding=1, bias=False)
        self.bnf = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_inp = x
        x1_d1 = self.daspp_1(x)

        x = torch.cat((x, x1_d1), 1)
        x1_d2 = self.daspp_2(x)

        x = torch.cat((x, x1_d2), 1)
        x1_d3 = self.daspp_3(x)

        x = torch.cat((x, x1_d3), 1)
        x1_d4 = self.daspp_4(x)

        x = torch.cat((x, x1_d4), 1)
        x1_d5 = self.daspp_5(x)
        x = torch.cat((x_inp, x1_d1, x1_d2, x1_d3, x1_d4, x1_d5), 1)

        out = self.relu(self.bnf(self.convf(x)))
        return out


class SparsetoDenseNet(BaseModel):
    """Sparse to Dense Network """

    default_config = {

        'model_type': 'resnet50',
        'input_shape': (240, 320, 1),
        'min_depth': 0.5,
        'max_depth': 10.0,
        'multiscale': True,
        'do_5x5': True,
        'interp_n': True,

    }

    def _init(self):

        ##Encoder for sparse depth
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        in_channel = 1

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        pretrained_features = resnet50()
        c_out = [int(1.25 * 2048), int(1.25 * 1024), int(1.25 * 512), int(1.25 * 256), int(1.25 * 64)]

        self.conv1 = pretrained_features.conv1
        self.bn1 = pretrained_features.bn1
        self.maxpool = pretrained_features.maxpool
        self.layer1 = pretrained_features.layer1
        self.layer2 = pretrained_features.layer2
        self.layer3 = pretrained_features.layer3
        self.layer4 = pretrained_features.layer4

        d0, d1, d2, d3, d4 = 512, 256, 128, 64, 32

        # Decoder for  sparse to dense

        block = Gudi_UpProj_Block_Cat
        block_simple = Gudi_UpProj_Block

        h = self.config['input_shape'][0]
        w = self.config['input_shape'][1]
        self.gud_up_proj_layer1 = self._make_gud_up_conv_layer(block, c_out[0], d0, math.ceil(h / 16), math.ceil(w / 16), c_out[1], self.config['do_5x5'],
                                                               self.config['interp_n'])
        self.gud_up_proj_layer2 = self._make_gud_up_conv_layer(block, d0, d1, math.ceil(h / 8), math.ceil(w / 8), c_out[2], self.config['do_5x5'],
                                                               self.config['interp_n'])
        self.ASPP = ASPP(d1)

        self.gud_up_proj_layer3 = self._make_gud_up_conv_layer(block, d1, d2, math.ceil(h / 4), math.ceil(w / 4), c_out[3], self.config['do_5x5'],
                                                               self.config['interp_n'])
        self.gud_up_proj_layer4 = self._make_gud_up_conv_layer(block, d2, d3, math.ceil(h / 2), math.ceil(w / 2), c_out[4], self.config['do_5x5'],
                                                               self.config['interp_n'])
        self.gud_up_proj_layer5 = self._make_gud_up_conv_layer(block_simple, d3, d4, h, w, self.config['do_5x5'], self.config['interp_n'])
        self.conv_final = nn.Conv2d(d4, 1, kernel_size=3, stride=1, padding=1, bias=True)

        if self.config['multiscale']:
            self.conv_scale8 = nn.Conv2d(d1, 1, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv_scale4 = nn.Conv2d(d2, 1, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv_scale2 = nn.Conv2d(d3, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def _make_gud_up_conv_layer(self, up_proj_block, in_channels, out_channels, oheight, owidth, side_ch=0, do_5x5=True, interp_nearest=True):
        return up_proj_block(in_channels, out_channels, oheight, owidth, side_ch, do_5x5, interp_nearest)

    def _forward(self, data):

        # Inputs from previous modules 
        anchor_keypoints = data['anchor_keypoints']
        keypoints_3d = data['keypoints_3d']
        range_mask = data['range_mask']
        features = data['features']
        skip_half = data['skip_half']
        skip_quarter = data['skip_quarter']
        skip_eight = data['skip_eight']
        skip_sixteenth = data['skip_sixteenth']
        sequence_length = data['sequence_length']

        del data

        # Impute learnt sparse depth into a sparse image
        sparse_depth_learnt = torch.zeros((anchor_keypoints.shape[0], self.config['input_shape'][0], self.config['input_shape'][1])).to(anchor_keypoints.device)
        anchor_keypoints_index = anchor_keypoints.long()
        bselect = torch.arange(anchor_keypoints.shape[0], dtype=torch.long)
        bselect = bselect.unsqueeze(1).unsqueeze(1)
        bselect = bselect.repeat(1, anchor_keypoints_index.shape[1], 1).to(anchor_keypoints.device)
        anchor_keypoints_indexchunk = torch.cat((bselect, anchor_keypoints_index[:, :, [1]], anchor_keypoints_index[:, :, [0]]), 2)
        anchor_keypoints_indexchunk = anchor_keypoints_indexchunk.view(-1, 3).t()

        kp3d_val = keypoints_3d[:, :, 2].view(-1, 1).t()
        kp3d_val = torch.clamp(kp3d_val, min=0.0, max=self.config['max_depth'])
        kp3d_filter = (range_mask > 0).view(-1, 1).t()
        kp3d_filter = (kp3d_filter) & (kp3d_val > self.config['min_depth']) & (kp3d_val < self.config['max_depth'])
        kp3d_val = kp3d_val * kp3d_filter.float()
        sparse_depth_learnt[anchor_keypoints_indexchunk.chunk(chunks=3, dim=0)] = kp3d_val
        sparse_depth_learnt = sparse_depth_learnt.unsqueeze(1)

        pred = {}

        # Forward pass
        x = self.relu(self.bn1(self.conv1(sparse_depth_learnt)))
        skip_half = torch.cat((x, skip_half), 1)

        x = self.maxpool(x)
        x = self.layer1(x)
        skip_quarter = torch.cat((x, skip_quarter), 1)

        x = self.layer2(x)
        skip_eight = torch.cat((x, skip_eight), 1)

        x = self.layer3(x)
        skip_sixteenth = torch.cat((x, skip_sixteenth), 1)

        x = self.layer4(x)
        x = torch.cat((features, x), 1)  # 160

        x = self.gud_up_proj_layer1(x, skip_sixteenth)
        x = self.gud_up_proj_layer2(x, skip_eight)
        x = self.ASPP(x)

        if self.config['multiscale']:
            x_8 = self.conv_scale8(x)

        x = self.gud_up_proj_layer3(x, skip_quarter)

        if self.config['multiscale']:
            x_4 = self.conv_scale4(x)

        x = self.gud_up_proj_layer4(x, skip_half)

        if self.config['multiscale']:
            x_2 = self.conv_scale2(x)

        x = self.gud_up_proj_layer5(x)
        x = self.conv_final(x)

        if self.config['multiscale']:
            pred['multiscale'] = [x_2, x_4, x_8]

        depth_dense = x

        pred['dense_depth'] = depth_dense

        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self):
        raise NotImplementedError
