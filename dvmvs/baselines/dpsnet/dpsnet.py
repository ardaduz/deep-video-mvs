from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data
from torch.autograd import Variable

from dvmvs.utils import freeze_batchnorm

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1, h, w)).type_as(depth)  # [1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1, h, w)).type_as(depth)  # [1, H, W]
    ones = Variable(torch.ones(1, h, w)).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert (all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).contiguous().view(b, 3, -1).cuda()  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2 * (X / Z) / (w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(b, h, w, 2)


def inverse_warp(feat, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        feat: the source feature (where to sample pixels) -- [B, CH, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B34')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')

    assert (intrinsics_inv.size() == intrinsics.size())

    batch_size, _, feat_height, feat_width = feat.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)

    pose_mat = pose
    pose_mat = pose_mat.cuda()

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:], padding_mode)  # [B,H,W,2]
    projected_feat = torch.nn.functional.grid_sample(feat, src_pixel_coords, padding_mode=padding_mode, align_corners=True)

    return projected_feat


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left = F.pad(torch.index_select(left, 3, Variable(torch.LongTensor([i for i in range(shift, width)])).cuda()), (shift, 0, 0, 0))
        shifted_right = F.pad(torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width - shift)])).cuda()), (shift, 0, 0, 0))
        out = torch.cat((shifted_left, shifted_right), 1).view(batch, filters * 2, 1, height, width)
        return out


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


def convtext(in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )


class PSNet(nn.Module):
    def __init__(self, nlabel, mindepth):
        super(PSNet, self).__init__()
        self.nlabel = nlabel
        self.mindepth = mindepth

        self.feature_extraction = feature_extraction()

        self.convs = nn.Sequential(
            convtext(33, 128, 3, 1, 1),
            convtext(128, 128, 3, 1, 2),
            convtext(128, 128, 3, 1, 4),
            convtext(128, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1)
        )

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, ref, targets, pose, intrinsics, intrinsics_inv):

        intrinsics4 = intrinsics.clone()
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics4[:, :2, :] = intrinsics4[:, :2, :] / 4
        intrinsics_inv4[:, :2, :2] = intrinsics_inv4[:, :2, :2] * 4

        refimg_fea = self.feature_extraction(ref)

        disp2depth = Variable(torch.ones(refimg_fea.size(0), refimg_fea.size(2), refimg_fea.size(3))).cuda() * self.mindepth * self.nlabel
        for j, target in enumerate(targets):
            cost = Variable(
                torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.nlabel, refimg_fea.size()[2], refimg_fea.size()[3]).zero_()).cuda()
            targetimg_fea = self.feature_extraction(target)
            for i in range(self.nlabel):
                depth = torch.div(disp2depth, i + 1e-16)
                targetimg_fea_t = inverse_warp(targetimg_fea, depth, pose[j], intrinsics4, intrinsics_inv4)
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea_t

            cost = cost.contiguous()
            cost0 = self.dres0(cost)
            cost0 = self.dres1(cost0) + cost0
            cost0 = self.dres2(cost0) + cost0
            cost0 = self.dres3(cost0) + cost0
            cost0 = self.dres4(cost0) + cost0
            cost0 = self.classify(cost0)

            if j == 0:
                costs = cost0
            else:
                costs = costs + cost0

        costs = costs / len(targets)

        costss = Variable(torch.FloatTensor(refimg_fea.size()[0], 1, self.nlabel, refimg_fea.size()[2], refimg_fea.size()[3]).zero_()).cuda()
        for i in range(self.nlabel):
            costt = costs[:, :, i, :, :]
            costss[:, :, i, :, :] = self.convs(torch.cat([refimg_fea, costt], 1)) + costt

        costs = F.interpolate(costs, [self.nlabel, ref.size()[2], ref.size()[3]], mode='trilinear', align_corners=False)
        costs = torch.squeeze(costs, 1)
        pred0 = F.softmax(costs, dim=1)
        pred0 = disparityregression(self.nlabel)(pred0)
        depth0 = self.mindepth * self.nlabel / (pred0.unsqueeze(1) + 1e-16)

        costss = F.interpolate(costss, [self.nlabel, ref.size()[2], ref.size()[3]], mode='trilinear', align_corners=False)
        costss = torch.squeeze(costss, 1)
        pred = F.softmax(costss, dim=1)
        pred = disparityregression(self.nlabel)(pred)
        depth = self.mindepth * self.nlabel / (pred.unsqueeze(1) + 1e-16)

        return depth0, depth

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(PSNet, self).train(mode)
        self.apply(freeze_batchnorm)
