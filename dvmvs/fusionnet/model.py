from collections import OrderedDict

import torch
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork

from dvmvs.config import Config
from dvmvs.convlstm import MVSLayernormConvLSTMCell
from dvmvs.layers import conv_layer, depth_layer_3x3

fpn_output_channels = 32
hyper_channels = 32


class StandardLayer(torch.nn.Module):
    def __init__(self, channels, kernel_size, apply_bn_relu):
        super(StandardLayer, self).__init__()
        self.conv1 = conv_layer(input_channels=channels,
                                output_channels=channels,
                                stride=1,
                                kernel_size=kernel_size,
                                apply_bn_relu=True)
        self.conv2 = conv_layer(input_channels=channels,
                                output_channels=channels,
                                stride=1,
                                kernel_size=kernel_size,
                                apply_bn_relu=apply_bn_relu)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownconvolutionLayer(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(DownconvolutionLayer, self).__init__()
        self.down_conv = conv_layer(input_channels=input_channels,
                                    output_channels=output_channels,
                                    stride=2,
                                    kernel_size=kernel_size,
                                    apply_bn_relu=True)

    def forward(self, x):
        x = self.down_conv(x)
        return x


class UpconvolutionLayer(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(UpconvolutionLayer, self).__init__()
        self.conv = conv_layer(input_channels=input_channels,
                               output_channels=output_channels,
                               stride=1,
                               kernel_size=kernel_size,
                               apply_bn_relu=True)

    def forward(self, x):
        x = torch.nn.functional.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x


class EncoderBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(EncoderBlock, self).__init__()
        self.down_convolution = DownconvolutionLayer(input_channels=input_channels,
                                                     output_channels=output_channels,
                                                     kernel_size=kernel_size)

        self.standard_convolution = StandardLayer(channels=output_channels,
                                                  kernel_size=kernel_size,
                                                  apply_bn_relu=True)

    def forward(self, x):
        x = self.down_convolution(x)
        x = self.standard_convolution(x)
        return x


class DecoderBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, apply_bn_relu, plus_one):
        super(DecoderBlock, self).__init__()
        # Upsample the inpput coming from previous layer
        self.up_convolution = UpconvolutionLayer(input_channels=input_channels,
                                                 output_channels=output_channels,
                                                 kernel_size=kernel_size)

        if plus_one:
            next_input_channels = input_channels + 1
        else:
            next_input_channels = input_channels

        # Aggregate skip and upsampled input
        self.convolution1 = conv_layer(input_channels=next_input_channels,
                                       output_channels=output_channels,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       apply_bn_relu=True)

        # Learn from aggregation
        self.convolution2 = conv_layer(input_channels=output_channels,
                                       output_channels=output_channels,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       apply_bn_relu=apply_bn_relu)

    def forward(self, x, skip, depth):
        x = self.up_convolution(x)

        if depth is None:
            x = torch.cat([x, skip], dim=1)
        else:
            depth = torch.nn.functional.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, skip, depth], dim=1)

        x = self.convolution1(x)
        x = self.convolution2(x)
        return x


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        backbone_mobile_layers = list(models.mnasnet1_0(pretrained=True).layers.children())

        self.layer1 = torch.nn.Sequential(*backbone_mobile_layers[0:8])
        self.layer2 = torch.nn.Sequential(*backbone_mobile_layers[8:9])
        self.layer3 = torch.nn.Sequential(*backbone_mobile_layers[9:10])
        self.layer4 = torch.nn.Sequential(*backbone_mobile_layers[10:12])
        self.layer5 = torch.nn.Sequential(*backbone_mobile_layers[12:14])

    def forward(self, image):
        layer1 = self.layer1(image)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        return layer1, layer2, layer3, layer4, layer5


class FeatureShrinker(torch.nn.Module):
    def __init__(self):
        super(FeatureShrinker, self).__init__()
        self.fpn = FeaturePyramidNetwork(in_channels_list=[16, 24, 40, 96, 320],
                                         out_channels=fpn_output_channels,
                                         extra_blocks=None)

    def forward(self, layer1, layer2, layer3, layer4, layer5):
        fpn_input = OrderedDict()
        fpn_input['layer1'] = layer1
        fpn_input['layer2'] = layer2
        fpn_input['layer3'] = layer3
        fpn_input['layer4'] = layer4
        fpn_input['layer5'] = layer5
        fpn_output = self.fpn(fpn_input)

        features_half = fpn_output['layer1']
        features_quarter = fpn_output['layer2']
        features_one_eight = fpn_output['layer3']
        features_one_sixteen = fpn_output['layer4']

        return features_half, features_quarter, features_one_eight, features_one_sixteen


class CostVolumeEncoder(torch.nn.Module):
    def __init__(self):
        super(CostVolumeEncoder, self).__init__()
        self.aggregator0 = conv_layer(input_channels=Config.train_n_depth_levels + fpn_output_channels,
                                      output_channels=hyper_channels,
                                      kernel_size=5,
                                      stride=1,
                                      apply_bn_relu=True)
        self.encoder_block0 = EncoderBlock(input_channels=hyper_channels,
                                           output_channels=hyper_channels * 2,
                                           kernel_size=5)
        ###
        self.aggregator1 = conv_layer(input_channels=hyper_channels * 2 + fpn_output_channels,
                                      output_channels=hyper_channels * 2,
                                      kernel_size=3,
                                      stride=1,
                                      apply_bn_relu=True)
        self.encoder_block1 = EncoderBlock(input_channels=hyper_channels * 2,
                                           output_channels=hyper_channels * 4,
                                           kernel_size=3)
        ###
        self.aggregator2 = conv_layer(input_channels=hyper_channels * 4 + fpn_output_channels,
                                      output_channels=hyper_channels * 4,
                                      kernel_size=3,
                                      stride=1,
                                      apply_bn_relu=True)
        self.encoder_block2 = EncoderBlock(input_channels=hyper_channels * 4,
                                           output_channels=hyper_channels * 8,
                                           kernel_size=3)

        ###
        self.aggregator3 = conv_layer(input_channels=hyper_channels * 8 + fpn_output_channels,
                                      output_channels=hyper_channels * 8,
                                      kernel_size=3,
                                      stride=1,
                                      apply_bn_relu=True)
        self.encoder_block3 = EncoderBlock(input_channels=hyper_channels * 8,
                                           output_channels=hyper_channels * 16,
                                           kernel_size=3)

    def forward(self, features_half, features_quarter, features_one_eight, features_one_sixteen, cost_volume):
        inp0 = torch.cat([features_half, cost_volume], dim=1)
        inp0 = self.aggregator0(inp0)
        out0 = self.encoder_block0(inp0)

        inp1 = torch.cat([features_quarter, out0], dim=1)
        inp1 = self.aggregator1(inp1)
        out1 = self.encoder_block1(inp1)

        inp2 = torch.cat([features_one_eight, out1], dim=1)
        inp2 = self.aggregator2(inp2)
        out2 = self.encoder_block2(inp2)

        inp3 = torch.cat([features_one_sixteen, out2], dim=1)
        inp3 = self.aggregator3(inp3)
        out3 = self.encoder_block3(inp3)

        return inp0, inp1, inp2, inp3, out3


class CostVolumeDecoder(torch.nn.Module):
    def __init__(self):
        super(CostVolumeDecoder, self).__init__()

        self.inverse_depth_base = 1 / Config.train_max_depth
        self.inverse_depth_multiplier = 1 / Config.train_min_depth - 1 / Config.train_max_depth

        self.decoder_block1 = DecoderBlock(input_channels=hyper_channels * 16,
                                           output_channels=hyper_channels * 8,
                                           kernel_size=3,
                                           apply_bn_relu=True,
                                           plus_one=False)

        self.decoder_block2 = DecoderBlock(input_channels=hyper_channels * 8,
                                           output_channels=hyper_channels * 4,
                                           kernel_size=3,
                                           apply_bn_relu=True,
                                           plus_one=True)

        self.decoder_block3 = DecoderBlock(input_channels=hyper_channels * 4,
                                           output_channels=hyper_channels * 2,
                                           kernel_size=3,
                                           apply_bn_relu=True,
                                           plus_one=True)

        self.decoder_block4 = DecoderBlock(input_channels=hyper_channels * 2,
                                           output_channels=hyper_channels,
                                           kernel_size=5,
                                           apply_bn_relu=True,
                                           plus_one=True)

        self.refine = torch.nn.Sequential(conv_layer(input_channels=hyper_channels + 4,
                                                     output_channels=hyper_channels,
                                                     kernel_size=5,
                                                     stride=1,
                                                     apply_bn_relu=True),
                                          conv_layer(input_channels=hyper_channels,
                                                     output_channels=hyper_channels,
                                                     kernel_size=5,
                                                     stride=1,
                                                     apply_bn_relu=True))

        self.depth_layer_one_sixteen = depth_layer_3x3(hyper_channels * 8)
        self.depth_layer_one_eight = depth_layer_3x3(hyper_channels * 4)
        self.depth_layer_quarter = depth_layer_3x3(hyper_channels * 2)
        self.depth_layer_half = depth_layer_3x3(hyper_channels)
        self.depth_layer_full = depth_layer_3x3(hyper_channels)

    def forward(self, image, skip0, skip1, skip2, skip3, bottom):
        # work on cost volume
        decoder_block1 = self.decoder_block1(bottom, skip3, None)
        sigmoid_depth_one_sixteen = self.depth_layer_one_sixteen(decoder_block1)
        inverse_depth_one_sixteen = self.inverse_depth_multiplier * sigmoid_depth_one_sixteen + self.inverse_depth_base

        decoder_block2 = self.decoder_block2(decoder_block1, skip2, sigmoid_depth_one_sixteen)
        sigmoid_depth_one_eight = self.depth_layer_one_eight(decoder_block2)
        inverse_depth_one_eight = self.inverse_depth_multiplier * sigmoid_depth_one_eight + self.inverse_depth_base

        decoder_block3 = self.decoder_block3(decoder_block2, skip1, sigmoid_depth_one_eight)
        sigmoid_depth_quarter = self.depth_layer_quarter(decoder_block3)
        inverse_depth_quarter = self.inverse_depth_multiplier * sigmoid_depth_quarter + self.inverse_depth_base

        decoder_block4 = self.decoder_block4(decoder_block3, skip0, sigmoid_depth_quarter)
        sigmoid_depth_half = self.depth_layer_half(decoder_block4)
        inverse_depth_half = self.inverse_depth_multiplier * sigmoid_depth_half + self.inverse_depth_base

        scaled_depth = torch.nn.functional.interpolate(sigmoid_depth_half, scale_factor=2, mode='bilinear', align_corners=True)
        scaled_decoder = torch.nn.functional.interpolate(decoder_block4, scale_factor=2, mode='bilinear', align_corners=True)
        scaled_combined = torch.cat([scaled_decoder, scaled_depth, image], dim=1)
        scaled_combined = self.refine(scaled_combined)
        inverse_depth_full = self.inverse_depth_multiplier * self.depth_layer_full(scaled_combined) + self.inverse_depth_base

        depth_full = 1.0 / inverse_depth_full.squeeze(1)
        depth_half = 1.0 / inverse_depth_half.squeeze(1)
        depth_quarter = 1.0 / inverse_depth_quarter.squeeze(1)
        depth_one_eight = 1.0 / inverse_depth_one_eight.squeeze(1)
        depth_one_sixteen = 1.0 / inverse_depth_one_sixteen.squeeze(1)

        return depth_full, depth_half, depth_quarter, depth_one_eight, depth_one_sixteen


class LSTMFusion(torch.nn.Module):
    def __init__(self):
        super(LSTMFusion, self).__init__()

        input_size = hyper_channels * 16

        hidden_size = hyper_channels * 16

        self.lstm_cell = MVSLayernormConvLSTMCell(input_dim=input_size,
                                                  hidden_dim=hidden_size,
                                                  kernel_size=(3, 3),
                                                  activation_function=torch.celu)

    def forward(self, current_encoding, current_state, previous_pose, current_pose, estimated_current_depth, camera_matrix):
        batch, channel, height, width = current_encoding.size()

        if current_state is None:
            hidden_state, cell_state = self.lstm_cell.init_hidden(batch_size=batch,
                                                                  image_size=(height, width))
        else:
            hidden_state, cell_state = current_state

        next_hidden_state, next_cell_state = self.lstm_cell(input_tensor=current_encoding,
                                                            cur_state=[hidden_state, cell_state],
                                                            previous_pose=previous_pose,
                                                            current_pose=current_pose,
                                                            estimated_current_depth=estimated_current_depth,
                                                            camera_matrix=camera_matrix)

        return next_hidden_state, next_cell_state
