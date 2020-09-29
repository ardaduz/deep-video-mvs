import torch
import torch.nn as nn

from dvmvs.utils import warp_frame_depth


class MVSLayernormConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, activation_function=None):
        super(MVSLayernormConvLSTMCell, self).__init__()

        self.activation_function = activation_function

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=False)

    def forward(self, input_tensor, cur_state, previous_pose, current_pose, estimated_current_depth, camera_matrix):
        h_cur, c_cur = cur_state

        if previous_pose is not None:
            transformation = torch.bmm(torch.inverse(previous_pose), current_pose)

            non_valid = estimated_current_depth <= 0.01
            h_cur = warp_frame_depth(image_src=h_cur,
                                     depth_dst=estimated_current_depth,
                                     src_trans_dst=transformation,
                                     camera_matrix=camera_matrix,
                                     normalize_points=False,
                                     sampling_mode='bilinear')
            b, c, h, w = h_cur.size()
            non_valid = torch.cat([non_valid] * c, dim=1)
            h_cur.data[non_valid] = 0.0

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        b, c, h, w = h_cur.size()
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)

        cc_g = torch.layer_norm(cc_g, [h, w])
        g = self.activation_function(cc_g)

        c_next = f * c_cur + i * g
        c_next = torch.layer_norm(c_next, [h, w])
        h_next = o * self.activation_function(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
