# Deep Back-Projection Networks for Super-Resolution
# https://arxiv.org/abs/1803.02735

import torch.nn as nn

import grokcso.models.backbones.blocks as B

class MSEloss(nn.Module):
    def __init__(self, alpha=0.9):
        super(MSEloss, self).__init__()
        self.MSE = nn.MSELoss()
        self.alpha = alpha
    def forward(self, img1, img2):
        loss = self.alpha*self.MSE(img1, img2)
        return loss


class DBPN_bb(nn.Module):
    def __init__(self,in_channels=1, out_channels=1, num_features=32, bp_stages=7, upscale_factor=3, norm_type=None, act_type='prelu'):
        super(DBPN_bb, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            projection_filter = 6
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            projection_filter = 8
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            projection_filter = 7

        feature_extract_1 = B.ConvBlock(in_channels, 128, kernel_size=3, norm_type=norm_type, act_type=act_type)
        feature_extract_2 = B.ConvBlock(128, num_features, kernel_size=1, norm_type=norm_type, act_type=act_type)

        bp_units = []
        for _ in range(bp_stages-1):
            bp_units.extend([B.UpprojBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                                                padding=padding, norm_type=norm_type, act_type=act_type),
                            B.DownprojBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                                                  padding=padding, norm_type=norm_type, act_type=act_type)])

        last_bp_unit = B.UpprojBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                                           padding=padding, norm_type=norm_type, act_type=act_type)
        conv_hr = B.ConvBlock(num_features, out_channels, kernel_size=1, norm_type=None, act_type=None)

        self.network = B.sequential(feature_extract_1, feature_extract_2, *bp_units, last_bp_unit, conv_hr)

    def forward(self, x):
        return self.network(x)
