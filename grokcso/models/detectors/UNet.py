import torch.nn.functional as F
import torch.nn as nn
from mmengine.model import BaseModule
import torch
from mmengine.registry import MODELS
from torch.nn.utils import spectral_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@MODELS.register_module()
class Discriminator_UNet(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, in_channels=3, mid_channels=64):
        super(Discriminator_UNet, self).__init__()
        norm = spectral_norm

        self.conv0 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(mid_channels, mid_channels * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(mid_channels * 2, mid_channels * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(mid_channels * 4, mid_channels * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(mid_channels * 8, mid_channels * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(mid_channels * 4, mid_channels * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False))

        self.conv9 = nn.Conv2d(mid_channels, 1, 3, 1, 1)
        print('using the UNet discriminator')

    def forward(self, x):
        x = x.view(-1,1,33,33).to(device)
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        x6 = F.interpolate(x6, size=x0.shape[2:], mode='bilinear', align_corners=False)
        x6 = x6 + x0

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out