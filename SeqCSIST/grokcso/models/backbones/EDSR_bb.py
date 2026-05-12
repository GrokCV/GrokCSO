import torch
from torch import nn

from .basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer

class MSEloss(nn.Module):
    def __init__(self, alpha=0.9):
        super(MSEloss, self).__init__()
        self.MSE = nn.MSELoss()
        self.alpha = alpha
    def forward(self, img1, img2):
        loss = self.alpha*self.MSE(img1, img2)
        return loss

class EDSR(nn.Module):
    """EDSR network structure for grayscale images.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3. Default: 4.
        res_scale (float): Used to scale the residual in residual block. Default: 1.
        img_range (float): Image range. Default: 255.
    """

    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=32,
                 num_block=16,
                 upscale=3,
                 res_scale=1,
                 img_range=255.):
        super(EDSR, self).__init__()

        self.img_range = img_range

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)

        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = Upsample(upscale, num_feat)
        
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = x * self.img_range  # Scale input
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x  # Add residual

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range  # Rescale output

        return x
