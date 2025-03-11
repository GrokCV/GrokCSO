import torch
import torch.nn as nn
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from typing import Dict, Union
from tools.utils import read_targets_from_xml_list
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class MeanShiftGray(nn.Conv2d):
  def __init__(self, gray_range, gray_mean=0.5, gray_std=1.0, sign=-1):
    super(MeanShiftGray, self).__init__(1, 1,
                                        kernel_size=1)  # 1 channel for grayscale
    std = torch.Tensor([gray_std])
    self.weight.data = torch.ones(1, 1, 1, 1) / std.view(1, 1, 1,
                                                         1)  # Single channel operation
    self.bias.data = sign * gray_range * torch.Tensor([gray_mean]) / std
    for p in self.parameters():
      p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


@MODELS.register_module()
class EDSR(BaseModel):
  def __init__(self,
               scale=3,
               n_resblocks=16,  # step 1
               n_feats=64,
               kernel_size=3
               ):
    super(EDSR, self).__init__()
    self.scale = scale          # step 2

    conv = default_conv
    act = nn.ReLU(True)
    url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
    if url_name in url:
      self.url = url[url_name]
    else:
      self.url = None
    self.sub_mean = MeanShiftGray(255)
    self.add_mean = MeanShiftGray(255, sign=1)

    # define head module
    m_head = [conv(1, n_feats, kernel_size)]

    # define body module
    m_body = [
      ResBlock(
        conv, n_feats, kernel_size, act=act, res_scale=1
      ) for _ in range(n_resblocks)
    ]
    m_body.append(conv(n_feats, n_feats, kernel_size))

    # define tail module
    m_tail = [
      Upsampler(conv, scale, n_feats, act=False),
      conv(n_feats, 1, kernel_size)
    ]

    self.head = nn.Sequential(*m_head)
    self.body = nn.Sequential(*m_body)
    self.tail = nn.Sequential(*m_tail)

    self.loss = nn.L1Loss()

  def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:
    # step 3
    mode = kwargs['mode']
    if mode == 'tensor':
      Phix = kwargs["Phix"]
    else:
      Phix = torch.stack(kwargs["gt_img_11"])
      Input_image = torch.stack(kwargs["gt_img_11"])
      Phix = Phix.squeeze(dim=1).to(device)

      batch_x = torch.stack(kwargs["gt"]).squeeze(dim=1).to(device)
      ann_paths = kwargs["ann_path"]
      file_id = kwargs["file_id"]

    x = Phix.view(-1, 1, 11, 11)

    x = self.sub_mean(x)
    x = self.head(x)

    res = self.body(x)
    res += x

    x = self.tail(res)
    x = self.add_mean(x)

    x_final = x.view(-1, 11*11*self.scale*self.scale)   # step 4

    if mode == 'tensor':
      return x_final
    elif mode == 'predict':
      targets_GT = read_targets_from_xml_list(ann_paths)
      return [{"x_final": x_final,
               "targets_GT": targets_GT,
               "file_id": file_id,
               "Input_image": Input_image,
               "img_gt": batch_x}]
    elif mode == 'loss':
      l1_loss = self.loss(x_final, batch_x)
      loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))

      return {'loss_discrepancy': loss_discrepancy}