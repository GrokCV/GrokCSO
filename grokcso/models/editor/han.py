import torch
import torch.nn as nn
import math
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from tools.utils import read_targets_from_xml_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def default_conv(in_channels, out_channels, kernel_size, bias=True):
  return nn.Conv2d(in_channels, out_channels, kernel_size,
                   padding=(kernel_size // 2), bias=bias)


@MODELS.register_module()
class HAN(BaseModel):
  """
  title: Single Image Super-Resolution via a Holistic Attention Network

  paper: https://link.springer.com/chapter/10.1007/978-3-030-58610-2_12

  code: https://github.com/wwlCape/HAN
  """

  def __init__(self,
               scale=3,
               n_resgroups=10,
               n_resblocks=20,
               n_feats=128,
               kernel_size=3,
               reduction=16,
               n_colors=1
               ):
    super(HAN, self).__init__()
    conv = default_conv
    self.scale = scale
    act = nn.ReLU(True)
    res_scale = 1

    # define head module
    modules_head = [conv(n_colors, n_feats, kernel_size)]

    # define body module
    modules_body = [
      ResidualGroup(conv, n_feats, kernel_size, reduction, act=act,
                    res_scale=res_scale, n_resblocks=n_resblocks)
      for _ in range(n_resgroups)
    ]

    modules_body.append(conv(n_feats, n_feats, kernel_size))

    # define tail module
    modules_tail = [Upsampler(conv, scale, n_feats, act=False),
                    conv(n_feats, n_colors, kernel_size)]

    self.head = nn.Sequential(*modules_head)
    self.body = nn.Sequential(*modules_body)
    self.csa = CSAM_Module(n_feats)
    self.la = LAM_Module(n_feats)
    self.last_conv = nn.Conv2d(n_feats * 11, n_feats, 3, 1, 1)
    self.last = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)
    self.tail = nn.Sequential(*modules_tail)

  def forward(self, **kwargs):

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

    x = self.head(x)
    res = x
    # pdb.set_trace()
    for name, midlayer in self.body._modules.items():
      res = midlayer(res)
      # print(name)
      if name == "0":
        res1 = res.unsqueeze(1)
      else:
        res1 = torch.cat([res.unsqueeze(1), res1], 1)
    # res = self.body(x)
    out1 = res
    # res3 = res.unsqueeze(1)
    # res = torch.cat([res1,res3],1)
    res = self.la(res1)
    out2 = self.last_conv(res)

    out1 = self.csa(out1)
    out = torch.cat([out1, out2], 1)
    res = self.last(out)

    res += x
    # res = self.csa(res)

    out = self.tail(res)

    x_final = out.view(-1, 11*11*self.scale*self.scale)
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
      loss_discrepancy = torch.mean(
        torch.pow(x_final - batch_x, 2))

      return {'loss_discrepancy': loss_discrepancy}

  def load_state_dict(self, state_dict, strict=False):
    own_state = self.state_dict()
    for name, param in state_dict.items():
      if name in own_state:
        if isinstance(param, nn.Parameter):
          param = param.data
        try:
          own_state[name].copy_(param)
        except Exception:
          if name.find("tail") >= 0:
            print("Replace pre-trained upsampler to new one...")
          else:
            raise RuntimeError(
              "While copying the parameter named {}, "
              "whose dimensions in the model are {} and "
              "whose dimensions in the checkpoint are {}.".format(
                name, own_state[name].size(), param.size()
              )
            )
      elif strict:
        if name.find("tail") == -1:
          raise KeyError('unexpected key "{}" in state_dict'.format(name))

    if strict:
      missing = set(own_state.keys()) - set(state_dict.keys())
      if len(missing) > 0:
        raise KeyError('missing keys in state_dict: "{}"'.format(missing))


# Channel Attention (CA) Layer
class CALayer(nn.Module):
  def __init__(self, channel, reduction=16):
    super(CALayer, self).__init__()
    # global average pooling: feature --> point
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    # feature channel downscale and upscale --> channel weight
    self.conv_du = nn.Sequential(
      nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
      nn.Sigmoid(),
    )

  def forward(self, x):
    y = self.avg_pool(x)
    y = self.conv_du(y)
    return x * y


class LAM_Module(nn.Module):
  """Layer attention module"""

  def __init__(self, in_dim):
    super(LAM_Module, self).__init__()
    self.chanel_in = in_dim

    self.gamma = nn.Parameter(torch.zeros(1))
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    """
    inputs :
        x : input feature maps( B X N X C X H X W)
    returns :
        out : attention value + input feature
        attention: B X N X N
    """
    m_batchsize, N, C, height, width = x.size()
    proj_query = x.view(m_batchsize, N, -1)
    proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
    energy = torch.bmm(proj_query, proj_key)
    energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(
      energy) - energy
    attention = self.softmax(energy_new)
    proj_value = x.view(m_batchsize, N, -1)

    out = torch.bmm(attention, proj_value)
    out = out.view(m_batchsize, N, C, height, width)

    out = self.gamma * out + x
    out = out.view(m_batchsize, -1, height, width)
    return out


class CSAM_Module(nn.Module):
  """Channel-Spatial attention module"""

  def __init__(self, in_dim):
    super(CSAM_Module, self).__init__()
    self.chanel_in = in_dim

    self.conv = nn.Conv3d(1, 1, 3, 1, 1)
    self.gamma = nn.Parameter(torch.zeros(1))
    # self.softmax  = nn.Softmax(dim=-1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    """
    inputs :
        x : input feature maps( B X N X C X H X W)
    returns :
        out : attention value + input feature
        attention: B X N X N
    """
    m_batchsize, C, height, width = x.size()
    out = x.unsqueeze(1)
    out = self.sigmoid(self.conv(out))

    # proj_query = x.view(m_batchsize, N, -1)
    # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
    # energy = torch.bmm(proj_query, proj_key)
    # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
    # attention = self.softmax(energy_new)
    # proj_value = x.view(m_batchsize, N, -1)

    # out = torch.bmm(attention, proj_value)
    # out = out.view(m_batchsize, N, C, height, width)

    out = self.gamma * out
    out = out.view(m_batchsize, -1, height, width)
    x = x * out + x
    return x


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
  def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False,
               act=nn.ReLU(True), res_scale=1):

    super(RCAB, self).__init__()
    modules_body = []
    for i in range(2):
      modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
      if bn:
        modules_body.append(nn.BatchNorm2d(n_feat))
      if i == 0:
        modules_body.append(act)
    modules_body.append(CALayer(n_feat, reduction))
    self.body = nn.Sequential(*modules_body)
    self.res_scale = res_scale

  def forward(self, x):
    res = self.body(x)
    # res = self.body(x).mul(self.res_scale)
    res += x
    return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
  def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale,
               n_resblocks):
    super(ResidualGroup, self).__init__()
    modules_body = []
    modules_body = [
      RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False,
           act=nn.ReLU(True), res_scale=1)
      for _ in range(n_resblocks)
    ]
    modules_body.append(conv(n_feat, n_feat, kernel_size))
    self.body = nn.Sequential(*modules_body)

  def forward(self, x):
    res = self.body(x)
    res += x
    return res


class Upsampler(nn.Sequential):
  def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

    m = []
    if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
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