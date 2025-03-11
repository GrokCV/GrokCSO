import torch.nn.functional as F
from mmengine.registry import MODELS
import torch.nn as nn
import torch
from mmengine.model import BaseModel
from tools.utils import read_targets_from_xml_list
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


## Second-order Channel Attention Network (SAN)
@MODELS.register_module()
class SAN(BaseModel):
    """
    title: Second-Order Attention Network for Single Image Super-Resolution
    paper: https://ieeexplore.ieee.org/document/8954252
    code: https://github.com/daitao/SAN
    """
    def __init__(self,
                 scale=3,
                 n_colors=1,
                 conv=default_conv,
                 n_resgroups=10,
                 n_resblocks = 5,
                 n_feats=64,
                 kernel_size=3,
                 reduction=16,
                 ):
        super(SAN, self).__init__()
        self.scale = scale

        act = nn.ReLU(inplace=True)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        ##
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 0.2
        self.n_resgroups = n_resgroups
        self.RG = nn.ModuleList(
            [
                LSRAG(conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks)
                for _ in range(n_resgroups)
            ]
        )
        self.conv_last = conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats, n_colors, kernel_size)]

        self.non_local = Nonlocal_CA(
            in_feat=n_feats, inter_feat=n_feats // 8, reduction=8, sub_sample=False, bn_layer=False
        )

        self.head = nn.Sequential(*modules_head)
        # self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)

        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, **kwargs):
        mode = kwargs['mode']
        if mode == 'tensor':
          Phix = kwargs["Phix"]
        else:
          Phix = torch.stack(kwargs["gt_img_11"])
          Input_image = torch.stack(kwargs["gt_img_11"])
          Phix = Phix.squeeze(dim=1).to(device)

          # 获取gt图像
          batch_x = torch.stack(kwargs["gt"]).squeeze(dim=1).to(device)
          # 获取标注路径
          ann_paths = kwargs["ann_path"]
          # 记录文件ID
          file_id = kwargs["file_id"]

        x = Phix.view(-1, 1, 11, 11)

        x = self.head(x)

        ## add nonlocal
        xx = self.non_local(x)

        # share-source skip connection
        residual = xx

        # res = self.RG(xx)
        # res = res + xx
        ## share-source residual gruop
        for i, l in enumerate(self.RG):
            xx = l(xx) + self.gamma * residual
            # xx = self.gamma*xx + residual
        # body part
        # res = self.body(xx)
        ##
        ## add nonlocal
        res = self.non_local(xx)
        ##
        # res = self.soca(res)
        # res += x
        res = res + x

        out = self.tail(res)

        x_final = out.view(-1, 11*self.scale*11*self.scale)

        # 根据不同模式返回相应结果
        if mode == 'tensor':
          return x_final  # 返回最终网络输出 todo: 对网络输出结果进行后处理操作，类似nms，使得输出结果更加准确
        elif mode == 'predict':
          # 读取目标标注信息并返回预测结果
          targets_GT = read_targets_from_xml_list(ann_paths)
          return [{"x_final": x_final,
                   "targets_GT": targets_GT,
                   "file_id": file_id,
                   "Input_image": Input_image,
                   "img_gt": batch_x}]
        elif mode == 'loss':
          # 计算损失函数
          loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))

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



## non_local module
class _NonLocalBlockND(nn.Module):
    def __init__(
        self, in_channels, inter_channels=None, dimension=3, mode="embedded_gaussian", sub_sample=True, bn_layer=True
    ):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ["embedded_gaussian", "gaussian", "dot_product", "concatenation"]

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0
                ),
                bn(self.in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0
            )
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None
        # self.fc = nn.Linear(64,2304,bias=True)
        # self.sub_bilinear = nn.Upsample(size=(48,48),mode='bilinear')
        # self.sub_maxpool = nn.AdaptiveMaxPool2d(output_size=(48,48))
        if mode in ["embedded_gaussian", "dot_product", "concatenation"]:
            self.theta = conv_nd(
                in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
            )
            self.phi = conv_nd(
                in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
            )

            if mode == "embedded_gaussian":
                self.operation_function = self._embedded_gaussian
            elif mode == "dot_product":
                self.operation_function = self._dot_product
            elif mode == "concatenation":
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False), nn.ReLU()
                )
        elif mode == "gaussian":
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size, C, H, W = x.shape

        # x_sub = self.sub_bilinear(x) # bilinear downsample
        # x_sub = self.sub_maxpool(x) # maxpool downsample

        ##
        # g_x = x.view(batch_size, self.inter_channels, -1)
        # g_x = g_x.permute(0, 2, 1)
        #
        # # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        # theta_x = x.view(batch_size, self.inter_channels, -1)
        # theta_x = theta_x.permute(0, 2, 1)
        # fc = self.fc(theta_x)
        # # phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # # f = torch.matmul(theta_x, phi_x)
        # # return f
        # # f_div_C = F.softmax(fc, dim=-1)
        # return fc

        ##
        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        # return f
        f_div_C = F.softmax(f, dim=-1)
        # return f_div_C
        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode="embedded_gaussian", sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(
            in_channels, inter_channels=inter_channels, dimension=1, mode=mode, sub_sample=sub_sample, bn_layer=bn_layer
        )


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode="embedded_gaussian", sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(
            in_channels, inter_channels=inter_channels, dimension=2, mode=mode, sub_sample=sub_sample, bn_layer=bn_layer
        )


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            # nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        _, _, h, w = x.shape
        y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_ave = self.conv_du(y_ave)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_ave


## second-order Channel attention (SOCA)
class SOCA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SOCA, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W : (W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H : H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H : (H + h1), W : (W + w1)]
        # subsample
        # subsample_scale = 2
        # subsample = nn.Upsample(size=(h // subsample_scale, w // subsample_scale), mode='nearest')
        # x_sub = subsample(x)
        # max_pool = nn.MaxPool2d(kernel_size=2)
        # max_pool = nn.AvgPool2d(kernel_size=2)
        # x_sub = self.max_pool(x)
        ##
        ## MPN-COV
        cov_mat = CovpoolLayer(x_sub)  # Global Covariance pooling layer
        cov_mat_sqrt = SqrtmLayer(
            cov_mat, 5
        )  # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt, 1)
        cov_mat_sum = cov_mat_sum.view(batch_size, C, 1, 1)
        # y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_cov = self.conv_du(cov_mat_sum)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_cov * x


## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, reduction=8, sub_sample=False, bn_layer=True):
        super(Nonlocal_CA, self).__init__()
        # second-order channel attention
        self.soca = SOCA(in_feat, reduction=reduction)
        # nonlocal module
        self.non_local = NONLocalBlock2D(
            in_channels=in_feat, inter_channels=inter_feat, sub_sample=sub_sample, bn_layer=bn_layer
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## divide feature map into 4 part
        batch_size, C, H, W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]

        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return nonlocal_feat


## Residual  Block (RB)
class RB(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        reduction,
        bias=True,
        bn=False,
        act=nn.ReLU(inplace=True),
        res_scale=1,
        dilation=2,
    ):
        super(RB, self).__init__()
        modules_body = []

        # self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma1 = 1.0
        # self.salayer = SALayer(n_feat, reduction=reduction, dilation=dilation)
        # self.salayer = SALayer2(n_feat, reduction=reduction, dilation=dilation)

        self.conv_first = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)
        )

        self.res_scale = res_scale

    def forward(self, x):
        y = self.conv_first(x)
        y = y + x

        return y


## Local-source Residual Attention Group (LSRARG)
class LSRAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(LSRAG, self).__init__()
        ##
        self.rcab = nn.ModuleList(
            [
                RB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1)
                for _ in range(n_resblocks)
            ]
        )
        self.soca = SOCA(n_feat, reduction=reduction)
        self.conv_last = conv(n_feat, n_feat, kernel_size)
        self.n_resblocks = n_resblocks
        ##
        # modules_body = []
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 0.2
        # for i in range(n_resblocks):
        #     modules_body.append(RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1))
        # modules_body.append(SOCA(n_feat,reduction=reduction))
        # # modules_body.append(Nonlocal_CA(in_feat=n_feat, inter_feat=n_feat//8, reduction =reduction, sub_sample=False, bn_layer=False))
        # modules_body.append(conv(n_feat, n_feat, kernel_size))
        # self.body = nn.Sequential(*modules_body)
        ##

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        # batch_size,C,H,W = x.shape
        # y_pre = self.body(x)
        # y_pre = y_pre + x
        # return y_pre

        ## share-source skip connection

        for i, l in enumerate(self.rcab):
            # x = l(x) + self.gamma*residual
            x = l(x)
        x = self.soca(x)
        x = self.conv_last(x)

        x = x + residual

        return x


class Upsampler(nn.Sequential):
  def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

    m = []
    if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
      for _ in range(int(math.log(scale, 2))):
        m.append(conv(n_feat, 4 * n_feat, 3, bias))
        # m.append(conv(n_feat, 2 * n_feat, 3, bias))
        # m.append(conv(2 * n_feat, 4 * n_feat, 3, bias))
        m.append(nn.PixelShuffle(2))
        if bn: m.append(nn.BatchNorm2d(n_feat))
        if act: m.append(act())
    elif scale == 3:
      m.append(conv(n_feat, 9 * n_feat, 3, bias))
      m.append(nn.PixelShuffle(3))
      if bn: m.append(nn.BatchNorm2d(n_feat))
      if act: m.append(act())
    else:
      raise NotImplementedError

    super(Upsampler, self).__init__(*m)

"""
@file: MPNCOV.py
@author: Jiangtao Xie
@author: Peihua Li

Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
"""
import numpy as np
from torch.autograd import Function


class Covpool(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        I_hat = (-1.0 / M / M) * torch.ones(M, M, device=x.device) + (1.0 / M) * torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2))
        ctx.save_for_backward(input, I_hat)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, I_hat = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        grad_input = grad_output + grad_output.transpose(1, 2)
        grad_input = grad_input.bmm(x).bmm(I_hat)
        grad_input = grad_input.reshape(batchSize, dim, h, w)
        return grad_input


class Sqrtm(Function):
    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
        Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad=False, device=x.device)
        Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, iterN, 1, 1)
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
            Z[:, 0, :, :] = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - Z[:, i - 1, :, :].bmm(Y[:, i - 1, :, :]))
                Y[:, i, :, :] = Y[:, i - 1, :, :].bmm(ZY)
                Z[:, i, :, :] = ZY.bmm(Z[:, i - 1, :, :])
            ZY = 0.5 * Y[:, iterN - 2, :, :].bmm(I3 - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]))
        y = ZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, ZY, normA, Y, Z)
        ctx.iterN = iterN
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postComAux = (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 * torch.sqrt(normA))
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        if iterN < 2:
            der_NSiter = 0.5 * (der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace))
        else:
            dldY = 0.5 * (
                der_postCom.bmm(I3 - Y[:, iterN - 2, :, :].bmm(Z[:, iterN - 2, :, :]))
                - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]).bmm(der_postCom)
            )
            dldZ = -0.5 * Y[:, iterN - 2, :, :].bmm(der_postCom).bmm(Y[:, iterN - 2, :, :])
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dldY_ = 0.5 * (dldY.bmm(YZ) - Z[:, i, :, :].bmm(dldZ).bmm(Z[:, i, :, :]) - ZY.bmm(dldY))
                dldZ_ = 0.5 * (YZ.bmm(dldZ) - Y[:, i, :, :].bmm(dldY).bmm(Y[:, i, :, :]) - dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        grad_input = der_NSiter.div(normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[i, :, :] += (der_postComAux[i] - grad_aux[i] / (normA[i] * normA[i])) * torch.ones(
                dim, device=x.device
            ).diag()
        return grad_input, None


class Triuvec(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        x = x.reshape(batchSize, dim * dim)
        I = torch.ones(dim, dim).triu().t().reshape(dim * dim)
        index = I.nonzero()
        y = torch.zeros(batchSize, dim * (dim + 1) / 2, device=x.device)
        for i in range(batchSize):
            y[i, :] = x[i, index].t()
        ctx.save_for_backward(input, index)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, index = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        grad_input = torch.zeros(batchSize, dim, dim, device=x.device, requires_grad=False)
        grad_input = grad_input.reshape(batchSize, dim * dim)
        for i in range(batchSize):
            grad_input[i, index] = grad_output[i, :].reshape(index.size(), 1)
        grad_input = grad_input.reshape(batchSize, dim, dim)
        return grad_input


def CovpoolLayer(var):
    return Covpool.apply(var)


def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)


def TriuvecLayer(var):
    return Triuvec.apply(var)

if __name__ == '__main__':
  from mmengine.analysis import get_model_complexity_info
  from mmengine.analysis.print_helper import _format_size

  model = SAN(3)
  model = model.to(device)

  outputs = get_model_complexity_info(model, input_shape=(64, 121),
                                           show_table=True, show_arch=True)
  flops, params = _format_size(outputs['flops']), _format_size(outputs['params'])

  print("flops:{}".format(flops))
  print("params:{}".format(params))