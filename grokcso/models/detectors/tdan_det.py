# Copyright (c) OpenMMLab. All rights reserved.
import os
print(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d, DeformConv2dPack, deform_conv2d
from mmengine.model import BaseModel
from mmengine.model.weight_init import constant_init
from torch.nn.modules.utils import _pair
from mmengine.registry import MODELS

from mmagic.models.archs import PixelShufflePack, ResidualBlockNoBN
from mmagic.models.utils import make_layer
# from mmagic.registry import MODELS
from tools.utils import read_targets_from_xml_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Interpolate(nn.Module):
    """ 插值层，用于替换PixelShufflePack """
    def __init__(self, scale_factor, mode='bilinear', align_corners=True):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class MSEloss(nn.Module):
    def __init__(self, alpha=0.9):
        super(MSEloss, self).__init__()
        self.MSE = nn.MSELoss()
        self.alpha = alpha
    def forward(self, img1, img2):
        loss = self.alpha*self.MSE(img1, img2)
        return loss

@MODELS.register_module()
class TDAN_Net(BaseModel):
    """TDAN network structure for video super-resolution.

    Support only x4 upsampling.

    Paper:
        TDAN: Temporally-Deformable Alignment Network for Video Super-
        Resolution, CVPR, 2020

    Args:
        in_channels (int): Number of channels of the input image. Default: 3.
        mid_channels (int): Number of channels of the intermediate features.
            Default: 64.
        out_channels (int): Number of channels of the output image. Default: 3.
        num_blocks_before_align (int): Number of residual blocks before
            temporal alignment. Default: 5.
        num_blocks_after_align (int): Number of residual blocks after
            temporal alignment. Default: 10.
    """

    def __init__(self,
                 in_channels=1,
                 mid_channels=32,
                 out_channels=1,
                 num_blocks_before_align=5,
                 num_blocks_after_align=10):

        super(TDAN_Net, self).__init__()
        
        Phi_lrs_Name = 'SeqCSIST/data/phi_0.5.mat'
        Phi_lrs = sio.loadmat(Phi_lrs_Name)
        Phi_input = Phi_lrs['phi']
        self.Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor).to(device)

        Qinit_Name = 'SeqCSIST/data/track_5000_20/train/qinit.mat'
        Qinit_lrs = sio.loadmat(Qinit_Name)
        Qinit = Qinit_lrs['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)
        
        self.feat_extract = nn.Sequential(
            ConvModule(in_channels, mid_channels, 3, padding=1),
            make_layer(
                ResidualBlockNoBN,
                num_blocks_before_align,
                mid_channels=mid_channels))

        self.feat_aggregate = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1, bias=True),
            DeformConv2dPack(
                mid_channels, mid_channels, 3, padding=1, deform_groups=8),
            DeformConv2dPack(
                mid_channels, mid_channels, 3, padding=1, deform_groups=8))
        self.align_1 = AugmentedDeformConv2dPack(
            mid_channels, mid_channels, 3, padding=1, deform_groups=8)
        self.align_2 = DeformConv2dPack(
            mid_channels, mid_channels, 3, padding=1, deform_groups=8)
        self.to_rgb = nn.Conv2d(mid_channels, 1, 3, padding=1, bias=True)

        self.reconstruct = nn.Sequential(
            ConvModule(in_channels * 5, mid_channels, 3, padding=1),
            make_layer(
                ResidualBlockNoBN,
                num_blocks_after_align,
                mid_channels=mid_channels),
            PixelShufflePack(mid_channels, mid_channels, 3, upsample_kernel=3),
            # Interpolate(scale_factor=(33/11)), 
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False))
        self.lq_pixel_loss = MSEloss(alpha=0.01)
        self.fg_loss = MSEloss(alpha=1)
        
    def forward(self, **kwargs):
        mode = kwargs['mode']
        Phi = self.Phi
        Qinit = self.Qinit
        """Forward function for TDANNet.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            tuple[Tensor]: Output HR image with shape (n, c, 4h, 4w) and
            aligned LR images with shape (n, t, c, h, w).
        """
        if mode == 'loss':
            Phix = torch.stack(kwargs['gt_img_11'])
            Phix = Phix.squeeze(dim=1)

            batch = torch.stack(kwargs["batch_x"])
            batch_x = batch.squeeze(dim=1)


        elif mode == 'predict':
            image_name = []
            Phi_x = torch.stack(kwargs["gt_img_11"])
            Phix = Phi_x.squeeze(dim=1)
            ann_paths = kwargs["ann_path"]
            targets_GT = read_targets_from_xml_list(ann_paths)
            image_name = kwargs["image_name"]
        else:
            print("Invalid mode:", mode)
            return None
        
        # x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        x = Phix
        
        t, hw = x.size()
        # print(Phix.shape)
        lr_center = x[t // 2, :] .view(1,1,11,11) # LR center frame
        # print(Phix.shape)
        # extract features
        fea_extract = self.feat_extract(x.view(-1, 1, 11, 11)).view(t, -1, 11, 11)
        # alignment of LR frames

        aligned_lrs = []
        index = []   # 序列索引
        final = []   # 用于存储每个序列的检测结果
        aligned_imgs = {}
       
        for j in range(2 ,t-2):     # 对第 j 个序列进行处理
            aligned_lrs = []
            for i in range(-2,3):
                m = j+i
                # print(m)
                feat_center = fea_extract[j, :, :, :].contiguous()
                feat_center = feat_center.view(1, 32, 11, 11)

                if i == 0:
                    aligned_lrs.append(lr_center)
                else:
                    feat_neig = fea_extract[m, :, :, :].contiguous()
                    feat_neig = feat_neig.view(1, 32, 11, 11)
                    # print(feat_center.shape)
                    # print(feat_neig.shape)
                    feat_agg = torch.cat([feat_center, feat_neig], dim=1)
                    # print(feat_agg.shape)
                    feat_agg = self.feat_aggregate(feat_agg)
                    aligned_feat = self.align_2(self.align_1(feat_neig, feat_agg))
                    aligned_lrs.append(self.to_rgb(aligned_feat))

            aligned_lrs = torch.cat(aligned_lrs, dim=1)
            x_final = self.reconstruct(aligned_lrs).view(1, 1089)
            aligned_lrs = aligned_lrs.view(5, hw)
            # output HR center frame and the aligned LR frames
            # return self.reconstruct(aligned_lrs).view(1, hw), aligned_lrs.view(5, hw)
            index.append(j)
            final.append(x_final)
            aligned_imgs[j] = aligned_lrs
        
        if mode == 'tensor':
            return [final, aligned_imgs]
        elif mode == 'predict':
            final = torch.cat(final, dim=0)
            return [final, image_name, targets_GT, index]
        elif mode == 'loss':
            
            # 将对齐后的支持帧扩展为5维，计算对齐损失
            loss_pix_lq_all = 0
            for i in range(len(index)):
                # print(i)
                m = index[i]
                lq_ref = x[m, :].expand(aligned_imgs[m].size(0), -1)  # 支持帧扩展为5维
                loss_pix_lq = self.lq_pixel_loss(aligned_imgs[m], lq_ref)
                loss_pix_lq_all += loss_pix_lq
            
            # 回归损失
            loss_fg_all = 0
            for i in range(len(index)):
                n = index[i]
                # final_exp = final[i].expand(5, -1)
                # print(final_exp.shape)
                # print(batch_x[n].shape)
                loss_fg = self.fg_loss(final[i], batch_x[n])
                loss_fg_all += loss_fg
            
            # 按权重计算总损失
            loss_all =loss_pix_lq_all + loss_fg_all
            return {'loss': loss_all, 
                    'loss_align': loss_pix_lq_all,
                    'loss_reg': loss_fg_all
                    }

class AugmentedDeformConv2dPack(DeformConv2d):
    """Augmented Deformable Convolution Pack.

    Different from DeformConv2dPack, which generates offsets from the
    preceding feature, this AugmentedDeformConv2dPack takes another feature to
    generate the offsets.

    Args:
        in_channels (int): Number of channels in the input feature.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple[int]): Size of the convolving kernel.
        stride (int or tuple[int]): Stride of the convolution. Default: 1.
        padding (int or tuple[int]): Zero-padding added to both sides of the
            input. Default: 0.
        dilation (int or tuple[int]): Spacing between kernel elements.
            Default: 1.
        groups (int): Number of blocked connections from input channels to
            output channels. Default: 1.
        deform_groups (int): Number of deformable group partitions.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)

        self.init_offset()

    def init_offset(self):
        """Init constant offset."""
        constant_init(self.conv_offset, val=0, bias=0)

    def forward(self, x, extra_feat):
        """Forward function."""
        offset = self.conv_offset(extra_feat)
        return deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups)