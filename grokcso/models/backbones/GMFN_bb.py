import torch
import torch.nn as nn
from grokcso.models.backbones.blocks import ConvBlock, DeconvBlock, MeanShift, ResidualDenseBlock_8C

class MSEloss(nn.Module):
    def __init__(self, alpha=0.9):
        super(MSEloss, self).__init__()
        self.MSE = nn.MSELoss()
        self.alpha = alpha
    def forward(self, img1, img2):
        loss = self.alpha*self.MSE(img1, img2)
        return loss

class GFMRDB(nn.Module):
    def __init__(self, num_features, num_blocks, num_refine_feats, num_reroute_feats, act_type, norm_type=None):
        super(GFMRDB, self).__init__()

        self.num_refine_feats = num_refine_feats
        self.num_reroute_feats = num_reroute_feats

        self.RDBs_list = nn.ModuleList([ResidualDenseBlock_8C(
            num_features, kernel_size=3, gc=num_features, act_type=act_type
            ) for _ in range(num_blocks)])

        self.GFMs_list = nn.ModuleList([
                ConvBlock(
                    in_channels=num_reroute_feats*num_features, out_channels=num_features, kernel_size=1,
                    norm_type=norm_type, act_type=act_type
                ),
                ConvBlock(
                    in_channels=2*num_features, out_channels=num_features, kernel_size=1,
                    norm_type=norm_type, act_type=act_type)
            ])


    def forward(self, input_feat, last_feats_list):

        cur_feats_list = []

        if len(last_feats_list) == 0:
            for b in self.RDBs_list:
                input_feat = b(input_feat)
                cur_feats_list.append(input_feat)
        else:
            for idx, b in enumerate(self.RDBs_list):

                # refining the lowest-level features
                if idx < self.num_refine_feats:
                    select_feat = self.GFMs_list[0](torch.cat(last_feats_list, 1))
                    input_feat = self.GFMs_list[1](torch.cat((select_feat, input_feat), 1))

                input_feat = b(input_feat)
                cur_feats_list.append(input_feat)

        # rerouting the highest-level features
        return cur_feats_list[-self.num_reroute_feats:]

class GMFN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=32, num_steps=4, num_blocks=8,
                 num_reroute_feats=8, num_refine_feats=16, upscale_factor=3, act_type='prelu', norm_type=None):
        super(GMFN, self).__init__()

        # 确定上采样因子和卷积参数
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        else:
            raise ValueError("upscale_factor must be 2, 3, or 4.")

        self.num_features = num_features
        self.num_steps = num_steps
        self.upscale_factor = upscale_factor

        # 初始特征提取模块
        self.conv_in = ConvBlock(in_channels, 4 * num_features, kernel_size=3, act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * num_features, num_features, kernel_size=1, act_type=act_type, norm_type=norm_type)

        # 多个残差密集块和门控反馈模块
        self.block = GFMRDB(num_features, num_blocks, num_refine_feats, num_reroute_feats,
                            act_type=act_type, norm_type=norm_type)

        # 重建模块
        self.upsample = nn.functional.interpolate
        self.out = DeconvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                               act_type='prelu', norm_type=norm_type)
        self.conv_out = ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type)

    def forward(self, lr_img):
        # 上采样低分辨率红外图像
        up_lr_img = self.upsample(lr_img, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        init_feat = self.feat_in(self.conv_in(lr_img))

        sr_imgs = []
        last_feats_list = []

        for _ in range(self.num_steps):
            last_feats_list = self.block(init_feat, last_feats_list)
            out = torch.add(up_lr_img, self.conv_out(self.out(last_feats_list[-1])))
            sr_imgs.append(out)

        return sr_imgs
