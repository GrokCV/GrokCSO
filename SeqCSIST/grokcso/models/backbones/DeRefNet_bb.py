import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import init
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2dPack
from mmcv.ops import DeformConv2d, deform_conv2d
from mmengine.model.weight_init import constant_init
from mmagic.models.archs import ResidualBlockNoBN
from mmagic.models.utils import make_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MSEloss(nn.Module):
    def __init__(self, alpha=0.9):
        super(MSEloss, self).__init__()
        self.MSE = nn.MSELoss()
        self.alpha = alpha
    def forward(self, img1, img2):
        loss = self.alpha*self.MSE(img1, img2)
        return loss

class AugmentedDeformConv2dPack(DeformConv2d):

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

class BasicBlock(torch.nn.Module):
  def __init__(self):
    super(BasicBlock, self).__init__()

    self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
    self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

    self.conv1_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
    self.conv2_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
    self.conv1_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
    self.conv2_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

  def forward(self, x, PhiTPhi, PhiTb):
    x = x - self.lambda_step * torch.mm(x, PhiTPhi)
    x = x + self.lambda_step * PhiTb
    x_input = x.view(-1, 1, 33, 33)

    x = F.conv2d(x_input, self.conv1_forward, padding=1)
    x = F.relu(x)
    x_forward = F.conv2d(x, self.conv2_forward, padding=1)

    x = torch.mul(torch.sign(x_forward),
                  F.relu(torch.abs(x_forward) - self.soft_thr))

    x = F.conv2d(x, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_backward = F.conv2d(x, self.conv2_backward, padding=1)

    x_pred = x_backward.view(-1, 1089)

    x = F.conv2d(x_forward, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_est = F.conv2d(x, self.conv2_backward, padding=1)

    symloss = x_est - x_input

    return [x_pred, symloss]

## TDFA module
class TDDA(nn.Module):
    def __init__(self,
                in_channels = 1,
                mid_channels = 32,
                out_channels = 1,
                position_dim = 1089
                ):
        super(TDDA, self).__init__()
        
        self.position_dim = position_dim
        self.sigmoid = nn.Sigmoid()
        # self.head_conv = nn.Conv2d(in_channels * 2, out_channels, 3, 1, 1, bias=False)
        self.mlp = nn.Linear(position_dim, position_dim)
        self.feat_extract = nn.Sequential(
            ConvModule(in_channels, mid_channels, 3, padding=1)
            )

        self.feat_aggregate = SK(mid_channels)
        
        self.align_1 = AugmentedDeformConv2dPack(
            mid_channels, mid_channels, 3, padding=1, deform_groups=1)
        self.align_2 = DeformConv2dPack(
            mid_channels, mid_channels, 3, padding=1, deform_groups=1)
        self.conv = nn.Conv2d(mid_channels, 1, 3, padding=1, bias=True)
        # self.tail_conv = nn.Conv2d(in_channels * 5, out_channels, 3, 1, 1, bias=False)
        self.tail_conv = nn.Sequential(
            ConvModule(in_channels * 5, mid_channels, 3, padding=1),    # 改帧
            make_layer(
                ResidualBlockNoBN,
                5,
                mid_channels=mid_channels),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False))
    
    def generate_positional_encoding(self, seq_len, feature_dim):
        # Define the time step vector
        time_steps = torch.arange(seq_len).view(-1, 1)  # 形状为 (20, 1)
        pe = time_steps.repeat(1, feature_dim) 
        pe = pe / 20
    
        return pe
    
    
    def forward(self, x):
        t, hw = x.shape
        ## add position encoding
        position_encoding = self.generate_positional_encoding(t, self.position_dim).to(x.device)
        position_encoding  = self.mlp(position_encoding) 
        position_encoding = self.sigmoid(position_encoding)
        x = x * position_encoding
        x = x.view(-1, 1, 33, 33)
    
        # alignment of frames
        index = []   
        final = []   
        aligned_imgs = {}
        for j in range(2 ,t-2):     
            aligned_fea = []
            for i in range(-2,3):   
                m = j+i
                feat_center = x[j, :, :, :].contiguous()
                feat_center = feat_center.view(1, -1, 33, 33)
                if i == 0:
                    aligned_fea.append(feat_center)
                else:
                    feat_center = self.feat_extract(feat_center)
                    feat_neig = x[m, :, :, :].contiguous()
                    feat_neig = feat_neig.view(1, -1, 33, 33)
                    feat_neig = self.feat_extract(feat_neig)
                    feat_agg = self.feat_aggregate(feat_center, feat_neig)
                    aligned_feat = self.align_2(self.align_1(feat_neig, feat_agg))
                    aligned_fea.append(self.conv(aligned_feat))
                    
            aligned_fea = torch.cat(aligned_fea, dim=1)

            # output center frame and the aligned frames
            x_final = self.tail_conv(aligned_fea).view(1, hw)
            aligned_fea = aligned_fea.view(5, hw)       
            index.append(j)
            final.append(x_final)
            aligned_imgs[j] = aligned_fea
        return final, index, aligned_imgs


## selective attention module
class SK(nn.Module):

  def __init__(self, dim):
    """
    注意：
        dim 必须是一个正整数且大于等于 2。
    """
    super().__init__()
    
    self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
    
    self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
    
    self.conv_squeeze = nn.Conv2d(2, 2, 3, padding=1)

  def forward(self, x, y):

    attn1 = x
    attn2 = y

    
    attn1 = self.conv0_s(attn1)
    attn2 = self.conv1_s(attn2)
    
    
    attn = torch.cat([attn1, attn2], dim=1)
    avg_attn = torch.mean(attn, dim=1, keepdim=True)
    max_attn, _ = torch.max(attn, dim=1, keepdim=True)
    agg = torch.cat([avg_attn, max_attn], dim=1)
    sig = self.conv_squeeze(agg).sigmoid()

    
    att1 = attn1 * sig[:, 0, :, :].unsqueeze(1)
    att2 = attn2 * sig[:, 1, :, :].unsqueeze(1)
    
    att = torch.cat([att1, att2], dim=1)
    
    return att
        