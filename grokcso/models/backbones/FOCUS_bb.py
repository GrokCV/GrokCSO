import torch
import torch.nn as nn
import torch.nn.functional as F

class MSEloss(nn.Module):
    def __init__(self, alpha=0.9):
        super(MSEloss, self).__init__()
        self.MSE = nn.MSELoss()
        self.alpha = alpha
    def forward(self, img1, img2):
        loss = self.alpha*self.MSE(img1, img2)
        return loss

class FOCUS_bb(nn.Module):
    def __init__(self, upscale_factor=3):
        super(FOCUS_bb, self).__init__()
        
        self.stage1 = ISGU(upscale_factor=upscale_factor)
        
        # refinement
        # 去除 Stage 1 产生的伪影，平滑背景，锐化目标
        self.stage2 = ISGU(upscale_factor=1)
        
        self.relu = nn.PReLU()

    def forward(self, x):
        out1 = self.stage1(x)
        
        out1_act = self.relu(out1)
        
        residual = self.stage2(out1_act)
        
        # 最终结果 = 粗预测 + 修正量
        out_final = out1_act + residual
        
        return out_final, out1, out1_act

class ISGU(nn.Module):
    def __init__(self, upscale_factor):
        super(ISGU, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv_intensity = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))  # 强度预测分支
        self.conv_mask = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1)) # 掩码预测分支
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.sigmoid = nn.Sigmoid() # 将掩码限制在 [0, 1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        intensity_map = self.conv_intensity(x) # (B, 9, H, W)
        mask_map = self.conv_mask(x)           # (B, 9, H, W)
        
        intensity_hr = self.pixel_shuffle(intensity_map) # (B, 1, 3H, 3W)
        mask_hr = self.pixel_shuffle(mask_map)           # (B, 1, 3H, 3W)
        
        mask_prob = self.sigmoid(mask_hr) # mask 变为概率图
        out = intensity_hr * mask_prob    # 稀疏化输出
        return out  
