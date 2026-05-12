import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from tools.utils import read_targets_from_xml_list
import numpy as np
import scipy.io as sio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MSEloss(nn.Module):
    def __init__(self, alpha=0.9):
        super(MSEloss, self).__init__()
        self.MSE = nn.MSELoss()
        self.alpha = alpha
    def forward(self, img1, img2):
        loss = self.alpha*self.MSE(img1, img2)
        return loss
    
@MODELS.register_module()
class RPCANet(BaseModel):
    def __init__(self, stage_num=6, slayers=6, llayers=3, mlayers=3, channel=32):
        super(RPCANet, self).__init__()
        Phi_lrs_Name = 'SeqCSIST/data/phi_0.5.mat'
        Phi_lrs = sio.loadmat(Phi_lrs_Name)
        Phi_input = Phi_lrs['phi']
        self.Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor).to(device)

        Qinit_Name = 'SeqCSIST/data/track_5000_20/train/qinit.mat'
        Qinit_lrs = sio.loadmat(Qinit_Name)
        Qinit = Qinit_lrs['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)
        self.stage_num = stage_num
        self.decos = nn.ModuleList()
        self.mse = MSEloss()
        # self.softiou = SoftLoULoss()
        for _ in range(stage_num):
            self.decos.append(DecompositionModule(slayers=slayers, llayers=llayers,
                                                  mlayers=mlayers, channel=channel))
        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, **kwargs):
        mode = kwargs['mode']
        Qinit = self.Qinit
        if mode == 'loss':
            batch = torch.stack(kwargs["batch_x"])
            batch_x = batch.squeeze(dim=1)
            Phi_x = torch.stack(kwargs["gt_img_11"])
            Phix = Phi_x.squeeze(dim=1)

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
        
        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        D = x
        # print(D.shape)
        D = D.view(20, 1, 33, 33)
        T = torch.zeros(D.shape).to(D.device)
        for i in range(self.stage_num):
            D_out, T = self.decos[i](D, T)
        D_out = D_out.view(-1, 1089)
        D = D.view(-1, 1089)
        T = T.view(-1, 1089)
        # if self.mode == 'train':
        #     return D,T
        # else:
        #     return T
        if mode == 'tensor':
            return T
        elif mode == 'predict':
            return [T[2:18,:], image_name, targets_GT]
        elif mode == 'loss':
            # 特征提取损失
            # loss_softiou = self.softiou(T[2:18,:], batch_x[2:18,:])
            loss_mse_1 = self.mse(D_out[2:18,:], D[2:18,:])
            loss_mse_2 = self.mse(T[2:18,:], batch_x[2:18,:])
            gamma = torch.Tensor([0.1]).to(device)
            # loss_all = loss_softiou + torch.mul(gamma, loss_mse)
            loss_all = loss_mse_2 + torch.mul(gamma, loss_mse_1)
            return {'loss': loss_all, 
                # 'loss_softiou': loss_softiou,
                'loss_mse_1': loss_mse_1,
                'loss_mse_2': loss_mse_2,
                }

class DecompositionModule(object):
    pass


class DecompositionModule(nn.Module):
    def __init__(self, slayers=6, llayers=3, mlayers=3, channel=32):
        super(DecompositionModule, self).__init__()
        self.lowrank = LowrankModule(channel=channel, layers=llayers)
        self.sparse = SparseModule(channel=channel, layers=slayers)
        self.merge = MergeModule(channel=channel, layers=mlayers)

    def forward(self, D, T):
        B = self.lowrank(D, T)
        T = self.sparse(D, B, T)
        D = self.merge(B, T)
        return D, T


class LowrankModule(nn.Module):
    def __init__(self, channel=32, layers=3):
        super(LowrankModule, self).__init__()

        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        #self.relu = nn.ReLU()
        #self.gamma = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def forward(self, D, T):
        x = D - T
        B = x + self.convs(x)
        return B


class SparseModule(nn.Module):
    def __init__(self, channel=32, layers=6) -> object:
        super(SparseModule, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        self.epsilon = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def forward(self, D, B, T):
        # print(D.shape)
        x = T + D - B
        T = x - self.epsilon * self.convs(x)
        return T


class MergeModule(nn.Module):
    def __init__(self, channel=32, layers=3):
        super(MergeModule, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.mapping = nn.Sequential(*convs)

    def forward(self, B, T):
        x = B + T
        D = self.mapping(x)
        return D