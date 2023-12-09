import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset

from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.registry import MODELS, DATASETS, METRICS

from torch.utils.data import DataLoader, default_collate
from torch.optim import Adam
from mmengine.runner import Runner

from typing import Dict, Optional, Tuple, Union
from mmengine.optim import OptimWrapper
import scipy.io as sio
import numpy as np

# # 将phi修改为字典
Phi_data_Name = 'sampling_matrix/a_phi_0_3.mat'
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']
Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

# Q作为一个固定的矩阵
Qinit_Name = 'sampling_matrix/Q_3.mat'
Qinit_data = sio.loadmat(Qinit_Name)
Qinit = Qinit_data['Qinit']
Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Qinit = Qinit.to(device)

##  训练数据存放位置
Training_data_Name = 'train_data.mat'
Training_data = sio.loadmat('data/' + Training_data_Name)
Training_labels = Training_data['matrices']

# Define ISTA-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        # nn.Parameter是一种特殊的Variable，但其默认需要求导，即requires_grad=True，
        # lambda_step和soft_thr会在训练过程中不断更新
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        # 定义了32个3*3的卷积核，输入通道数为1，输出通道数为32
        # 卷积核的个数等于输出通道数，卷积核的输入通道数等于输入图像的通道数，这里输入图像的通道数为1
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        # PhiTPhi表示phi的转置乘以phi，PhiTb表示phi的转置乘以b
        # x等于x减去lambda_step乘以x乘以PhiTPhi，再加上lambda_step乘以PhiTb，即公式一
        # x_input表示x的形状变为单通道的33*33的矩阵
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 33, 33)

        # 第一层卷积是将输入通道为1，输出通道为32，卷积核大小为3*3，padding为1，表示不改变输入图像的尺寸
        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        # 添加非线性层
        x = F.relu(x)
        # 第二层卷积是将输入通道为32，输出通道为32，卷积核大小为3*3，padding为1，表示不改变输入图像的尺寸
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        # 对x_forward进行软阈值处理，即公式二的求解，公式二本身的求解是很难的，这里采用了软阈值处理，求解出F(x)的值
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        # 对x进行反卷积，即将F(x)进行反卷积，得到x
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        # x_backward表示x的形状变为单通道的33*33的矩阵
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        # x_pred表示x_backward的形状变为单通道的1089的矩阵，方便下一次迭代计算
        x_pred = x_backward.view(-1, 1089)

        # 对x_forward进行反卷积，即将F(
        # x)进行反卷积，得到x_est用来计算loss，是为了让两个卷积相反，即让第二个卷积的卷积核为第一个卷积的反卷积
        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        # 以上代码实现了一次迭代过程，即公式一、二，中间使用了软阈值处理，完成了公式二的计算，
        return [x_pred, symloss]

@MODELS.register_module()
class ISTANet(BaseModel):
  def __init__(self, LayerNo):
    super(ISTANet, self).__init__()
    onelayer = []
    self.LayerNo = LayerNo

    for i in range(LayerNo):
      onelayer.append(BasicBlock())

    self.fcs = nn.ModuleList(onelayer)

  def forward(self, data, mode):
    batch_x = np.squeeze(data)
    batch_x = batch_x.to(device)
    Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

    PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
    PhiTb = torch.mm(Phix, Phi)

    x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

    layers_sym = []  # for computing symmetric loss

    for i in range(self.LayerNo):
      [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
      layers_sym.append(layer_sym)

    x_final = x

    if mode == 'tensor':
      return x_final
    elif mode == 'predict':
      return x_final
    elif mode == 'loss':
      return x_final,  layers_sym,  batch_x

  def train_step(self, data, optim_wrapper):
      # Enable automatic mixed precision training context.
      data = np.squeeze(data[0])
      with optim_wrapper.optim_context(self):
        x_final, layers_sym, batch_x = self.forward(data, mode='loss')
      loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))
      loss_constraint = torch.mean(torch.pow(layers_sym[0], 2))
      for k in range(8):
        loss_constraint += torch.mean(torch.pow(layers_sym[k + 1], 2))
      gamma = torch.Tensor([0.01])
      loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)
      print(loss_all)
      optim_wrapper.update_params(loss_all)
      return {'loss': loss_all, 'loss_discrepancy': loss_discrepancy,}

@DATASETS.register_module()
class RandomDataset(Dataset):
  def __init__(self, data, length):
    self.data = data
    self.len = length

  def __getitem__(self, index):
    return torch.Tensor(self.data[index, :]).float(), 0

  def __len__(self):
    return self.len


runner = Runner(
    # 你的模型
    model=ISTANet(
        LayerNo=9,
        ),
    # 模型检查点、日志等都将存储在工作路径中
    work_dir='exp/ista_model',

    # 训练所用数据
    train_dataloader=DataLoader(
        dataset=RandomDataset(
          data = Training_labels,
          length = 88912
            ),
        shuffle=True,
        pin_memory=True,
        batch_size=64,
        num_workers=2),
    # 训练相关配置
    train_cfg=dict(
        by_epoch=True,   # 根据 epoch 计数而非 iteration
        max_epochs=150,
        ),
    # 优化器封装，MMEngine 中的新概念，提供更丰富的优化选择。
    # 通常使用默认即可，可缺省。有特殊需求可查阅文档更换，如
    # 'AmpOptimWrapper' 开启混合精度训练
    optim_wrapper=dict(
        optimizer=dict(
            type=Adam,
            lr=0.0001)),
    # 参数调度器，用于在训练中调整学习率/动量等参数
)

# 开始训练你的模型吧
runner.train()