import torch
import numpy as np
import torch.nn as nn
import torch
from mmengine.registry import MODELS
import scipy.io as sio
from mmengine.model import BaseModel
from typing import Dict, Union
# from grokcso.models.blocks import *
from tools.utils import read_targets_from_xml_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def hard_shrink(r_, tau_):
  """
  硬阈值神经元的PyTorch实现。

  Args:
      r_: 输入张量
      tau_: 阈值

  Returns:
      硬阈值处理后的张量
  """
  return torch.relu(torch.sign(torch.abs(r_) - tau_)) * r_


class BasicBlock(torch.nn.Module):
  def __init__(self, theta, Q, W):
    super(BasicBlock, self).__init__()
    self.B = nn.Parameter(Q)
    self.W = nn.Parameter(W)
    self.theta = nn.Parameter(theta)

  def forward(self, xh, y):

    By = torch.matmul(self.B, y)
    xh = hard_shrink(torch.matmul(self.W, xh) + By, self.theta)

    return xh

@MODELS.register_module()
class LIHT(BaseModel):
  """
  Implementation of deep neural network model in PyTorch.
  """

  def __init__(self, LayerNo,  # T，表示网络的层数
                 Phi_data_Name,
                 Qinit_Name):
    super(LIHT, self).__init__()

    Phi_data = sio.loadmat(Phi_data_Name)
    Phi = Phi_data['phi']
    self._A = Phi.astype(np.float32)
    self.Phi1 = torch.from_numpy(Phi).type(torch.FloatTensor)
    self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

    # 加载Qinit矩阵数据并转换为Tensor类型
    Qinit_data = sio.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']
    # 将 Qinit 转换为 float32 类型
    self.Qinit = Qinit.astype(np.float32)
    self.Q = torch.from_numpy(self.Qinit).type(torch.FloatTensor)
    self.Qinit = self.Q.to(device)
    self.QT = self.Q.t()

    self.W = torch.eye(self._A.shape[1], dtype=torch.float32)

    self._T = LayerNo
    self._p = 1.2
    self._lam = 0.4


    self.theta = np.sqrt(self._lam)
    self.theta = torch.ones((self._N, 1), dtype=torch.float32) * self.theta

    onelayer = []
    self.LayerNo = LayerNo
    for i in range(self.LayerNo):
      onelayer.append(BasicBlock(self.theta, self.Q, self.W))

    self.fcs = nn.ModuleList(onelayer)

  def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:

    mode = kwargs['mode']
    Phi = self.Phi
    # 处理 'loss' 模式下的输入数据
    if mode == 'loss':
      batch_x = torch.stack(kwargs["batch_x"])  # 获取真实图像的batch
      Phix = torch.stack(kwargs["gt_img_11"]).squeeze(dim=1)
    elif mode == 'predict':
      Phix = torch.stack(kwargs["gt_img_11"]).squeeze(dim=1)  # 获取输入图像并移除无关维度
      ann_paths = kwargs["ann_path"]  # 获取标注路径
      image_name = kwargs["image_name"]
      Input_image = torch.stack(kwargs["gt_img_11"])  # 记录输入图像

    # y 是phix转置
    y = Phix.t()

    # 初始化网络的输入x
    xh = torch.matmul(self.Qinit, y)  # 计算xh = Qinit * y

    for i in range(self.LayerNo):
      xh = self.fcs[i](xh, y)

    x_final = xh.t()

    # 根据不同模式返回相应结果
    if mode == 'tensor':
      return x_final  # 返回最终网络输出 todo: 对网络输出结果进行后处理操作，类似nms，使得输出结果更加准确
    elif mode == 'predict':
      # 读取目标标注信息并返回预测结果
      targets_GT = read_targets_from_xml_list(ann_paths)
      return [x_final[2:18,:], image_name, targets_GT]
    elif mode == 'loss':
      # 计算损失函数
      loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))  # 计算平方误差损失

      # 损失值为对称损失和约束损失的加权和，返回损失字典
      return {'loss_discrepancy': loss_discrepancy}