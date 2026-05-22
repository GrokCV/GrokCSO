import torch
import torch.nn as nn
from mmengine.registry import MODELS
import scipy.io as sio
from mmengine.model import BaseModel
from typing import Dict, Union
# from grokcso.models.blocks import *
from tools.utils import read_targets_from_xml_list
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class BasicBlock(torch.nn.Module):
  def __init__(self, Phi, Qinit, theta):
    super(BasicBlock, self).__init__()
    """
         初始化BasicBlock模块。

         参数:
         kwargs (dict): 模块初始化参数，包括
             - c: 通道数，用于构造网络的输入形状
    """
    self._N = Phi.shape[1]
    self._A = Phi.astype(np.float32)

    W = np.eye(self._N, dtype=np.float32) - np.matmul(Qinit, self._A)
    self.Bs = nn.Parameter(torch.from_numpy(Qinit), requires_grad=True)
    # 定义可学习的参数soft_thr，初始化为0.01
    theta = np.ones((self._N, 1), dtype=np.float32) * theta
    self.soft_thr = nn.Parameter(torch.from_numpy(theta), requires_grad=True)
    self.Ws = nn.Parameter(torch.from_numpy(W), requires_grad=True)

  def forward(self, xh, y):
    By = torch.matmul(self.Bs, y)  # 计算By = B * y
    Wxh = torch.matmul(self.Ws, xh)  # 计算Wxh = W * xh
    xh = soft_threshold(Wxh + By, self.soft_thr)

    return xh


def soft_threshold(input_tensor, theta):
  """
  软阈值函数实现
  Args:
      input_tensor: 输入张量
      theta: 阈值参数
  Returns:
      处理后的张量
  """
  # 确保阈值非负
  theta = torch.clamp(theta, min=0.0)

  # 计算软阈值
  return torch.sign(input_tensor) * torch.maximum(
    torch.abs(input_tensor) - theta,
    torch.zeros_like(input_tensor)
  )

@MODELS.register_module()
class LISTA(BaseModel):
    def __init__(self,
                 LayerNo,  # T，表示网络的层数
                 Phi_data_Name,
                 Qinit_Name,
                 lam=0.4
                 ):
      super(LISTA, self).__init__()

      # 加载Phi矩阵数据并转换为Tensor类型
      Phi_data = sio.loadmat(Phi_data_Name)
      Phi = Phi_data['phi']
      self._A = Phi.astype(np.float32)
      self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

      # 初始化网络的层数

      self._T = LayerNo
      self.M = Phi.shape[0]
      self._N = Phi.shape[1]

      self._lam = lam
      self._scale = 1.001 * np.linalg.norm(self._A, ord=2) ** 2
      print('self._scale:', self._scale)
      self._theta = (self._lam / self._scale).astype(np.float32)
      self._theta = np.ones((self._N, 1), dtype=np.float32) * self._theta

      # 加载Qinit矩阵数据并转换为Tensor类型
      Qinit_data = sio.loadmat(Qinit_Name)
      Qinit = Qinit_data['Qinit']
      # 将 Qinit 转换为 float32 类型
      self.Qinit = Qinit.astype(np.float32)
      self.Qinit1 = torch.from_numpy(self.Qinit).type(torch.FloatTensor).to(
        device)

      onelayer = []
      self.LayerNo = LayerNo
      for i in range(self.LayerNo):
        onelayer.append(BasicBlock(Phi, self.Qinit, self._theta))

      self.fcs = nn.ModuleList(onelayer)

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:

      mode = kwargs['mode']
      Phi = self.Phi
      Qinit = self.Qinit1

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
      xh = torch.matmul(Qinit, y)  # 计算xh = Qinit * y

      for i in range(self.LayerNo):
        xh = self.fcs[i](xh, y)

      x_final = xh.t()

      # 根据不同模式返回相应结果
      if mode == 'tensor':
        return x_final   # 返回最终网络输出 todo: 对网络输出结果进行后处理操作，类似nms，使得输出结果更加准确
      elif mode == 'predict':
        # 读取目标标注信息并返回预测结果
        targets_GT = read_targets_from_xml_list(ann_paths)
        return [x_final[2:18,:], image_name, targets_GT]
      elif mode == 'loss':
        # 计算损失函数
        loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))  # 计算平方误差损失

        # 损失值为对称损失和约束损失的加权和，返回损失字典
        return {'loss_discrepancy': loss_discrepancy}