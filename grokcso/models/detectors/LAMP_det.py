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


def shrink_lamp(r_, rvar_, lam_):
  """
  Implementation of thresholding neuron in Learned AMP model.
  """
  # 计算 theta_，这里 rvar_ 和 lam_ 应该都是 tensor
  theta_ = torch.maximum(torch.sqrt(rvar_) * lam_,
                         torch.tensor(0.0, dtype=r_.dtype, device=r_.device))

  # 计算 xh_，应用符号和软阈值函数
  xh_ = torch.sign(r_) * torch.maximum(torch.abs(r_) - theta_,
                                       torch.tensor(0.0, dtype=r_.dtype,
                                                    device=r_.device))
  return xh_


class BasicBlock(torch.nn.Module):
  def __init__(self, Phi, Qinit, _lam=0.4):
    super(BasicBlock, self).__init__()

    self._M = Phi.shape[0]
    self._N = Phi.shape[1]
    B = (Phi.T / (np.linalg.norm(Phi, ord=2) ** 2)).astype(np.float32)
    self._lam = np.ones((self._N, 1), dtype=np.float32) * _lam
    self.lam = nn.Parameter(torch.from_numpy(self._lam), requires_grad=True)
    self.B = nn.Parameter(torch.from_numpy(B), requires_grad=True)

  def forward(self, xh, y, Phi, OneOverM, NOverM, vt):
    yh = torch.matmul(Phi, xh)  # 计算yh = Phi * xh

    xhl0 = torch.mean((xh.abs() > 0).float(), dim=0)  # 计算xh的绝对值大于0的均值

    bt = xhl0 * NOverM  # 计算bt = xhl0 * N/M
    vt = y - yh + bt * vt  # 计算vt

    rvar = torch.sum(vt ** 2, dim=0) * OneOverM  # 计算rvar = sum(vt^2) / M
    rh = xh + torch.matmul(self.B, vt)  # 计算rh = xh + B * vt

    xh = shrink_lamp(rh, rvar, self.lam)  # 计算xh = shrinkage(rh, lam)

    return xh, vt


@MODELS.register_module()
class LAMP(BaseModel):
    def __init__(self,
                 LayerNo,  # T，表示网络的层数
                 Phi_data_Name,
                 Qinit_Name,
                 lam=0.4
                 ):
      super(LAMP, self).__init__()

      # 加载Phi矩阵数据并转换为Tensor类型
      Phi_data = sio.loadmat(Phi_data_Name)
      Phi = Phi_data['phi']
      self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

      # 加载Qinit矩阵数据并转换为Tensor类型
      Qinit_data = sio.loadmat(Qinit_Name)
      Qinit = Qinit_data['Qinit']
      # 将 Qinit 转换为 float32 类型
      self.Qinit = Qinit.astype(np.float32)

      # 初始化网络的层数
      self.LayerNo = LayerNo

      self.M = Phi.shape[0]  # M = 121
      self.N = Phi.shape[1]  # N = 1089

      onelayer = []
      self.LayerNo = LayerNo
      for i in range(self.LayerNo):
        onelayer.append(BasicBlock(Phi, self.Qinit, lam))

      self.fcs = nn.ModuleList(onelayer)

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:

      mode = kwargs['mode']
      Phi = self.Phi
      Qinit = torch.from_numpy(self.Qinit).type(torch.FloatTensor).to(
        device)

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
      xh = torch.matmul(Qinit, y).to(device)  # 计算xh = Qinit * y

      OneOverM = torch.Tensor([1 / self.M]).to(device)  # 设置权重系数
      NOverM = torch.Tensor([self.N / self.M]).to(device)  # 设置权重系数
      vt = torch.zeros_like(y, device=device)  # 初始化vt

      for i in range(self.LayerNo):
        xh, vt = self.fcs[i](xh, y, self.Phi, OneOverM, NOverM, vt)

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