import torch
import numpy as np
import torch.nn as nn
import torch
from mmengine.registry import MODELS
import scipy.io as sio
from mmengine.model import BaseModel
from typing import Dict, Union
from tools.utils import read_targets_from_xml_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn


def shrink_free(input_, theta_):
  """
  不带非负约束的软收缩函数的PyTorch实现。

  Args:
      input_: 输入张量
      theta_: 收缩阈值
  """
  return torch.sign(input_) * torch.maximum(torch.abs(input_) - theta_,
                                            torch.tensor(0.0))


def shrink_ss(inputs_, theta_, q):
  """
  对幅度最大的q%的元素不进行软收缩的特殊收缩函数。

  Args:
      inputs_: 输入张量
      theta_: 收缩阈值
      q: 百分比值(0-100)
  """
  abs_ = torch.abs(inputs_)

  # 计算(100-q)%分位数作为阈值
  # PyTorch没有直接的percentile函数,需要使用quantile
  # 注意keepdim=True保持维度一致
  thres_ = torch.quantile(
    abs_,
    1.0 - q / 100.0,
    dim=0,
    keepdim=True
  )

  # 同时满足两个条件的元素会被选入支撑集:
  # 1. 绝对值大于theta_
  # 2. 绝对值大于分位数阈值
  index_ = torch.logical_and(abs_ > theta_, abs_ > thres_)
  index_ = index_.to(inputs_.dtype)  # 转换为浮点类型

  # 使用detach()代替tf.stop_gradient
  # 这样在反向传播时不会计算这部分梯度
  index_ = index_.detach()

  # 计算补集索引
  cindex_ = 1.0 - index_

  # 对支撑集中的元素保持原值,对补集中的元素进行收缩
  return (torch.mul(index_, inputs_) +
          shrink_free(torch.mul(cindex_, inputs_), theta_))


class BasicBlock(torch.nn.Module):
  def __init__(self, Phi, Qinit, theta):
    super(BasicBlock, self).__init__()
    """
         初始化BasicBlock模块。

         参数:
         kwargs (dict): 模块初始化参数，包括
             - c: 通道数，用于构造网络的输入形状
    """
    # Initialize the shared weight matrix W
    self.W = nn.Parameter(torch.from_numpy(Qinit), requires_grad=True)

    # 定义可学习的参数soft_thr，初始化为0.01
    self.theta = nn.Parameter(torch.from_numpy(theta))
    self.alpha = nn.Parameter(torch.Tensor([1.0]))

  def forward(self, xh, y, percent, res):
    zh = xh + self.alpha * torch.matmul(self.W, res)
    xh = shrink_ss(zh, self.theta, percent)

    return xh


@MODELS.register_module()
class TiLISTA(BaseModel):
  """
  Implementation of deep neural network model in PyTorch.
  """

  def __init__(self, LayerNo=16,  # T，表示网络的层数
               Phi_data_Name="/data1/dym/GrokCSO-Dev/data/sampling_matrix/a_phi_0_3.mat",
               Qinit_Name="/data1/dym/GrokCSO-Dev/data/initial_matrix/Q_3.mat"
               ):
    super(TiLISTA, self).__init__()

    Phi_data = sio.loadmat(Phi_data_Name)
    Phi = Phi_data['phi']
    self._A = Phi.astype(np.float32)
    self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)
    self._T = LayerNo

    # 加载Qinit矩阵数据并转换为Tensor类型
    Qinit_data = sio.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']
    # 将 Qinit 转换为 float32 类型
    self.Qinit = Qinit.astype(np.float32)
    self.W = torch.from_numpy(self.Qinit).type(torch.FloatTensor).to(
      device)

    self._p = 1.2
    self._maxp = 13
    self._lam = 0.4
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]
    self._scale = 1.001 * np.linalg.norm(self._A, ord=2) ** 2
    self._theta = (self._lam / self._scale).astype(np.float32)
    self._theta = np.ones((self._N, 1), dtype=np.float32) * self._theta

    self._ps = [(t + 1) * self._p for t in range(self._T)]
    self._ps = np.clip(self._ps, 0.0, self._maxp)

    onelayer = []
    self.LayerNo = LayerNo
    for i in range(self.LayerNo):
      onelayer.append(BasicBlock(Phi, self.Qinit, self._theta))

    self.fcs = nn.ModuleList(onelayer)

  def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:

    mode = kwargs['mode']
    Phi = self.Phi
    Qinit = self.W
    if mode == 'tensor':
      Phix = kwargs["Phix"]
    else:
      Phix = torch.stack(kwargs["gt_img_11"])
      Input_image = torch.stack(kwargs["gt_img_11"])
      Phix = Phix.squeeze(dim=1).to(device)
      # Phix = Phix - self.noise_level

      # 获取gt图像
      batch_x = torch.stack(kwargs["gt"]).squeeze(dim=1).to(device)
      # 获取标注路径
      ann_paths = kwargs["ann_path"]
      # 记录文件ID
      file_id = kwargs["file_id"]

    # y 是phix转置
    y = Phix.t()

    # 初始化网络的输入x
    xh = torch.matmul(Qinit, y)  # 计算xh = Qinit * y
    intermediate = [xh.t()]

    for i in range(self.LayerNo):
      percent = self._ps[i]
      res = y - torch.matmul(Phi, xh)
      xh = self.fcs[i](xh, y, percent, res)
      intermediate.append(xh.t())

    x_final = xh.t()

    # 根据不同模式返回相应结果
    if mode == 'tensor':
      return [x_final, intermediate]
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
      loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))  # 计算平方误差损失

      # 损失值为对称损失和约束损失的加权和，返回损失字典
      return {'loss_discrepancy': loss_discrepancy}


if __name__ == '__main__':
  from mmengine.analysis import get_model_complexity_info
  from mmengine.analysis.print_helper import _format_size

  model = TiLISTA()
  model = model.to(device)

  outputs = get_model_complexity_info(model, input_shape=(121, 64),
                                           show_table=True, show_arch=True)
  flops, params = _format_size(outputs['flops']), _format_size(outputs['params'])

  print("flops:{}".format(flops))
  print("params:{}".format(params))