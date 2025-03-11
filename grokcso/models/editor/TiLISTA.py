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

  return torch.sign(input_) * torch.maximum(torch.abs(input_) - theta_,
                                            torch.tensor(0.0))


def shrink_ss(inputs_, theta_, q):

  abs_ = torch.abs(inputs_)

  thres_ = torch.quantile(
    abs_,
    1.0 - q / 100.0,
    dim=0,
    keepdim=True
  )

  index_ = torch.logical_and(abs_ > theta_, abs_ > thres_)
  index_ = index_.to(inputs_.dtype)

  index_ = index_.detach()

  cindex_ = 1.0 - index_

  return (torch.mul(index_, inputs_) +
          shrink_free(torch.mul(cindex_, inputs_), theta_))


class BasicBlock(torch.nn.Module):
  def __init__(self, Phi, Qinit, theta):
    super(BasicBlock, self).__init__()
    # Initialize the shared weight matrix W
    self.W = nn.Parameter(torch.from_numpy(Qinit), requires_grad=True)

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

  def __init__(self, LayerNo=16,
               Phi_data_Name="phi_3.mat",
               Qinit_Name="Q_3.mat"
               ):
    super(TiLISTA, self).__init__()

    Phi_data = sio.loadmat(Phi_data_Name)
    Phi = Phi_data['phi']
    self._A = Phi.astype(np.float32)
    self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)
    self._T = LayerNo

    Qinit_data = sio.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']
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

      batch_x = torch.stack(kwargs["gt"]).squeeze(dim=1).to(device)
      ann_paths = kwargs["ann_path"]
      file_id = kwargs["file_id"]

    y = Phix.t()

    xh = torch.matmul(Qinit, y)
    intermediate = [xh.t()]

    for i in range(self.LayerNo):
      percent = self._ps[i]
      res = y - torch.matmul(Phi, xh)
      xh = self.fcs[i](xh, y, percent, res)
      intermediate.append(xh.t())

    x_final = xh.t()

    if mode == 'tensor':
      return [x_final, intermediate]
    elif mode == 'predict':
      targets_GT = read_targets_from_xml_list(ann_paths)
      return [{"x_final": x_final,
               "targets_GT": targets_GT,
               "file_id": file_id,
               "Input_image": Input_image,
               "img_gt": batch_x}]
    elif mode == 'loss':
      loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))

      return {'loss_discrepancy': loss_discrepancy}
