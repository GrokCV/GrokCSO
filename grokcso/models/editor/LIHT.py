import torch
import numpy as np
import torch.nn as nn
import torch
from mmengine.registry import MODELS
import scipy.io as sio
from mmengine.model import BaseModel
from typing import Dict, Union
from tools.utils import read_targets_from_xml_list
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hard_shrink(r_, tau_):
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

  def __init__(self, LayerNo=16,
               Phi_data_Name="phi_3.mat",
               Qinit_Name="Q_3.mat"
               ):
    super(LIHT, self).__init__()

    Phi_data = sio.loadmat(Phi_data_Name)
    Phi = Phi_data['phi']
    self._A = Phi.astype(np.float32)
    self.Phi1 = torch.from_numpy(Phi).type(torch.FloatTensor)
    self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

    Qinit_data = sio.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']
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

    xh = torch.matmul(self.Qinit, y)

    intermediate = [xh.t()]

    for i in range(self.LayerNo):
      xh = self.fcs[i](xh, y)
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

