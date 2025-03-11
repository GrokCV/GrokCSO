import torch
import torch.nn as nn
from mmengine.registry import MODELS
import scipy.io as sio
from mmengine.model import BaseModel
from typing import Dict, Union
from tools.utils import read_targets_from_xml_list
import numpy as np
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BasicBlock(torch.nn.Module):
  def __init__(self, Phi, Qinit, theta):
    super(BasicBlock, self).__init__()
    self._N = Phi.shape[1]
    self._A = Phi.astype(np.float32)

    W = np.eye(self._N, dtype=np.float32) - np.matmul(Qinit, self._A)
    self.Bs = nn.Parameter(torch.from_numpy(Qinit), requires_grad=True)
    theta = np.ones((self._N, 1), dtype=np.float32) * theta
    self.soft_thr = nn.Parameter(torch.from_numpy(theta), requires_grad=True)
    self.Ws = nn.Parameter(torch.from_numpy(W), requires_grad=True)

  def forward(self, xh, y):
    By = torch.matmul(self.Bs, y)
    Wxh = torch.matmul(self.Ws, xh)
    xh = soft_threshold(Wxh + By, self.soft_thr)

    return xh


def soft_threshold(input_tensor, theta):

  theta = torch.clamp(theta, min=0.0)

  return torch.sign(input_tensor) * torch.maximum(
    torch.abs(input_tensor) - theta,
    torch.zeros_like(input_tensor)
  )


@MODELS.register_module()
class LISTA(BaseModel):
    def __init__(self,
                 LayerNo=16,
                 Phi_data_Name="phi_3.mat",
                 Qinit_Name="Q_3.mat",
                 lam=0.4
                 ):
      super(LISTA, self).__init__()

      Phi_data = sio.loadmat(Phi_data_Name)
      Phi = Phi_data['phi']
      self._A = Phi.astype(np.float32)
      self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

      Qinit_data = sio.loadmat(Qinit_Name)
      Qinit = Qinit_data['Qinit']
      self.Qinit = Qinit.astype(np.float32)
      self.Qinit1 = torch.from_numpy(self.Qinit).type(torch.FloatTensor).to(
        device)

      self._T = LayerNo
      self.M = Phi.shape[0]
      self._N = Phi.shape[1]

      self._lam = lam
      self._scale = 1.001 * np.linalg.norm(self._A, ord=2) ** 2
      print('self._scale:', self._scale)
      self._theta = (self._lam / self._scale).astype(np.float32)
      self._theta = np.ones((self._N, 1), dtype=np.float32) * self._theta

      onelayer = []
      self.LayerNo = LayerNo
      for i in range(self.LayerNo):
        onelayer.append(BasicBlock(Phi, self.Qinit, self._theta))

      self.fcs = nn.ModuleList(onelayer)

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:

      mode = kwargs['mode']
      Phi = self.Phi
      Qinit = self.Qinit1

      if mode == 'tensor':
        Phix = kwargs["Phix"]
      else:
        Phix = torch.stack(kwargs["gt_img_11"])
        Input_image = torch.stack(kwargs["gt_img_11"])
        Phix = Phix.squeeze(dim=1).to(device)

        batch_x = torch.stack(kwargs["gt"]).squeeze(dim=1).to(device)
        ann_paths = kwargs["ann_path"]
        file_id = kwargs["file_id"]

      y = Phix.t()

      xh = torch.matmul(Qinit, y)
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






