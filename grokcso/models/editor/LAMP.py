import torch
import torch.nn as nn
from mmengine.registry import MODELS
import scipy.io as sio
from mmengine.model import BaseModel
from typing import Dict, Union
from tools.utils import read_targets_from_xml_list
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def shrink_lamp(r_, rvar_, lam_):
  """
  Implementation of thresholding neuron in Learned AMP model.
  """
  theta_ = torch.maximum(torch.sqrt(rvar_) * lam_,
                         torch.tensor(0.0, dtype=r_.dtype, device=r_.device))

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
    yh = torch.matmul(Phi, xh)

    xhl0 = torch.mean((xh.abs() > 0).float(), dim=0)

    bt = xhl0 * NOverM
    vt = y - yh + bt * vt

    rvar = torch.sum(vt ** 2, dim=0) * OneOverM
    rh = xh + torch.matmul(self.B, vt)

    xh = shrink_lamp(rh, rvar, self.lam)

    return xh, vt


@MODELS.register_module()
class LAMP(BaseModel):
    def __init__(self,
                 LayerNo=16,
                 Phi_data_Name="/data1/dym/GrokCSO-Dev/data/sampling_matrix/a_phi_0_3.mat",
                 Qinit_Name="/data1/dym/GrokCSO-Dev/data/initial_matrix/Q_3.mat",
                 lam=0.4
                 ):
      super(LAMP, self).__init__()

      Phi_data = sio.loadmat(Phi_data_Name)
      Phi = Phi_data['phi']
      self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

      Qinit_data = sio.loadmat(Qinit_Name)
      Qinit = Qinit_data['Qinit']
      self.Qinit = Qinit.astype(np.float32)

      self.LayerNo = LayerNo

      self.M = Phi.shape[0]
      self.N = Phi.shape[1]

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

      OneOverM = torch.Tensor([1 / self.M]).to(device)
      NOverM = torch.Tensor([self.N / self.M]).to(device)
      vt = torch.zeros_like(y, device=device)

      for i in range(self.LayerNo):
        xh, vt = self.fcs[i](xh, y, self.Phi, OneOverM, NOverM, vt)
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






