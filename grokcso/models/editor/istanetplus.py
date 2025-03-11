import torch
import torch.nn as nn
from mmengine.registry import MODELS
import scipy.io as sio
import torch.nn.functional as F
from typing import Dict, Union
from torch.nn import init
from mmengine.model import BaseModel
from tools.utils import read_targets_from_xml_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BasicBlock(torch.nn.Module):
  def __init__(self, **kwargs):
    super(BasicBlock, self).__init__()
    c = kwargs['c']
    self.c = c

    self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
    self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

    self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

    self.conv1_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
    self.conv2_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
    self.conv1_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
    self.conv2_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

    self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

  def forward(self, x, PhiTPhi, PhiTb):
    x = x - self.lambda_step * torch.mm(x, PhiTPhi)
    x = x + self.lambda_step * PhiTb
    x_input = x.view(-1, 1, 11 * self.c, 11 * self.c)

    x_D = F.conv2d(x_input, self.conv_D, padding=1)

    x = F.conv2d(x_D, self.conv1_forward, padding=1)
    x = F.relu(x)
    x_forward = F.conv2d(x, self.conv2_forward, padding=1)

    x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) -
                                                self.soft_thr))

    x = F.conv2d(x, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_backward = F.conv2d(x, self.conv2_backward, padding=1)

    x_G = F.conv2d(x_backward, self.conv_G, padding=1)

    x_pred = x_input + x_G

    x_pred = x_pred.view(-1, 11 * self.c * 11 * self.c)

    x = F.conv2d(x_forward, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
    symloss = x_D_est - x_D

    return [x_pred, symloss]


@MODELS.register_module()
class ISTANetplus(BaseModel):
    def __init__(self, LayerNo,
                 Phi_data_Name,
                 Qinit_Name,
                 c=3):
        super(ISTANetplus, self).__init__()

        Phi_data = sio.loadmat(Phi_data_Name)
        Phi_input = Phi_data['phi']
        self.Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor).to(device)

        Qinit_data = sio.loadmat(Qinit_Name)
        Qinit = Qinit_data['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock(c=c))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:
      mode = kwargs['mode']
      Phi = self.Phi
      Qinit = self.Qinit

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

      PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
      PhiTb = torch.mm(Phix, Phi)

      x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

      layers_sym = []  # for computing symmetric loss
      intermediate = [x]

      for i in range(self.LayerNo):
        [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
        layers_sym.append(layer_sym)
        intermediate.append(x)

      x_final = x

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
        loss_constraint = torch.mean(torch.pow(layers_sym[0], 2))
        for k in range(self.LayerNo - 1):
          loss_constraint += torch.mean(torch.pow(layers_sym[k + 1], 2))
        gamma = torch.Tensor([0.01]).to(device)
        return {'loss_discrepancy': loss_discrepancy, '0.01 * loss_constraint': torch.mul(gamma, loss_constraint)}

