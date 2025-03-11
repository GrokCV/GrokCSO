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


def initialize_weights(self):
  for m in self.modules():
    if isinstance(m, nn.Conv2d):
      init.xavier_normal_(m.weight)
      if m.bias is not None:
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
      init.constant_(m.weight, 1)
      init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
      init.normal_(m.weight, 0, 0.01)
      init.constant_(m.bias, 0)


class Fista_BasicBlock(torch.nn.Module):

  def __init__(self, features=32):
    super(Fista_BasicBlock, self).__init__()
    self.Sp = nn.Softplus()

    self.conv_D = nn.Conv2d(1, features, (3, 3), stride=1, padding=1)
    self.conv1_forward = nn.Conv2d(features, features, (3, 3), stride=1,
                                   padding=1)
    self.conv2_forward = nn.Conv2d(features, features, (3, 3), stride=1,
                                   padding=1)
    self.conv3_forward = nn.Conv2d(features, features, (3, 3), stride=1,
                                   padding=1)
    self.conv4_forward = nn.Conv2d(features, features, (3, 3), stride=1,
                                   padding=1)

    self.conv1_backward = nn.Conv2d(features, features, (3, 3), stride=1,
                                    padding=1)
    self.conv2_backward = nn.Conv2d(features, features, (3, 3), stride=1,
                                    padding=1)
    self.conv3_backward = nn.Conv2d(features, features, (3, 3), stride=1,
                                    padding=1)
    self.conv4_backward = nn.Conv2d(features, features, (3, 3), stride=1,
                                    padding=1)
    self.conv_G = nn.Conv2d(features, 1, (3, 3), stride=1, padding=1)

  def forward(self, x, PhiTPhi, PhiTb, lambda_step, soft_thr):
    x = x - self.Sp(lambda_step) * torch.mm(x, PhiTPhi)
    x = x + self.Sp(lambda_step) * PhiTb

    x_input = x.view(-1, 1, 33, 33)

    x_D = self.conv_D(x_input)

    x = self.conv1_forward(x_D)
    x = F.relu(x)
    # x = self.conv2_forward(x)
    # x = F.relu(x)
    # x = self.conv3_forward(x)
    # x = F.relu(x)
    x_forward = self.conv4_forward(x)

    # soft-thresholding block
    x_st = torch.mul(torch.sign(x_forward),
                     F.relu(torch.abs(x_forward) - self.Sp(soft_thr)))

    x = self.conv1_backward(x_st)
    x = F.relu(x)
    # x = self.conv2_backward(x)
    # x = F.relu(x)
    # x = self.conv3_backward(x)
    # x = F.relu(x)
    x_backward = self.conv4_backward(x)

    x_G = self.conv_G(x_backward)

    # prediction output (skip connection); non-negative output
    x_pred = F.relu(x_input + x_G)
    x_pred = x_pred.view(-1, 1089)

    # compute symmetry loss
    x = self.conv1_backward(x_forward)
    x = F.relu(x)
    # x = self.conv2_backward(x)
    # x = F.relu(x)
    # x = self.conv3_backward(x)
    # x = F.relu(x)
    x_D_est = self.conv4_backward(x)
    symloss = x_D_est - x_D

    return [x_pred, symloss, x_st]


def l1_loss(pred, target, l1_weight):
  """
  Compute L1 loss;
  l1_weigh default: 0.1
  """
  err = torch.mean(torch.abs(pred - target))
  err = l1_weight * err
  return err


@MODELS.register_module()
class Fista(BaseModel):
    def __init__(self,
                 LayerNo=7,
                 Phi_data_Name="phi_3.mat",
                 Qinit_Name="Q_3.mat"
                 ):
        super(Fista, self).__init__()

        self.LayerNo = LayerNo

        Phi_data = sio.loadmat(Phi_data_Name)
        Phi = Phi_data['phi']
        self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

        Qinit_data = sio.loadmat(Qinit_Name)
        Qinit = Qinit_data['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

        onelayer = []

        self.bb = Fista_BasicBlock()
        for i in range(LayerNo):
            onelayer.append(self.bb)

        self.fcs = nn.ModuleList(onelayer)
        self.fcs.apply(initialize_weights)

        # thresholding value
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        # gradient step
        self.w_mu = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu = nn.Parameter(torch.Tensor([0.1]))
        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        self.b_rho = nn.Parameter(torch.Tensor([0]))

        self.Sp = nn.Softplus()

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

        batch_x = torch.stack(kwargs["gt"])
        batch_x = batch_x.squeeze(dim=1).to(device)

        ann_paths = kwargs["ann_path"]
        file_id = kwargs["file_id"]

      PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
      PhiTb = torch.mm(Phix, Phi)

      x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

      xold = x
      y = xold
      layers_sym = []  # for computing symmetric loss
      layers_st = []
      intermediate = [x]

      for i in range(self.LayerNo):
        theta_ = self.w_theta * i + self.b_theta
        mu_ = self.w_mu * i + self.b_mu

        [xnew, layer_sym, layer_st] = self.fcs[i](y, PhiTPhi, PhiTb, mu_, theta_)
        intermediate.append(xnew)
        rho_ = (self.Sp(self.w_rho * i + self.b_rho) - self.Sp(
          self.b_rho)) / self.Sp(self.w_rho * i + self.b_rho)
        y = xnew + rho_ * (xnew - xold)  # two-step update
        xold = xnew

        layers_st.append(layer_st)
        layers_sym.append(layer_sym)

      x_final = xnew

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
        loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2)) + \
                           l1_loss(x_final, batch_x, 0.1)
        loss_constraint = 0
        for k, _ in enumerate(layers_sym, 0):
          loss_constraint += torch.mean(torch.pow(layers_sym[k], 2))

        sparsity_constraint = 0
        for k, _ in enumerate(layers_st, 0):
          sparsity_constraint += torch.mean(torch.abs(layers_st[k]))

        # loss = loss_discrepancy + gamma * loss_constraint
        loss = loss_discrepancy + 0.01 * loss_constraint + 0.001 * sparsity_constraint
        return {'loss': loss}






