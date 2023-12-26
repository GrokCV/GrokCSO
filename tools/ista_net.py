import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset

from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.registry import MODELS, DATASETS, METRICS

from torch.utils.data import DataLoader, default_collate
from torch.optim import Adam
from mmengine.runner import Runner

from typing import Dict, Optional, Tuple, Union
from mmengine.optim import OptimWrapper
import scipy.io as sio
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Phi_data_Name = 'sampling_matrix/a_phi_0_3.mat'
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']
Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

Qinit_Name = 'sampling_matrix/Q_3.mat'
Qinit_data = sio.loadmat(Qinit_Name)
Qinit = Qinit_data['Qinit']
Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Qinit = Qinit.to(device)

Training_data_Name = 'train_data.mat'
Training_data = sio.loadmat('data/' + Training_data_Name)
Training_labels = Training_data['matrices']

# Define ISTA-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 33, 33)

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        x_pred = x_backward.view(-1, 1089)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]

@MODELS.register_module()
class ISTANet(BaseModel):
  def __init__(self, LayerNo):
    super(ISTANet, self).__init__()
    onelayer = []
    self.LayerNo = LayerNo

    for i in range(LayerNo):
      onelayer.append(BasicBlock())

    self.fcs = nn.ModuleList(onelayer)

  def forward(self, data, mode):
    batch_x = np.squeeze(data)
    batch_x = batch_x.to(device)
    Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

    PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
    PhiTb = torch.mm(Phix, Phi)

    x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

    layers_sym = []  # for computing symmetric loss

    for i in range(self.LayerNo):
      [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
      layers_sym.append(layer_sym)

    x_final = x

    if mode == 'tensor':
      return x_final
    elif mode == 'predict':
      return x_final
    elif mode == 'loss':
      return x_final,  layers_sym,  batch_x

  def train_step(self, data, optim_wrapper):
      # Enable automatic mixed precision training context.
      data = np.squeeze(data[0])
      with optim_wrapper.optim_context(self):
        x_final, layers_sym, batch_x = self.forward(data, mode='loss')
      loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))
      loss_constraint = torch.mean(torch.pow(layers_sym[0], 2))
      for k in range(8):
        loss_constraint += torch.mean(torch.pow(layers_sym[k + 1], 2))
      gamma = torch.Tensor([0.01]).to(device)
      loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)
      print(loss_all)
      optim_wrapper.update_params(loss_all)
      return {'loss': loss_all, 'loss_discrepancy': loss_discrepancy,}

@DATASETS.register_module()
class RandomDataset(Dataset):
  def __init__(self, data, length):
    self.data = data
    self.len = length

  def __getitem__(self, index):
    return torch.Tensor(self.data[index, :]).float(), 0

  def __len__(self):
    return self.len


runner = Runner(
    model=ISTANet(
        LayerNo=9,
        ),
    work_dir='exp/ista_model',

    train_dataloader=DataLoader(
        dataset=RandomDataset(
          data = Training_labels,
          length = 88912
            ),
        shuffle=True,
        pin_memory=True,
        batch_size=64,
        num_workers=2),
    train_cfg=dict(
        by_epoch=True,
        max_epochs=150,
        ),
    optim_wrapper=dict(
        optimizer=dict(
            type=Adam,
            lr=0.0001)),
)

runner.train()
