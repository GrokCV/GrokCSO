from torch.nn import init
import torch.nn.functional as F
from ..tricks.attention_block import *


class att_BasicBlock(torch.nn.Module):
  def __init__(self, **kwargs):
    super(att_BasicBlock, self).__init__()
    c = kwargs['c']
    self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

    self.conv1_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 1, 3, 3)))
    self.conv2_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
    self.conv1_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
    self.conv2_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(1, 64, 3, 3)))

    self.att = LSKmodule(64)
    self.c = c

  def forward(self, x, PhiTPhi, PhiTb):
    x = x - self.lambda_step * torch.mm(x, PhiTPhi)
    x = x + self.lambda_step * PhiTb
    x_input = x.view(-1, 1, 11*self.c, 11*self.c)

    x = F.conv2d(x_input, self.conv1_forward, padding=1)
    x = F.relu(x)
    x_forward = F.conv2d(x, self.conv2_forward, padding=1)

    x = torch.mul(torch.sign(x_forward),
                  F.relu(torch.abs(x_forward) - self.att(x_forward)))

    x = F.conv2d(x, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_backward = F.conv2d(x, self.conv2_backward, padding=1)

    x_pred = x_backward.view(-1, 11*11*self.c*self.c)

    x = F.conv2d(x_forward, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_est = F.conv2d(x, self.conv2_backward, padding=1)

    symloss = x_est - x_input

    return [x_pred, symloss]

