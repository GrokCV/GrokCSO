import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class BasicBlock(torch.nn.Module):
  def __init__(self):
    super(BasicBlock, self).__init__()

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
    x_input = x.view(-1, 1, 33, 33)

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

    x_pred = x_pred.view(-1, 1089)

    x = F.conv2d(x_forward, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
    symloss = x_D_est - x_D

    return [x_pred, symloss]