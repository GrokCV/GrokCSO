import torch
from torch.nn import init
import torch.nn.functional as F
from ..tricks.attention_block import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Dynamic_BasicBlock(torch.nn.Module):
  def __init__(self, **kwargs):
    super(Dynamic_BasicBlock, self).__init__()
    c = kwargs['c']
    lambda_weight = kwargs['lambda_weight']

    self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
    self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

    self.conv1_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 1, 3, 3)))
    self.conv2_forward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))

    self.conv1_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
    self.conv2_backward = nn.Parameter(
      init.xavier_normal_(torch.Tensor(1, 64, 3, 3)))

    self.w1 = nn.Linear(11*11*c*c, 256)
    self.w2 = nn.Linear(256, 9)

    self.lambda_weight = torch.Tensor([lambda_weight]).to(device)
    self.c = c

  def forward(self, x, PhiTPhi, PhiTb):
    minor = x

    x = x - self.lambda_step * torch.mm(x, PhiTPhi)
    x = x + self.lambda_step * PhiTb
    x_input = x.view(-1, 1, 11 * self.c, 11 * self.c)

    x = F.conv2d(x_input, self.conv1_forward, padding=1)
    x = F.relu(x)
    x_forward = F.conv2d(x, self.conv2_forward, padding=1)

    minor = self.w1(minor)
    weights = self.w2(minor)

    weights = weights.reshape(-1, 1, 3, 3)
    weights = nn.Parameter(data=weights, requires_grad=False)

    x = F.conv2d(input=x_input, weight=weights, stride=1, padding=1, groups=1)
    x = F.sigmoid(x)

    x_forward = self.lambda_weight * x_forward + (1 - self.lambda_weight) * x

    x = torch.mul(torch.sign(x_forward),
                  F.relu(torch.abs(x_forward) - self.soft_thr))

    x = F.conv2d(x, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_backward = F.conv2d(x, self.conv2_backward, padding=1)

    x_pred = x_backward.view(-1, 11 * self.c * 11 * self.c)

    x = F.conv2d(x_forward, self.conv1_backward, padding=1)
    x = F.relu(x)
    x_est = F.conv2d(x, self.conv2_backward, padding=1)

    symloss = x_est - x_input

    return [x_pred, symloss]

