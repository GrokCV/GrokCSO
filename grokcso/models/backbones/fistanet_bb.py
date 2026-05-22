import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

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