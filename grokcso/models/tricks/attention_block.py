import torch
import torch.nn as nn
import torch.nn.functional as F


class LSKmodule(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.convl = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
    self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
    self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
    self.conv_squeeze = nn.Conv2d(2, 2, 3, padding=1)
    self.conv_m = nn.Conv2d(dim // 2, dim, 1)

  def forward(self, x):
    attn1 = x
    attn2 = self.convl(attn1)

    attn1 = self.conv0_s(attn1)
    attn2 = self.conv1_s(attn2)

    attn = torch.cat([attn1, attn2], dim=1)
    avg_attn = torch.mean(attn, dim=1, keepdim=True)
    max_attn, _ = torch.max(attn, dim=1, keepdim=True)
    agg = torch.cat([avg_attn, max_attn], dim=1)
    sig = self.conv_squeeze(agg).sigmoid()

    attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
           attn2 * sig[:, 1, :, :].unsqueeze(1)
    attn = self.conv_m(attn)

    return x * attn

