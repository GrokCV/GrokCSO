import torch
import torch.nn as nn
import torch.nn.functional as F

Qinit = None
def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)

class condition_network(nn.Module):
    def __init__(self,LayerNo):
        super(condition_network, self).__init__()

        self.fc1 = nn.Linear(1, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, LayerNo+LayerNo, bias=True)

        self.act123 = nn.LeakyReLU(inplace=True)
        # self.act3 = nn.Softplus()

    def forward(self, x):
        x=x[:,0:1]

        x = self.act123(self.fc1(x))
        x = self.act123(self.fc2(x))
        x = self.act123(self.fc3(x))
        num=x.shape[1]
        num=int(num/2)
        return x[0,0:num],x[0,num:]

class ResidualBlock_basic(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_basic, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = x[1]
        content = x[0]

        out = self.act(self.conv1(content))
        out = self.conv2(out)
        return content + out, cond


class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.head_conv = nn.Conv2d(2, 32, 3, 1, 1, bias=True)
        self.ResidualBlocks = nn.Sequential(
            ResidualBlock_basic(nf=32),
            ResidualBlock_basic(nf=32)
        )
        self.tail_conv = nn.Conv2d(32, 1, 3, 1, 1, bias=True)


    def forward(self, x,  PhiWeight, PhiTWeight, PhiTb,lambda_step,x_step):
        x = x - lambda_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x = x + lambda_step * PhiTb
        x_input = x

        noise= x_step.repeat(x_input.shape[0], 1, x_input.shape[2], x_input.shape[3])
        x_input_cat = torch.cat((x_input,noise),1)

        x_mid = self.head_conv(x_input_cat)
        cond = None
        x_mid, cond = self.ResidualBlocks([x_mid, cond])
        x_mid = self.tail_conv(x_mid)

        x_pred = x_input + x_mid

        return x_pred