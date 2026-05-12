import torch
import scipy.io
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import read_targets_from_xml_list
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from grokcso.models.backbones.ISTA_Net_pp_bb import *
from tools.utils import get_cond

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@MODELS.register_module()
class ISTA_Net_pp(BaseModel):
    def __init__(self, LayerNo):
        super(ISTA_Net_pp, self).__init__()
        
        Phi_lrs_Name = 'SeqCSIST/data/phi_0.5.mat'
        Phi_lrs = sio.loadmat(Phi_lrs_Name)
        Phi_input = Phi_lrs['phi']
        self.Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor).to(device)
        self.sigma = 0.5
        
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.condition = condition_network(LayerNo)

    def forward(self, **kwargs): 
        mode = kwargs['mode']
        Phi = self.Phi
        cond = get_cond(self.sigma, 0, 'org_sigma')
        lambda_step,x_step = self.condition(cond)
        
        if mode == 'loss':
            batch = torch.stack(kwargs["batch_x"])
            batch_x = batch.squeeze(dim=1)
            Phi_x = torch.stack(kwargs["gt_img_11"])
            Phix = Phi_x.squeeze(dim=1)
            PhiWeight = Phi.contiguous().view(121, 1, 33, 33)

        elif mode == 'predict':
            image_name = []
            Phi_x = torch.stack(kwargs["gt_img_11"])
            Phix = Phi_x.squeeze(dim=1)
            ann_paths = kwargs["ann_path"]
            targets_GT = read_targets_from_xml_list(ann_paths)
            image_name = kwargs["image_name"]
            PhiWeight = Phi.contiguous().view(121, 1, 33, 33)

        # Initialization-subnet
        Phi = torch.squeeze(Phi)
        PhiTWeight = Phi.t().contiguous().view(1089, 121, 1, 1)
        PhiTb = F.conv2d(Phix.view(-1, 121, 1, 1), PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb   

        for i in range(self.LayerNo):
            x = self.fcs[i](x,  PhiWeight, PhiTWeight, PhiTb,lambda_step[i],x_step[i])


        x_final = x.view(-1,1089)
        

        if mode == 'predict':
            return [x_final[2:18], image_name, targets_GT]
        elif mode == 'loss':
            loss_discrepancy = torch.mean(torch.pow(x_final[2:18] - batch_x[2:18], 2))
            return {'loss_all': loss_discrepancy}
        elif mode == 'tensor':
            return x_final, image_name