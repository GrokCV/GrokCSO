import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmagic.registry import MODELS
from mmengine.registry import MODELS
import scipy.io as sio
from tools.utils import read_targets_from_xml_list
from grokcso.models.backbones.istanet_bb import BasicBlock, MSEloss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@MODELS.register_module()
class ISTANet(BaseModel):

    def __init__(self,
                 LayerNo=8
                 ):

        super(ISTANet, self).__init__()

        Phi_lrs_Name = 'SeqCSIST/data/phi_0.5.mat'
        Phi_lrs = sio.loadmat(Phi_lrs_Name)
        Phi_input = Phi_lrs['phi']
        self.Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor).to(device)

        Qinit_Name = 'SeqCSIST/data/track_5000_20/train/qinit.mat'
        Qinit_lrs = sio.loadmat(Qinit_Name)
        Qinit = Qinit_lrs['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

        onelayer = []
        self.LayerNo = LayerNo
        for i in range(LayerNo):
          onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.discrepancy = MSEloss(alpha=1)

    def forward(self, **kwargs):
        mode = kwargs['mode']
        Phi = self.Phi
        Qinit = self.Qinit
        if mode == 'loss':
            # batch_x = torch.stack(lrs['matrices'])
            # Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
            batch = torch.stack(kwargs["batch_x"])
            batch_x = batch.squeeze(dim=1)
            Phi_x = torch.stack(kwargs["gt_img_11"])
            Phix = Phi_x.squeeze(dim=1)

        elif mode == 'predict':
            image_name = []
            Phi_x = torch.stack(kwargs["gt_img_11"])
            Phix = Phi_x.squeeze(dim=1)
            ann_paths = kwargs["ann_path"]
            targets_GT = read_targets_from_xml_list(ann_paths)
            image_name = kwargs["image_name"]
            
        else:
            print("Invalid mode:", mode)
            return None
        
        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)
        
        layers_sym = []  # for computing symmetric loss
        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)
        
        x_final = x
        
        if mode == 'tensor':
            return x_final
        elif mode == 'predict':
            return [x_final[2:18,:], image_name, targets_GT]
        elif mode == 'loss':
            loss_discrepancy = self.discrepancy(x_final[2:18,:], batch_x[2:18,:])
            loss_constraint = torch.mean(torch.pow(layers_sym[0], 2))
            for k in range(7):
              loss_constraint += torch.mean(torch.pow(layers_sym[k + 1], 2))
            gamma = torch.Tensor([0.01]).to(device)
            loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

            return {'loss': loss_all, 
                'loss_discrepancy': loss_discrepancy,
                'loss_constraint': loss_constraint
                }