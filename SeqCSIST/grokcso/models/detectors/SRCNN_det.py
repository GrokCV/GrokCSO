import torch
import sys
import torch.nn as nn
from mmengine.model import BaseModel
from mmagic.registry import MODELS
from mmengine.registry import MODELS
import scipy.io as sio
from tools.utils import read_targets_from_xml_list
from grokcso.models.backbones.SRCNN_bb import SRCNN, MSEloss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@MODELS.register_module()
class SRCNN_det(BaseModel):

    def __init__(self
                 ):

        super(SRCNN_det, self).__init__()

        Phi_lrs_Name = 'SeqCSIST/data/phi_0.5.mat'
        Phi_lrs = sio.loadmat(Phi_lrs_Name)
        Phi_input = Phi_lrs['phi']
        self.Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor).to(device)

        Qinit_Name = 'SeqCSIST/data/track_5000_20/train/qinit.mat'
        Qinit_lrs = sio.loadmat(Qinit_Name)
        Qinit = Qinit_lrs['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

        self.SRCNN = SRCNN()

        #self.lq_pixel_loss = MSEloss(alpha=0.9)
        self.fg_loss = MSEloss(alpha=1)

    def forward(self, **kwargs):
        mode = kwargs['mode']
        Phi = self.Phi
        Qinit = self.Qinit
        if mode == 'loss':
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
        
        x = x.view(-1, 1, 33, 33)
        final = self.SRCNN(x)
        final = final.view(-1, 1089)
        
        
        if mode == 'tensor':
            return [final[2:18], image_name, targets_GT]
        elif mode == 'predict':
            return [final[2:18], image_name, targets_GT]
        elif mode == 'loss':
            # 回归损失
            loss_fg = self.fg_loss(final, batch_x)

            loss_all = loss_fg
            return {'loss': loss_all,
                'loss_reg': loss_fg,
                }
