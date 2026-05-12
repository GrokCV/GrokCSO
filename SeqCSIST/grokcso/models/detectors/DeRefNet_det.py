import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmagic.registry import MODELS
from mmengine.registry import MODELS
import scipy.io as sio
from tools.utils import read_targets_from_xml_list
from grokcso.models.backbones.DeRefNet_bb import BasicBlock, MSEloss, TDDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@MODELS.register_module()
class DeRefNet(BaseModel):

    def __init__(self,
                 LayerNo=9
                 ):

        super(DeRefNet, self).__init__()

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
        self.TDDA = TDDA()

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
        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)
        
        x = x
        
        final, index, aligned_imgs = self.TDDA(x)
        
        if mode == 'tensor':
            return final, aligned_imgs
        elif mode == 'predict':
            final = torch.cat(final, dim=0)
            return [final, image_name, targets_GT, x, index]
        elif mode == 'loss':
            # Feature extraction loss
            loss_constraint = torch.mean(torch.pow(layers_sym[0], 2))
            for k in range(self.LayerNo-1):
                loss_constraint += torch.mean(torch.pow(layers_sym[k + 1], 2))
            # Expand the aligned support frame to 5 dimensions and calculate the alignment loss
            loss_pix_lq_all = 0
            for i in range(len(index)):
                m = index[i]
                lq_ref = x[m, :].expand(aligned_imgs[m].size(0)-1, -1)  # 支持帧扩展为5维
                alignments = torch.cat((aligned_imgs[m][:2], aligned_imgs[m][3:]), dim=0)
                loss_pix_lq = torch.sum(torch.abs(alignments-lq_ref))
                loss_pix_lq_all += loss_pix_lq
            # Regression Loss
            loss_fg_all = 0
            for i in range(len(index)):
                n = index[i]
                loss_fg = self.fg_loss(final[i], batch_x[n])
                loss_fg_all += loss_fg
            # Calculate the total loss by weight
            alpha = torch.Tensor([0.1]).to(device)
            gamma = torch.Tensor([0.1]).to(device)
            loss_all = torch.mul(gamma, loss_constraint)+ loss_fg_all + torch.mul(alpha, loss_pix_lq_all) 

            return {'loss': loss_all, 
                'loss_constraint': torch.mul(gamma, loss_constraint),
                'loss_align': torch.mul(alpha, loss_pix_lq_all),
                'loss_reg': loss_fg_all,
                }
