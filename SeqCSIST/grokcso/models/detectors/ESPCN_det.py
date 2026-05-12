import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmagic.registry import MODELS
from mmengine.registry import MODELS
import scipy.io as sio
from tools.utils import read_targets_from_xml_list
from grokcso.models.backbones.ESPCN_bb import ESPCNNet, MSEloss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@MODELS.register_module()
class ESPCN(BaseModel):

    def __init__(self
                 ):

        super(ESPCN, self).__init__()

        self.ESPCNNet = ESPCNNet(upscale_factor = 3)

        #self.lq_pixel_loss = MSEloss(alpha=0.9)
        self.fg_loss = MSEloss(alpha=1)

    def forward(self, **kwargs):
        mode = kwargs['mode']
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
        
        x = Phix.view(-1, 1, 11, 11)
        x = self.ESPCNNet(x)
        
        final = x.view(-1, 1089)
        
        if mode == 'tensor':
            return final, aligned_imgs
        elif mode == 'predict':
            return [final[2:18], image_name, targets_GT]
        elif mode == 'loss':
            # 回归损失
            loss_fg_all = 0
            loss_fg_all = self.fg_loss(final[2:18], batch_x[2:18])
            loss_all = loss_fg_all 

            return {'loss': loss_all
                }
