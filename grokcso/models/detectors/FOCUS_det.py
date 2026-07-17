import torch
import torch.nn as nn
import random
from mmengine.model import BaseModel
from mmagic.registry import MODELS
from mmengine.registry import MODELS
import scipy.io as sio
from tools.utils import read_targets_from_xml_list
from grokcso.models.backbones.FOCUS_bb import FOCUS_bb, MSEloss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@MODELS.register_module()
class FOCUS(BaseModel):

    def __init__(self,
                 ):

        super(FOCUS, self).__init__()

        self.FOCUS = FOCUS_bb(upscale_factor = 3)

        #self.lq_pixel_loss = MSEloss(alpha=0.9)
        self.fg_loss = MSEloss(alpha=1)

    def forward(self, **kwargs):
        mode = kwargs['mode']
        if mode == 'loss':
            batch = torch.stack(kwargs["batch_x"])
            batch_x = batch.squeeze(dim=1)
            Phi_x = torch.stack(kwargs["gt_img_11"])
            Phix = Phi_x.squeeze(dim=1)
            
            noise_std = random.uniform(0.001, 0.05)      # 噪声标准差范围
            noise = torch.randn_like(Phix) * noise_std
            
            input_x = Phix + noise   # 有噪声

        elif mode == 'predict':
            image_name = []
            Phi_x = torch.stack(kwargs["gt_img_11"])
            Phix = Phi_x.squeeze(dim=1)
            ann_paths = kwargs["ann_path"]
            targets_GT = read_targets_from_xml_list(ann_paths)
            image_name = kwargs["image_name"]
                        
            input_x = Phix     # 无噪声
            
            
        else:
            print("Invalid mode:", mode)
            return None
        
        x = input_x.view(-1, 1, 11, 11)
        x, out1, out1_act= self.FOCUS(x)
        
        final = x.view(-1, 1089)
        
        if mode == 'tensor':
            return final
        elif mode == 'predict':
            return [final, image_name, targets_GT]
        elif mode == 'loss':
            # 回归损失
            loss_fg_all = 0
            loss_fg_all = self.fg_loss(final, batch_x)
            # 能量守恒的约束
            energy_HR = torch.sum(final, dim=1)
            energy_LR = torch.sum(Phix, dim=1)
            loss_energy = torch.mean(torch.abs(energy_HR - energy_LR))
            # 稀疏约束
            loss_sparsity = torch.mean(torch.abs(final))

            return {'loss_fg_all': loss_fg_all,
                    'loss_energy': 0.1*loss_energy,
                    'loss_sparsity': 0.01*loss_sparsity
                }
