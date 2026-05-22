import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmagic.registry import MODELS
from mmengine.registry import MODELS
import scipy.io as sio
import numpy as np
from tools.utils import read_targets_from_xml_list
from grokcso.models.backbones.USRNet_bb import USRNet_bb, MSEloss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian_kernel(kernel_size=3, sigma=0.5):
    # 生成一个高斯核
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    kernel = kernel / np.sum(kernel)  # 归一化
    return torch.tensor(kernel, dtype=torch.float32)

k = gaussian_kernel(kernel_size=3, sigma=0.5)
k = k.view(1, 1, 3, 3)

k = k.expand(20, -1, -1, -1)  # 将k的批次维度扩展为与x相同


@MODELS.register_module()
class USRNet(BaseModel):

    def __init__(self
                 ):

        super(USRNet, self).__init__()

        self.USRNet_bb = USRNet_bb()
        self.k = k

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
        sigma = torch.tensor(0.0).to(device)
        x = Phix.view(-1, 1, 11, 11)
        x = self.USRNet_bb(x, self.k, sf = 3, sigma=sigma)
        
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
