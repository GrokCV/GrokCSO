import torch
import torch.nn as nn
from mmengine.registry import MODELS
import scipy.io as sio
from mmengine.model import BaseModel
from typing import Dict, Union
from grokcso.models.blocks import *
from tools.utils import read_targets_from_xml_list, xml_2_matrix_single
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@MODELS.register_module()
class DISTA(BaseModel):
    def __init__(self,
                 LayerNo,
                 Phi_data_Name,
                 Qinit_Name,
                 block="DIST_BasicBlock",
                 lambda_weight=0.5,
                 c=3,
                 noise_level=0,
                 is_mat=False
                 ):
        super(DISTA, self).__init__()
        self.noise_level = noise_level
        self.c = c
        self.is_mat = is_mat

        Phi_data = sio.loadmat(Phi_data_Name)
        Phi = Phi_data['phi']
        self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

        Qinit_data = sio.loadmat(Qinit_Name)
        Qinit = Qinit_data['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

        module_dict = globals()
        BasicBlock = module_dict[block]

        onelayer = []
        self.LayerNo = LayerNo
        for i in range(LayerNo):
          onelayer.append(BasicBlock(lambda_weight=lambda_weight, c=c))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:
      mode = kwargs['mode']
      Phi = self.Phi
      Qinit = self.Qinit

      if mode == 'tensor':
        Phix = kwargs["Phix"]
      else:
        Phix = torch.stack(kwargs["gt_img_11"])
        Input_image = torch.stack(kwargs["gt_img_11"])
        Phix = Phix.squeeze(dim=1).to(device)
        batch_x = torch.stack(kwargs["gt"])
        batch_x = batch_x.squeeze(dim=1).to(device)

        ann_paths = kwargs["ann_path"]
        file_id = kwargs["file_id"]

        count = kwargs["count"]
      # Phix = Phix - self.noise_level

      PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
      PhiTb = torch.mm(Phix, Phi)

      x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

      layers_sym = []  # for computing symmetric loss

      intermediate = [x]

      for i in range(self.LayerNo):
        [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
        intermediate.append(x)
        layers_sym.append(layer_sym)

      x_final = x

      if mode == 'tensor':
        return x_final
      elif mode == 'predict':
        targets_GT = read_targets_from_xml_list(ann_paths, self.is_mat)
        return [{"x_final": x_final,
                "targets_GT": targets_GT,
                "file_id": file_id,
                "Input_image": Input_image,
                "img_gt": batch_x,
                 "count": count}]
      elif mode == 'loss':
        loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))
        loss_constraint = torch.mean(torch.pow(layers_sym[0], 2))
        for k in range(self.LayerNo - 1):
          loss_constraint += torch.mean(torch.pow(layers_sym[k + 1], 2))
        gamma = torch.Tensor([0.01]).to(device) # 0.01 when c=3
        return {'loss_discrepancy': loss_discrepancy,
                'loss_constraint': torch.mul(gamma, loss_constraint)}






