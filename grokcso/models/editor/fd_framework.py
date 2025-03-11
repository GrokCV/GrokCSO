import torch
import torch.nn as nn
from mmengine.registry import MODELS
import scipy.io as sio
from mmengine.model import BaseModel
from typing import Dict, Union
from grokcso.models.blocks import *
from tools.utils import read_targets_from_xml_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@MODELS.register_module()
class FDFrameWork(BaseModel):
    """
     FDFrameWork:Used to implement an unfolding network architecture based on multi-layer iteration.

     Args:
         LayerNo (int): The number of layers in the network, each layer is built using the specified block.
         Phi_data_Name (str): Path to a .mat file containing the Phi matrix, which is the input transformation matrix of the network.
         Qinit_Name (str): Path to a .mat file containing the Qinit matrix, which is the initial weight matrix of the network.
         block (str): The name of the basic module for building the network. The default is "DIST_BasicBlock".
         lambda_weight (float): The weight of the two branches of the dynamic feature extraction module is 0.5 by default.
         c (int): The pixel division multiple, the default is 3.

     Attributes:
         Phi (Tensor): The Phi matrix loaded by reading the Phi_data_Name file is used for linear transformation of the data.
         Qinit (Tensor): The Qinit matrix loaded by reading the Qinit_Name file is used as the initial weight matrix.
         fcs (ModuleList): A list of network layers containing LayerNo basic modules.
     """
    def __init__(self,
                 LayerNo,
                 Phi_data_Name,
                 Qinit_Name,
                 block="DIST_BasicBlock",
                 lambda_weight=0.5,
                 c=3):
        super(FDFrameWork, self).__init__()

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
        # Phix = Phix - self.noise_level

        batch_x = torch.stack(kwargs["gt"]).squeeze(dim=1).to(device)
        ann_paths = kwargs["ann_path"]
        file_id = kwargs["file_id"]

      PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
      PhiTb = torch.mm(Phix, Phi)

      x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

      layers_sym = []
      intermediate = [x]

      for i in range(self.LayerNo):
        [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
        layers_sym.append(layer_sym)
        intermediate.append(x)

      x_final = x

      if mode == 'tensor':
        return [x_final, intermediate]
      elif mode == 'predict':
        targets_GT = read_targets_from_xml_list(ann_paths)
        return [{"x_final": x_final,
                "targets_GT": targets_GT,
                "file_id": file_id,
                "Input_image": Input_image,
                 "img_gt": batch_x}]
      elif mode == 'loss':
        loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))
        loss_constraint = torch.mean(torch.pow(layers_sym[0], 2))
        for k in range(self.LayerNo - 1):
          loss_constraint += torch.mean(torch.pow(layers_sym[k + 1], 2))
        gamma = torch.Tensor([0.01]).to(device)

        return {'loss_discrepancy': loss_discrepancy, '0.01 * loss_constraint': torch.mul(gamma, loss_constraint)}
