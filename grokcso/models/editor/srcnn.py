import torch
import torch.nn as nn
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from typing import Dict, Union
from tools.utils import read_targets_from_xml_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@MODELS.register_module()
class SRCNN(BaseModel):
  def __init__(self, upscale_factor=3):
    super(SRCNN, self).__init__()
    self.upscale_factor = upscale_factor
    self.img_upsampler = nn.Upsample(scale_factor=self.upscale_factor,
                                     mode="bicubic", align_corners=False)
    self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=3 // 2)
    self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=3 // 2)
    self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=3 // 2)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:
      mode = kwargs['mode']
      if mode == 'tensor':
        Phix = kwargs["Phix"]
      else:
        Phix = torch.stack(kwargs["gt_img_11"])
        Input_image = torch.stack(kwargs["gt_img_11"])
        Phix = Phix.squeeze(dim=1).to(device)

        batch_x = torch.stack(kwargs["gt"]).squeeze(dim=1).to(device)
        ann_paths = kwargs["ann_path"]
        file_id = kwargs["file_id"]

      x = Phix.view(-1, 1, 11, 11)

      x = self.img_upsampler(x)
      x = self.relu(self.conv1(x))
      x = self.relu(self.conv2(x))
      x = self.conv3(x)

      x_final = x.view(-1, 11*11*self.upscale_factor*self.upscale_factor)

      if mode == 'tensor':
        return x_final
      elif mode == 'predict':
        targets_GT = read_targets_from_xml_list(ann_paths)
        return [{"x_final": x_final,
                "targets_GT": targets_GT,
                "file_id": file_id,
                "Input_image": Input_image,
                 "img_gt": batch_x}]
      elif mode == 'loss':
        loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))

        return {'loss_discrepancy': loss_discrepancy}



