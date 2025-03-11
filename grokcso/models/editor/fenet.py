import torch.nn as nn
import torch
import torch.nn.functional as F
from grokcso.models.tricks import UpsampleBlock, BasicConv2d, FEB
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from tools.utils import read_targets_from_xml_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@MODELS.register_module()
class FENet(BaseModel):
  """
  title: Frequency-Based Enhancement Network for Efficient Super-Resolution
  paper: https://ieeexplore.ieee.org/document/9778017

  code: https://github.com/pbehjatii/FENet-PyTorch
  """

  def __init__(self,
               scale=3,
               lst_channels=1,
               group=4,
               n_blocks=12
               ):
    super(FENet, self).__init__()

    wn = lambda x: torch.nn.utils.weight_norm(x)
    self.n_blocks = n_blocks
    self.scale = scale

    self.entry_1 = wn(nn.Conv2d(lst_channels, 64, 3, 1, 1))

    body = [FEB(wn, 64, 64) for _ in range(self.n_blocks)]
    self.body = nn.Sequential(*body)
    self.reduction = BasicConv2d(wn, 64 * 13, 64, 1, 1, 0)

    self.upscample = UpsampleBlock(64, scale=scale, multi_scale=False, wn=wn,
                                   group=group)
    self.exit = wn(nn.Conv2d(64, lst_channels, 3, 1, 1))

    # self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

  def forward(self, **kwargs):
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

    res = x
    x = self.entry_1(x)

    c0 = x
    out_blocks = []

    out_blocks.append(c0)

    for i in range(self.n_blocks):
      x = self.body[i](x)
      out_blocks.append(x)

    output = self.reduction(torch.cat(out_blocks, 1))

    output = output + x

    output = self.upscample(output, scale=self.scale)
    output = self.exit(output)

    skip = F.interpolate(
      res, (x.size(-2) * self.scale, x.size(-1) * self.scale), mode="bicubic",
      align_corners=False
    )

    out = skip + output
    x_final = out.view(-1, 11*11*self.scale*self.scale)
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
      loss_discrepancy = torch.mean(
        torch.pow(x_final - batch_x, 2))

      return {'loss_discrepancy': loss_discrepancy}
