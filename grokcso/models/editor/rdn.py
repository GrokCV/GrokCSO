import torch
import torch.nn as nn
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from typing import Dict, Union
from tools.utils import read_targets_from_xml_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


@MODELS.register_module()
class RDN(BaseModel):
    def __init__(self,
                 scale_factor=3,
                 num_channels=1,
                 num_features=64,
                 GO=64,
                 G=64,
                 D=16,
                 C=8
                 ):
        super(RDN, self).__init__()

        self.scale_factor = scale_factor
        self.G0 = GO
        self.G = G
        self.D = D
        self.C = C

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

        self.criteria = nn.L1Loss()

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:

      mode = kwargs['mode']
      if mode == 'tensor':
        Phix = kwargs["Phix"]
      else:
        Phix = torch.stack(kwargs["gt_img_11"])
        Input_image = torch.stack(kwargs["gt_img_11"])
        Phix = Phix.squeeze(dim=1).to(device)

        # 获取gt图像
        batch_x = torch.stack(kwargs["gt"]).squeeze(dim=1).to(device)
        # 获取标注路径
        ann_paths = kwargs["ann_path"]
        # 记录文件ID
        file_id = kwargs["file_id"]

      x = Phix.view(-1, 1, 11, 11)

      sfe1 = self.sfe1(x)
      sfe2 = self.sfe2(sfe1)

      x = sfe2
      local_features = []
      for i in range(self.D):
          x = self.rdbs[i](x)
          local_features.append(x)

      x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
      x = self.upscale(x)
      x = self.output(x)
      x_final = x.view(-1, 11*11*self.scale_factor*self.scale_factor)

      # 根据不同模式返回相应结果
      if mode == 'tensor':
        return x_final  # 返回最终网络输出 todo: 对网络输出结果进行后处理操作，类似nms，使得输出结果更加准确
      elif mode == 'predict':
        # 读取目标标注信息并返回预测结果
        targets_GT = read_targets_from_xml_list(ann_paths)
        return [{"x_final": x_final,
                 "targets_GT": targets_GT,
                 "file_id": file_id,
                 "Input_image": Input_image,
                 "img_gt": batch_x}]
      elif mode == 'loss':
        # 计算损失函数
        loss_l1 = self.criteria(x_final, batch_x)
        loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))

        return {'loss_discrepancy': loss_discrepancy}


if __name__ == '__main__':
  from mmengine.analysis import get_model_complexity_info
  from mmengine.analysis.print_helper import _format_size

  model = RDN(3)
  model = model.to(device)

  outputs = get_model_complexity_info(model, input_shape=(64, 121),
                                           show_table=True, show_arch=True)
  flops, params = _format_size(outputs['flops']), _format_size(outputs['params'])

  print("flops:{}".format(flops))
  print("params:{}".format(params))