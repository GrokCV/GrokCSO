import torch
import torch.nn as nn
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from typing import Dict, Union
from tools.utils import read_targets_from_xml_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@MODELS.register_module()
class SRCNN(BaseModel):  # 搭建SRCNN 3层卷积模型，Conve2d（输入层数，输出层数，卷积核大小，步长，填充层）
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

        # 获取gt图像
        batch_x = torch.stack(kwargs["gt"]).squeeze(dim=1).to(device)
        # 获取标注路径
        ann_paths = kwargs["ann_path"]
        # 记录文件ID
        file_id = kwargs["file_id"]

      x = Phix.view(-1, 1, 11, 11)

      x = self.img_upsampler(x)
      x = self.relu(self.conv1(x))
      x = self.relu(self.conv2(x))
      x = self.conv3(x)

      x_final = x.view(-1, 11*11*self.upscale_factor*self.upscale_factor)  # 最终网络输出

      # 根据不同模式返回相应结果
      if mode == 'tensor':
        return x_final   # 返回最终网络输出 todo: 对网络输出结果进行后处理操作，类似nms，使得输出结果更加准确
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
        loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))  # 计算平方误差损失

        return {'loss_discrepancy': loss_discrepancy}

#   def forward(self, input):
#
#       # 获取真实图像
#       Phix = input
#
#       x = Phix.view(-1, 1, 11, 11)
#
#       x = self.img_upsampler(x)
#       x = self.relu(self.conv1(x))
#       x = self.relu(self.conv2(x))
#       x = self.conv3(x)
#
#       x_final = x.view(-1, 11*11*self.upscale_factor*self.upscale_factor)  # 最终网络输出
#
#       return x_final
#
#
# if __name__ == '__main__':
#   from mmengine.analysis import get_model_complexity_info
#   from mmengine.analysis.print_helper import _format_size
#
#   model = SRCNN(3)
#   model = model.to(device)
#
#   outputs = get_model_complexity_info(model, input_shape=(64, 121),
#                                            show_table=True, show_arch=True)
#   flops, params = _format_size(outputs['flops']), _format_size(outputs['params'])
#
#   print("flops:{}".format(flops))
#   print("params:{}".format(params))



