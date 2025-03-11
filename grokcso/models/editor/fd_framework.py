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
     FDFrameWork类用于实现一种基于多层迭代的unfolding网络架构。

     Args:
         LayerNo (int): 网络的层数，每一层都使用指定的block构建。
         Phi_data_Name (str): 包含Phi矩阵的.mat文件路径，Phi是网络的输入变换矩阵。
         Qinit_Name (str): 包含Qinit矩阵的.mat文件路径，Qinit是网络初始权重矩阵。
         block (str): 构建网络的基础模块名称，默认为"DIST_BasicBlock"。
         lambda_weight (float): 动态特征提取模块两路分支的比重，默认为0.5。
         c (int): 像素划分的倍数，默认为3。

     Attributes:
         Phi (Tensor): 通过读取Phi_data_Name文件加载的Phi矩阵，用于数据的线性变换。
         Qinit (Tensor): 通过读取Qinit_Name文件加载的Qinit矩阵，作为初始权重矩阵。
         fcs (ModuleList): 包含LayerNo个基础模块的网络层列表。
     """
    def __init__(self,
                 LayerNo,
                 Phi_data_Name,
                 Qinit_Name,
                 block="DIST_BasicBlock",
                 lambda_weight=0.5,
                 c=3):
        super(FDFrameWork, self).__init__()

        # 加载Phi矩阵数据并转换为Tensor类型
        Phi_data = sio.loadmat(Phi_data_Name)
        Phi = Phi_data['phi']
        self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

        # 加载Qinit矩阵数据并转换为Tensor类型
        Qinit_data = sio.loadmat(Qinit_Name)
        Qinit = Qinit_data['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

        # 获取指定block类型的类定义
        module_dict = globals()
        BasicBlock = module_dict[block]

        # 构建包含LayerNo层的模块列表，每一层使用相同的BasicBlock模块
        onelayer = []
        self.LayerNo = LayerNo
        for i in range(LayerNo):
            onelayer.append(BasicBlock(lambda_weight=lambda_weight, c=c))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:
      """
              前向传播函数，根据不同模式（loss或predict）处理输入数据并返回结果。

              Args:
                  kwargs (dict): 输入参数，包括以下几种模式：
                      - 'mode': 表示网络的执行模式，值可以是 'loss', 'predict' 或 'tensor'。
                      - 其他参数根据模式不同而变化，如：
                        - 'gt_img_33': 用于loss模式下的真实图像。
                        - 'gt_img_11': 用于predict模式下的输入图像。
                        - 'ann_path': 用于predict模式下的标注路径。
                        - 'file_id': 用于predict模式下的文件ID。

              Returns:
                  Union[Dict[str, torch.Tensor], list]: 根据模式不同返回相应结果。
                      - 'loss' 模式下返回损失字典，包括 'loss_discrepancy' 和 'loss_constraint'。
                      - 'predict' 模式下返回包含预测结果和相关信息的列表。
                      - 'tensor' 模式下返回最终的网络输出。
      """
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

        # 获取gt图像
        batch_x = torch.stack(kwargs["gt"]).squeeze(dim=1).to(device)
        # 获取标注路径
        ann_paths = kwargs["ann_path"]
        # 记录文件ID
        file_id = kwargs["file_id"]

      # 计算Phi矩阵相关的中间结果
      PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)  # 计算PhiTPhi = Phi' * Phi
      PhiTb = torch.mm(Phix, Phi)  # 计算PhiTb = Phix * Phi

      # 初始化网络的输入x
      x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

      # 用于存储对称损失
      layers_sym = []
      intermediate = [x]

      # 逐层执行网络的前向传播
      for i in range(self.LayerNo):
        [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)  # 执行第i层网络
        layers_sym.append(layer_sym)  # 存储每一层的对称损失
        intermediate.append(x)  # 存储每一层的输出结果

      x_final = x  # 最终网络输出

      # 根据不同模式返回相应结果
      if mode == 'tensor':
        return [x_final, intermediate]
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
        loss_constraint = torch.mean(torch.pow(layers_sym[0], 2))  # 计算第一层的对称损失
        for k in range(self.LayerNo - 1):
          loss_constraint += torch.mean(torch.pow(layers_sym[k + 1], 2))  # 累加每一层的对称损失
        gamma = torch.Tensor([0.01]).to(device)  # 设置权重系数

        # 损失值为对称损失和约束损失的加权和，返回损失字典
        return {'loss_discrepancy': loss_discrepancy, '0.01 * loss_constraint': torch.mul(gamma, loss_constraint)}
