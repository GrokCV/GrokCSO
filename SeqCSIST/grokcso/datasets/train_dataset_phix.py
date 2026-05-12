from torch.utils.data import Dataset
# from mmdet.datasets.builder import DATASETS
from mmengine.registry import DATASETS
import numpy as np
import re
import os
from PIL import Image
import torch
from mmengine.registry import DATASETS
from tools.utils import xml_2_matrix_single


@DATASETS.register_module()
class TrainDataset_Phix(Dataset):
    def __init__(self, data_root, length, xml_root):
        self.data_dir = data_root
        self.image_files = sorted(os.listdir(data_root), key=TrainDataset_Phix.extract_number)
        self.ntest = length
        self.xml_root = xml_root
    
    @staticmethod
    def extract_number(filename):
        # 使用正则表达式在文件名中查找数字
        match = re.search(r'\d+', filename)
    
        # 如果找到了数字，提取并返回该数字
        # 否则，返回 -1
        return int(match.group()) if match else -1
  
    def __len__(self):

        return self.ntest

    def __getitem__(self, idx):
        ann_root = self.xml_root
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        file_id = self.image_files[idx][len("image_"):-len(".png")]
        xml_path = os.path.join(ann_root, "COS" + file_id + ".xml")
        
        batch_x = xml_2_matrix_single(xml_path)
        batch_x = torch.tensor(batch_x, dtype=torch.float32).view(-1,1089)

        gt_img_11 = Image.open(img_path)
        gt_img_11 = np.array(gt_img_11)
        gt_img_11 = gt_img_11.astype(np.float32)
        # 将图像 reshape 为 1*121 的矩阵，并转换为 tensor
        gt_img_11 = torch.Tensor(gt_img_11).float()
        gt_img_11 = gt_img_11.view(1, 121)


        return {"gt_img_11": gt_img_11, 
                "batch_x":batch_x
                }