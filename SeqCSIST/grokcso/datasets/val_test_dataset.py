from torch.utils.data import Dataset
from mmengine.registry import DATASETS
import numpy as np
import re
import os
from PIL import Image
import torch
from mmengine.registry import DATASETS

@DATASETS.register_module()
class Val_Test_Dataset(Dataset):
    def __init__(self, data_dir, length, xml_root):
        self.data_dir = data_dir
        self.image_files = sorted(os.listdir(data_dir), key=Val_Test_Dataset.extract_number)
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
        gt_img_11 = Image.open(img_path)
        gt_img_11 = np.array(gt_img_11)
        gt_img_11 = gt_img_11.astype(np.float32)
        # 将图像 reshape 为 1*121 的矩阵，并转换为 tensor
        gt_img_11 = torch.Tensor(gt_img_11).float()
        gt_img_11 = gt_img_11.view(1, 121)

        return {"gt_img_11": gt_img_11, 
                "image_name": self.image_files[idx], 
                "ann_path":xml_path
                }
