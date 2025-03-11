from typing import List
from mmengine.dataset import BaseDataset
from mmengine.registry import DATASETS
import numpy as np
import os
from PIL import Image
import torch
import xml.etree.cElementTree as ET


@DATASETS.register_module()
class NoiseDataset(BaseDataset):

  METAINFO = dict(dataset_type='CSO_Noise_Dataset', task_name="SR")

  def __init__(self, data_root, length, c=3, is_mat=False):
    self.cso_data_root = data_root
    self.length = length
    self.c = c
    self.is_mat = is_mat

    super().__init__(metainfo=self.METAINFO)

  def load_data_list(self) -> List[dict]:
    """
    :return: list[dict]: A list of annotation.
    include gt_img_11 and xml_data
    """
    # image path
    cso_img_root = os.path.join(self.cso_data_root, "cso_img")
    # label path
    ann_root = os.path.join(self.cso_data_root, "Annotations")

    image_files = os.listdir(cso_img_root)
    data_list = []
    for i in range(self.length):
      data_info = {}
      img_path = os.path.join(cso_img_root, image_files[i])
      gt_img_11 = Image.open(img_path)
      gt_img_11 = np.array(gt_img_11)

      gt_img_11 = torch.Tensor(gt_img_11)
      gt_img_11 = gt_img_11.view(1, 121)
      data_info["gt_img_11"] = gt_img_11

      image_name = image_files[i]
      file_id = image_name[len("image_"):-len(".png")]
      xml_path = os.path.join(ann_root, "CSO" + file_id + ".xml")
      data_info["ann_path"] = xml_path
      data_info["file_id"] = file_id

      gt, count = self.xml_path_2_matrix(xml_path)
      gt = torch.Tensor(gt)
      gt = gt.view(1, 11 * 11 * self.c * self.c)
      data_info["gt"] = gt
      data_info["count"] = count

      data_list.append(data_info)

    return data_list

  def xml_path_2_matrix(self, xml_path):
    """
    Project the target information into the C times sub-pixel division space
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    A = np.zeros((11 * self.c, 11 * self.c))
    count = 0
    for object_info in root.findall('object'):
      target_info = object_info.find('coordinate')
      if target_info is not None:
        count += 1
        x_c = float(target_info.find('xc').text)
        y_c = float(target_info.find('yc').text)
        if self.is_mat:
          x_c = x_c - 1
          y_c = y_c - 1
        brightness = float(target_info.find('brightness').text)
        # 将亮度值赋给A
        A[int(round(self.c * x_c + (self.c - 1) // 2, 0)),
          int(round(self.c * y_c + (self.c - 1) // 2, 0))] = brightness
    return A, count-1
