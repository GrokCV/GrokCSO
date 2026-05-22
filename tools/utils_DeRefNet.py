from scipy.integrate import dblquad
import numpy as np
import cv2
import os
import math
from torch import nn as nn
import torch
from gen_annotation import read_bounding_boxes_from_xml
from torch.nn.modules.utils import _pair, _single
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import scipy.io as sio


import glob
import scipy.io as sio
import xml.etree.cElementTree as ET
import json
import matplotlib.pyplot as plt

# 计算像元的幅度响应
def diffusion(x, y, target_x, target_y, ai, sigma):
  """扩散函数使用高斯函数"""
  return ai * (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - target_x) ** 2
                                + (y - target_y) ** 2) / (2 * sigma ** 2))


# 计算像元的幅度响应
def calculate_pixel_response(pixel_x, pixel_y, target_info, sigma):
  """计算像元灰度值"""
  response = 0.0
  for target in target_info:
    target_x, target_y, ai = target
    response += dblquad(diffusion, pixel_x - 1 / 2, pixel_x + 1 / 2,
                        lambda y: pixel_y - 1 / 2, lambda y: pixel_y + 1 / 2,
                        args=(target_x, target_y, ai, sigma))[0]
  return response


# 读取 xml 文件，将其转换为矩阵
def xml_2_matrix_single(xml_file):
  targets_GT, *_ = read_bounding_boxes_from_xml(xml_file)
  A = np.zeros((33, 33))
  for i in range(len(targets_GT)):
    x, y, lightness = targets_GT[i][0], targets_GT[i][1], targets_GT[i][2]
    A[int(round(3 * x + 1, 0)), int(round(3 * y + 1, 0))] = lightness
  return A

def read_targets_from_xml(xml_file_path):
  """解析 XML 文件，获取单张图片的所有信息"""
  tree = ET.parse(xml_file_path)
  root = tree.getroot()
  targets_GT = []
  for object_info in root.findall('object'):
    target_info = object_info.find('coordinate')
    if target_info is not None:
      x_c = float(target_info.find('xc').text)
      y_c = float(target_info.find('yc').text)
      brightness = float(target_info.find('brightness').text)
      targets_GT.append([x_c, y_c, brightness])
  return targets_GT

from PIL import Image

def show_contrast(gt, pred, batch_idx, idx, img_name, name, c=3):

  gt_image = gt
  image_3 = pred.cpu().numpy()
  titles = ["Target", "GT Image", f"CS={c} Image"]
  
  save_dir_pred = os.path.join("pngs", "ISTA_Net_pp")
  if not os.path.exists(save_dir_pred):
        os.makedirs(save_dir_pred)
  # save_dir_gt = os.path.join("pngs","GT")
  # if not os.path.exists(save_dir_gt):
  #       os.makedirs(save_dir_gt)
  # save_dir_Phix = os.path.join("pngs","Phix")
  # if not os.path.exists(save_dir_Phix):
  #       os.makedirs(save_dir_Phix)
  # 绘制和显示图像
  # 创建图像绘制环境
  plt.figure()  # 创建一个8x4英寸大小的图像窗口
  
  image_path = os.path.join('SeqCSIST/data/track_5000_20/test/image', name)
  img = Image.open(image_path)
  Phix_image = np.array(img)
  # # # 绘制第一张图像
  # plt.figure()
  # plt.imshow(Phix_image, cmap='gray')
  # plt.axis('off')
  # plt.savefig(os.path.join(save_dir_Phix,f"Phix_{idx+2}.png"), bbox_inches='tight', pad_inches=0, dpi=300)
  # plt.close()  # 关闭图像窗口，释放内存
  # # plt.subplot(131)  # 子图1
  # # plt.imshow(origin_image, cmap='gray')
  # # plt.title(titles[0])

  # # # 绘制第二张图像
  # # plt.subplot(132)  # 子图1
  # plt.figure()
  # plt.imshow(gt_image, cmap='gray')
  # plt.axis('off')
  # plt.savefig(os.path.join(save_dir_gt,f"GT_{idx+2}.png"), bbox_inches='tight', pad_inches=0, dpi=300)
  # plt.close()  # 关闭图像窗口，释放内存
  # # plt.title(titles[1])

  # 绘制第三张图像
  # plt.subplot(133)  # 子图2
  plt.figure()
  plt.imshow(image_3, cmap='gray')
  plt.axis('off')
  # plt.title(titles[2])

  plt.savefig(os.path.join(save_dir_pred,f"{img_name}_{idx+2}.png"), bbox_inches='tight', pad_inches=0, dpi=300)
  plt.close()  # 关闭图像窗口，释放内存

val_xml_root = '/opt/data/private/Simon/DeRef_Net/data/val/annotation'

def read_targets_from_xml_list(xml_file_path_list):
  batch_anns = []
  for xml_file_path in xml_file_path_list:
    xml_file_path = os.path.join(val_xml_root, xml_file_path)
    batch_anns.append(read_targets_from_xml(xml_file_path))
  return batch_anns

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, : target.size(2), : target.size(3)]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_cond(sigma, a, type):
    para_sigma = None
    para_noise = a / 5.0
    if type == 'org':
        para_sigma = sigma * 2.0 / 100.0
    elif type == 'org_sigma':
        para_sigma = sigma / 100.0

    para_sigma_np = np.array([para_sigma])

    para_sigma = torch.from_numpy(para_sigma_np).type(torch.FloatTensor)

    para_sigma = para_sigma.to(device)

    para_noise_np = np.array([para_noise])
    para_noise = torch.from_numpy(para_noise_np).type(torch.FloatTensor)

    para_noise = para_noise.to(device)
    para_sigma = para_sigma.view(1, 1)
    para_noise = para_noise.view(1, 1)
    para = torch.cat((para_sigma, para_noise), 1)


    return para

# # 读取 xml_root 路径下的所有 xml 文件，将其转换为矩阵
# def xml_2_matrix(xml_root):
#   x = []
#   for xml_file in os.listdir(xml_root):
#     print(xml_file)
#     A = xml_2_matrix_single(os.path.join(xml_root, xml_file))
#     x.append(A.reshape(1, 1089))
#   return x


# # 矩阵初始化
# def initialization(initial_matrix_root):
#   Qinit_Name = initial_matrix_root
#   # Computing Initialization Matrix:
#   if os.path.exists(Qinit_Name):
#       print("----------------Qinit 存在 --------------------------")
#   else:
#       # 读取矩阵 phi
#       Phi_data_Name = '/opt/data/private/xubohan/data/image_dan/image_dan/train/a_phi_0_3.mat'
#       Phi_data = sio.loadmat(Phi_data_Name)
#       Phi_input = Phi_data['phi']

#       # 读取读取 xml 文件并将训练数据转换为矩阵形式
#       Training_labels = xml_2_matrix("/opt/data/private/xubohan/data/image_duo/train/annotation")

#       # 计算初始化矩阵，根据最小二乘法的公式，Qinit = X * Y^T * (Y * Y^T)^(-1)
#       X_data = np.squeeze(Training_labels).T
#       print(X_data.shape)
#       Y_data = np.dot(Phi_input, X_data)
#       Y_YT = np.dot(Y_data, Y_data.transpose())
#       X_YT = np.dot(X_data, Y_data.transpose())
#       # 矩阵求逆
#       Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
#       # 删除临时变量，节省内存
#       del X_data, Y_data, X_YT, Y_YT
#       # 保存初始化矩阵
#       sio.savemat(Qinit_Name, {'Qinit': Qinit})
#       print("generate done")


# #生成初始值
# Qinit_Name = '../data/initial_matrix/Q_3.mat'
# initialization(Qinit_Name)


# # 将训练集数据转换
# Training_labels = xml_2_matrix("/opt/data/private/xubohan/data/image_duo/train/annotation")
# X_data = np.squeeze(Training_labels)
# file_path = '/opt/data/private/xubohan/data/image_duo/train/train.mat'
# sio.savemat(file_path, {'matrices': X_data})


# # 计算像元的幅度响应
# def diffusion(x, y, target_x, target_y, ai, sigma):
#   """扩散函数使用高斯函数"""
#   return ai * (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - target_x) ** 2
#                                 + (y - target_y) ** 2) / (2 * sigma ** 2))


# # 计算像元的幅度响应
# def calculate_pixel_response(pixel_x, pixel_y, target_info, sigma):
#   """计算像元灰度值"""
#   response = 0.0
#   for target in target_info:
#     target_x, target_y, ai = target
#     response += dblquad(diffusion, pixel_x - 1 / 2, pixel_x + 1 / 2,
#                         lambda y: pixel_y - 1 / 2, lambda y: pixel_y + 1 / 2,
#                         args=(target_x, target_y, ai, sigma))[0]
#   return response


# # 将image保存
# def save_image(k, image, location="data/test_image_folder/cso_img"):
#   image_output_location = os.path.join(location, f"image_{k}.png")
#   cv2.imwrite(image_output_location, image)


# def joint_matrix(image_list):
#   # 按照你想要的排列方式，将小矩阵拼接成一个大矩阵
#   result_matrix = np.zeros((27, 36))
#   for i in range(3):
#       for j in range(4):
#           small_matrix = image_list[i * 4 + j]
#           result_matrix[i * 9:i * 9 + 9, j * 9:j * 9 + 9] = small_matrix
#   return result_matrix

# def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
#     def one_scale(output, target, sparse):

#         b, _, h, w = output.size()

#         if sparse:
#             target_scaled = sparse_max_pool(target, (h, w))
#         else:
#             target_scaled = F.interpolate(target, (h, w), mode="area")
#         return EPE(output, target_scaled, sparse, mean=False)

#     if type(network_output) not in [tuple, list]:
#         network_output = [network_output]
#     if weights is None:
#         weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
#     assert len(weights) == len(network_output)

#     loss = 0
#     for output, weight in zip(network_output, weights):
#         loss += weight * one_scale(output, target_flow, sparse)
#     return loss

# def EPE(input_flow, target_flow, sparse=False, mean=True):
#     EPE_map = torch.norm(target_flow - input_flow, 2, 1)
#     batch_size = EPE_map.size(0)
#     if sparse:
#         # invalid flow is defined with both flow coordinates to be exactly 0
#         mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

#         EPE_map = EPE_map[~mask]
#     if mean:
#         return EPE_map.mean()
#     else:
#         return EPE_map.sum() / batch_size

# # 将 image 保存
# def save_image(k, image, location="data/test_image_folder/cso_img"):
#   image_output_location = os.path.join(location, f"image_{k}.png")
#   cv2.imwrite(image_output_location, image)


# def joint_matrix(image_list):
#   # 按照你想要的排列方式，将小矩阵拼接成一个大矩阵
#   result_matrix = np.zeros((27, 36))
#   for i in range(3):
#       for j in range(4):
#           small_matrix = image_list[i * 4 + j]
#           result_matrix[i * 9:i * 9 + 9, j * 9:j * 9 + 9] = small_matrix
#   return result_matrix


# # 生成带噪声的图像
# def create_image_with_noise(width, height, target_info, sigma, noise_mean=10,
#                             noise_std=5):
#   # 生成带噪声的图片
#   # 初始化图像
#   image = np.zeros((width, height))
#   for xi in range(0, width):
#     for yi in range(0, height):
#       # 计算每个像元的响应并累加到图像中
#       # if random_y + 4 > xi > random_y - 4 and random_x + 4 > yi > random_x - 4:
#       pixel_response = calculate_pixel_response(xi, yi, target_info, sigma)
#       noise = np.random.normal(noise_mean, noise_std)
#       image[yi, xi] = pixel_response + noise
#   return image


# # 生成无噪声的图像
# def create_image(width, height, target_info, sigma):
#   # 初始化图像，构造一个 w x h 大小的图片，图片中有点目标，需要知道成像函数
#   image0 = np.zeros((width, height))
#   for xi in range(0, width):
#     for yi in range(0, height):
#       # 计算每个像元的响应并累加到图像中
#       pixel_response = calculate_pixel_response(xi, yi, target_info, sigma)
#       image0[yi, xi] = pixel_response
#   return image0