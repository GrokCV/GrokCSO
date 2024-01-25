from scipy.integrate import dblquad
import numpy as np
import cv2
import os

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


# 将image保存
def save_image(k, image, location="data/test_image_folder/CSO_img"):
  image_output_location = os.path.join(location, f"image_{k}.png")
  cv2.imwrite(image_output_location, image)


def joint_matrix(image_list):
  # 按照你想要的排列方式，将小矩阵拼接成一个大矩阵
  result_matrix = np.zeros((27, 36))
  for i in range(3):
      for j in range(4):
          small_matrix = image_list[i * 4 + j]
          result_matrix[i * 9:i * 9 + 9, j * 9:j * 9 + 9] = small_matrix
  return result_matrix


# 生成带噪声的图像
def create_image_with_noise(width, height, target_info, sigma, noise_mean=10,
                            noise_std=5):
  # 生成带噪声的图片
  # 初始化图像
  image = np.zeros((width, height))
  for xi in range(0, width):
    for yi in range(0, height):
      # 计算每个像元的响应并累加到图像中
      # if random_y + 4 > xi > random_y - 4 and random_x + 4 > yi > random_x - 4:
      pixel_response = calculate_pixel_response(xi, yi, target_info, sigma)
      noise = np.random.normal(noise_mean, noise_std)
      image[yi, xi] = pixel_response + noise
  return image


# 生成无噪声的图像
def create_image(width, height, target_info, sigma):
  # 初始化图像，构造一个w x h大小的图片，图片中有点目标，需要知道成像函数
  image0 = np.zeros((width, height))
  for xi in range(0, width):
    for yi in range(0, height):
      # 计算每个像元的响应并累加到图像中
      pixel_response = calculate_pixel_response(xi, yi, target_info, sigma)
      image0[yi, xi] = pixel_response
  return image0
