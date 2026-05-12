import scipy.io as sio
from utils import calculate_pixel_response
import time


def gen_target_info(sub_pixel_size, c) -> list:
  """
  create target coordinates, (x, y, intensity)
  :param sub_pixel_size: sub-pixel size
  :param c: compression ratio
  :return: target information list
  """
  target_info = []
  for i in range(sub_pixel_size):
    for j in range(sub_pixel_size):
      target_info.append([float(1.0 * (i - (c - 1) // 2) / c),
                          float(1.0 * (j - (c - 1) // 2) / c),
                          1])
  return target_info


def gen_A(sub_pixel_size, c, target_info, sigma) -> list:
  """
  create low resolution imaging matrix
  :param sub_pixel_size: sub-pixel size
  :param c: compression ratio
  :param target_info: target information list
  :param sigma: Gaussian function variance
  :return: low resolution imaging matrix
  """
  A = []
  for xi in range(0, sub_pixel_size // c):
    for yi in range(0, sub_pixel_size // c):
      B = []
      for target in target_info:
        target_infos = []
        target_infos.append(target)
        pixel_response = calculate_pixel_response(yi, xi, target_infos, sigma)
        B.append(pixel_response)
      A.append(B)
  return A


if __name__ == '__main__':
  c = 5
  sub_pixel_size = 11 * c  # 33
  sigma = 0.15
  phi = gen_A(sub_pixel_size, c, gen_target_info(sub_pixel_size, c), sigma)
  file_path = f'data/sampling_matrix/phi_{c}_{sigma}.mat'
  sio.savemat(file_path, {'phi': phi})

