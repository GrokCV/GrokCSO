"""红外空间邻近目标仿真"""
from myutils import save_image, create_image, create_image_with_noise
import random
from 生成注解文件 import save_image_infos
import os


def generate_points_with_min_distance(x, y, n, min_distance):
  """在像素（x,y）中生成n个点，这些点之间的最小距离为min_distance"""

  step = 0
  points = []
  image_infos = []
  while len(points) < n:
    step += 1
    max_brightness = random.randint(220, 255)
    # 生成随机的点在像素（x,y）中
    new_x = random.uniform(x - 0.4, x + 0.6)
    new_y = random.uniform(y - 0.4, y + 0.6)

    # 检查新生成的点与已有点之间的最小距离
    valid = all(
      ((new_x - px) ** 2 + (new_y - py) ** 2 >= min_distance ** 2) for px,
                                                        py, _ in points)

    if valid:
      points.append((new_x, new_y, max_brightness))
      image_infos.append({"xc": f"{new_x}", "yc": f"{new_y}",
                          "brightness": f"{max_brightness}"})

    if step > 100000:
      n = max(0, n - 1)

  return points, image_infos


def generate_target_number(mu=3, sigma_t=1):
  """生成mu为均值，sigma_t为标准差的高斯分布的随机数"""

  random_float = random.gauss(mu, sigma_t)

  # 将随机浮点数四舍五入到最接近的整数
  random_integer = round(random_float)

  # 确保生成的整数在1到5的范围内
  random_integer = min(5, max(1, random_integer))
  print("目标数量：", random_integer)

  return random_integer


def generate_dataset(dataset_size, image_size=11, min_distance=0.6,
    location="data/CSO_11_100_10/32CSO_img",noise=False, minmax=False):

  for k in range(dataset_size):
    print(f"第{k}张图像：")

    # 生成sigma
    sigma = 0.5


    # 限定取点的位置
    min_x = 2
    max_x = 8
    min_y = 2
    max_y = 8

    random_x = random.randint(min_x, max_x)
    random_y = random.randint(min_y, max_y)
    print("随机像素：", random_x, random_y)

    # 生成目标个数
    random_integer = generate_target_number()

    # 生成目标信息（坐标和亮度）
    target_info, image_infos = generate_points_with_min_distance(random_x, random_y,
                                                    random_integer, min_distance)

    # 生成图片
    if noise:
      image = create_image_with_noise(image_size, image_size, target_info, sigma)
    else:
      image = create_image(image_size, image_size, target_info, sigma)

    image_location = location + "/CSO_img"
    if not os.path.exists(image_location):
      os.makedirs(image_location)
    save_image(k, image, image_location)

    xml_folder_location = location + "/Annotations"
    if not os.path.exists(xml_folder_location):
      os.makedirs(xml_folder_location)
    # 存储图片信息
    save_image_infos(image_infos, k, image_size, image_size,
                     xml_folder_location=xml_folder_location)


if __name__ == '__main__':
  '''
      生成大小为10000的数据集，最小距离为0.5，图像大小为11*11
  '''
  dataset_size = 10000
  image_size = 11
  min_distance = 0.5

  # 生成测试集
  location3 = "../data/ISTA-Net-master/CSO_11_10000"
  if not os.path.exists(location3):  # if it doesn't exist already
    os.makedirs(location3)
  generate_dataset(dataset_size, image_size=image_size,
                   min_distance=min_distance, location=location3, noise=False,
                   minmax=False)





