import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import cv2
import os
from scipy.linalg import norm
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
from utils import xml_2_matrix_single
import sys
import xml.etree.cElementTree as ET
import glob
import json
import shutil
import time

# 1. 生成 PSF 函数（使用高斯核）
def generate_psf(psf_size, sigma):
    x = np.linspace(-psf_size // 2, psf_size // 2, psf_size)
    y = np.linspace(-psf_size // 2, psf_size // 2, psf_size)
    x, y = np.meshgrid(x, y)
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= np.sum(psf)  # 归一化 PSF
    return psf

# 2. BID 反卷积算法
def bid_deconvolution(I_lr, PSF, max_iter, alpha, device):
    # 将图像和 PSF 移动到 GPU
    I_lr = torch.tensor(I_lr, dtype=torch.float32, device=device)
    PSF = torch.tensor(PSF, dtype=torch.float32, device=device)

    # 初始化高分辨率图像为 33x33
    I_hr = F.interpolate(I_lr.unsqueeze(0).unsqueeze(0), size=(33, 33), mode='bilinear', align_corners=False).squeeze()

    # 迭代反卷积过程
    for k in range(max_iter):
        # 使用卷积模拟观测图像
        I_sim = F.conv2d(I_hr.unsqueeze(0).unsqueeze(0), PSF.unsqueeze(0).unsqueeze(0), padding='same')
    
        # 将低分辨率图像上采样到 33x33
        I_lr_upsampled = F.interpolate(I_lr.unsqueeze(0).unsqueeze(0), size=(33, 33), mode='bilinear', align_corners=False).squeeze()
    
        # 在高分辨率域计算误差
        I_dev = I_sim.squeeze() - I_lr_upsampled

        # 更新高分辨率图像估计
        I_hr -= alpha * I_dev

        # 判断收敛
        if torch.max(torch.abs(I_dev)) < 0.1:
            #print(f'Converged at iteration: {k + 1}')
            break


    return I_hr.cpu().numpy()

# 3. 单张图片处理函数
def process_single_image(I_lr, PSF, max_iter, alpha, device):
    # 将输入图像调整为 11x11 大小
    I_lr = cv2.resize(I_lr, (11, 11), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # 调用 BID 反卷积函数
    I_hr = bid_deconvolution(I_lr, PSF, max_iter, alpha, device)

    return I_hr

# 4. BID 参数设置和处理函数调用
def run_bid_deconvolution(I_lr, psf_size, sigma, max_iter, alpha, device):
    """
    运行 BID 反卷积过程，返回单张高分辨率图像

    Parameters:
    - I_lr: 输入图像（低分辨率图像）
    - psf_size: PSF 的大小
    - sigma: 高斯核的标准差
    - max_iter: 迭代次数
    - alpha: 更新步长
    - device: 计算设备 ('cuda' 或 'cpu')

    Returns:
    - I_hr: 恢复后的高分辨率图像
    """
    # 生成高斯 PSF
    PSF = generate_psf(psf_size, sigma)

    # 处理单张低分辨率图像，返回高分辨率图像
    I_hr = process_single_image(I_lr, PSF, max_iter, alpha, device)
    return I_hr


def error(msg):
    print(msg)
    sys.exit(0)


def calculate_distance(xc, yc, xp, yp):
  distance = (xc - xp) ** 2 + (yc - yp) ** 2
  return distance


# 计算ap，核心代码，只要输入查准率和查全率的列表，就能计算ap
def voc_ap(rec, prec):
    rec.insert(0, 0.0)  # insert 0.0 at beginning of list
    rec.append(1.0)  # insert 1.0 at end of list
    m_rec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at beginning of list
    prec.append(0.0)  # insert 0.0 at end of list
    m_pre = prec[:]
    for i in range(len(m_pre)-2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i+1])
    i_list = []
    for i in range(1, len(m_rec)):
        if m_rec[i] != m_rec[i-1]:
            i_list.append(i)  # if it was matlab would be i + 1
    ap = 0.0
    for i in i_list:
        ap += ((m_rec[i]-m_rec[i-1])*m_pre[i])
    return ap, m_rec, m_pre


# 将文件按行存储到列表中
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def read_targets_from_xml(xml_file_path):
  """解析XML文件，获取单张图片的所有信息"""
  tree = ET.parse(xml_file_path)
  root = tree.getroot()
  targets_GT = []
  for object_info in root.findall('object'):
    target_info = object_info.find('coordinate')
    if target_info is not None:
      x_c = float(target_info.find('xc').text)
      y_c = float(target_info.find('yc').text)
      # brightness = float(target_info.find('brightness').text)
      targets_GT.append([x_c, y_c, 0])
  return targets_GT, len(targets_GT)

total_time = 0
count = 0
temp_time = 0
temp_count = 0

psf_size = 3
sigma = 0.5
max_iter = 1000
alpha = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(0, 15000):
  image = np.array(Image.open(
    f'/opt/data/private/Simon/DeRefNet/data/track_5000_20/val/image/image_{i}.png'
    ))
  img_name = f'image_{i}.png'
  result_dir = os.path.join("work_dir/outputs", "CSO_BID")
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  y = image.reshape(121, 1)
  start = time.time()
  x_output = run_bid_deconvolution(y, psf_size, sigma, max_iter, alpha, device)
  end = time.time()
  total_time += end - start
  temp_time += end - start
  count += 1
  temp_count += 1
  if temp_count == 100:
    print(f"第{count}次运行BID time平均时间:", temp_time / temp_count)
    temp_time = 0
    temp_count = 0

  x_output = np.array(x_output)

  brightness_threshold = 70
  # 将x_output中小于70的值置为0
  x_output[x_output < brightness_threshold] = 0
  # 对x_output每一行处理

  matrix = x_output.reshape(33, 33)
  # 找出非零值所在的行号和列号
  non_zero_indices = np.nonzero(matrix)
  txt_name = img_name[:-len(".png")] + ".txt"
  txt_path = os.path.join(result_dir, txt_name)
  with open(txt_path, "w") as f:
    for i in range(len(non_zero_indices[0])):
      f.write(str((non_zero_indices[0][i] - 1) / 3) + " " + str(
        (non_zero_indices[1][i] - 1) / 3) + " " +
              str(matrix[
                    non_zero_indices[0][i], non_zero_indices[1][i]]) + "\n")

print("average time:", total_time / count)

def compute():
    GT_PATH = "/opt/data/private/Simon/DeRefNet/data/track_5000_20/val/annotation"
    DR_PATH = os.path.join("work_dir/outputs", "CSO_BID")

    cso_mAP = 0
    for i in range(5):
      MAX_DISTANCE = (i + 1) * 0.05
      """
       Create a ".temp_files/" and "output/" directory
      """
      TEMP_FILES_PATH = ".temp_files"
      if not os.path.exists(TEMP_FILES_PATH):  # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)

      # 将gt中以xml格式存储的文件地址存储到列表中
      ground_truth_files_list = glob.glob(GT_PATH + '/*.xml')
      if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
      ground_truth_files_list.sort()

      total_gt_counter = 0

      gt_files = []
      for txt_file in ground_truth_files_list:
        # 提取文件名中的数字
        file_id = txt_file.split(".xml", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # 去除文件前面COS三个字母
        file_id = file_id[3:]

        bounding_boxes, box_nums = read_targets_from_xml(txt_file)
        total_gt_counter += box_nums

        # dump bounding_boxes into a ".json" file
        new_temp_file = TEMP_FILES_PATH + "/" + "image_" + file_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
          json.dump(bounding_boxes, outfile)

      # get a list with the detection-results files
      dr_files_list = glob.glob(DR_PATH + '/*.txt')
      dr_files_list.sort()

      bounding_boxes = []
      for txt_file in dr_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        lines = file_lines_to_list(txt_file)
        for line in lines:
          xc, yc, brightness = line.split()
          bounding_boxes.append({"brightness": brightness, "file_id": file_id,
                                 "xc": xc, "yc": yc})

      bounding_boxes.sort(key=lambda x: float(x['brightness']), reverse=True)
      with open(TEMP_FILES_PATH + "/CSO_dr.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

      """
       Load detection-results of that class
      """
      dr_file = TEMP_FILES_PATH + "/CSO_dr.json"
      dr_data = json.load(open(dr_file))

      """
       Assign detection-results to ground-truth objects
      """
      nd = len(dr_data)

      tp = [0] * nd  # creates an array of zeros of size nd
      fp = [0] * nd

      count_true_positives = {}
      count_true_positives["CSO_targets"] = 0

      # 枚举所有预测框
      for idx, detection in enumerate(dr_data):
        file_id = detection["file_id"]

        gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
        ground_truth_data = json.load(open(gt_file))
        min_distance = 100000
        gt_match = -1
        x_P, y_P = detection["xc"], detection["yc"]
        # 对每一个预测框，找到其所在文件中与之类别相同的框，计算其与预测框的iou，找到最大的iou
        for obj in ground_truth_data:
          # look for a class_name match
          # obj的内容是[xc, yc, 0]
          xc, yc = obj[0], obj[1]
          distance = calculate_distance(xc, yc, float(x_P), float(y_P))
          if distance < min_distance:
            min_distance = distance
            gt_match = obj

        max_overlap = MAX_DISTANCE

        if min_distance <= (max_overlap ** 2):
          if not gt_match[2]:
            # true positive
            tp[idx] = 1
            gt_match[2] = 1
            count_true_positives["CSO_targets"] += 1
            # update the ".json" file
            with open(gt_file, 'w') as f:
              f.write(json.dumps(ground_truth_data))

          else:
            # false positive (multiple detection)
            fp[idx] = 1
        else:
          # false positive
          fp[idx] = 1

      cumsum = 0
      for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
      cumsum = 0
      for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

      rec = tp[:]
      for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / total_gt_counter

      prec = tp[:]
      for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

      ap, mrec, mprec = voc_ap(rec[:], prec[:])

      text = "{0:.2f}%".format(ap * 100) + " = " + "csoAP " + f"threshold(" \
                                                              f"{MAX_DISTANCE})"

      cso_mAP += ap * 100

      print(text)

      if i == 4:
        # dump(results, self.out_file_path)
        print(
          "cso_mAP: " + str(cso_mAP / 5) + "\n",)

    shutil.rmtree(TEMP_FILES_PATH)
    shutil.rmtree(DR_PATH)

compute()