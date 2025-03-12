from scipy.integrate import dblquad
import numpy as np
import cv2

import scipy.io as sio
import xml.etree.cElementTree as ET
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


# Calculate the magnitude response of a pixel
def diffusion(x, y, target_x, target_y, ai, sigma):
  """The diffusion function uses a Gaussian function"""
  return ai * (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - target_x) ** 2
                                + (y - target_y) ** 2) / (2 * sigma ** 2))


# Calculate the magnitude response of a pixel
def calculate_pixel_response(pixel_x, pixel_y, target_info, sigma):
  """Calculate pixel gray value"""
  response = 0.0
  for target in target_info:
    target_x, target_y, ai = target
    response += dblquad(diffusion, pixel_x - 1 / 2, pixel_x + 1 / 2,
                        lambda y: pixel_y - 1 / 2, lambda y: pixel_y + 1 / 2,
                        args=(target_x, target_y, ai, sigma))[0]
  return response


def save_image(k, image, location="data/test_image_folder/cso_img"):
  image_output_location = os.path.join(location, f"image_{k}.png")
  cv2.imwrite(image_output_location, image)


def save_image1(k, image, location="data/test_image_folder/cso_img"):
  import matplotlib.pyplot as plt
  image_output_location = os.path.join(location, f"image_{k}.pdf")
  plt.imshow(image, cmap='gray')
  plt.savefig(image_output_location, bbox_inches='tight', pad_inches=0,
              dpi=600)


def create_image(width, height, target_info, sigma, noise_mean=35000, noise_std=5):
  image0 = np.zeros((width, height))
  for xi in range(0, width):
    for yi in range(0, height):
      pixel_response = calculate_pixel_response(xi, yi, target_info, sigma)
      image0[yi, xi] = pixel_response
  if noise_mean == 0 and noise_std == 0:
    return image0
  else:
    background = \
      np.random.normal(loc=noise_mean, scale=noise_std, size=(11, 11))
    background = np.clip(background, 34985, 35015)
    image0 = (image0 + background)
    return image0


def xml_2_matrix_single(xml_file, c=3, is_mat=False):
  """
    If the magnification is c, the corresponding coordinate relationship is
          （i, j） -> ( c * i + (c - 1) // 2, c * j + (c - 1) // 2)
  """
  targets_GT, *_ = read_bounding_boxes_from_xml(xml_file)
  A = np.zeros((11 * c, 11 * c))
  for i in range(len(targets_GT)):
    x, y, lightness = targets_GT[i][0], targets_GT[i][1], targets_GT[i][2]
    if is_mat:
      x = x - 1
      y = y - 1
    A[int(round(c * x + (c - 1) // 2, 0)),
      int(round(c * y + (c - 1) // 2, 0))] = lightness
  return A


def xml_2_matrix(xml_root, c=3):
  """
  Convert all XML files in the XML folder into a large matrix with a shape of
  (n, 11 * c * 11 * c)
  """
  x = []
  count = 0
  for xml_file in os.listdir(xml_root):
    A = xml_2_matrix_single(os.path.join(xml_root, xml_file), c)
    x.append(A.reshape(1, 11 * 11 * c * c))
    count += 1
  print(f"共有{count}个训练样本")
  return x


def initialization(initial_matrix_root,
                   Phi_data_root="data/sampling_matrix/a_phi_0_3.mat",
                   label_matrix_root="data/train_mat.mat"):
  """
  create the initialization matrix
  :param initial_matrix_root: the path of the initialization matrix
  :param Phi_data_root: the path of the sampling matrix
  :param label_matrix_root: the path of the label matrix
  :param c: the magnification
  :return:
  """

  Qinit_Name = initial_matrix_root
  # Computing Initialization Matrix:
  if os.path.exists(Qinit_Name):
      print("-------Qinit exits---------")
  else:
      Phi_data = sio.loadmat(Phi_data_root)
      Phi_input = Phi_data['phi']

      Training_labels = sio.loadmat(label_matrix_root)
      Training_labels = Training_labels['matrices']

      # Qinit = X * Y^T * (Y * Y^T)^(-1)
      X_data = np.squeeze(Training_labels).T
      print(X_data.shape)
      Y_data = np.dot(Phi_input, X_data)
      Y_YT = np.dot(Y_data, Y_data.transpose())
      X_YT = np.dot(X_data, Y_data.transpose())
      Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
      del X_data, Y_data, X_YT, Y_YT
      sio.savemat(Qinit_Name, {'Qinit': Qinit})
      print("generate done")


def read_targets_from_xml(xml_file_path, is_mat=False):
  tree = ET.parse(xml_file_path)
  root = tree.getroot()
  targets_GT = []
  for object_info in root.findall('object'):
    target_info = object_info.find('coordinate')
    if target_info is not None:
      x_c = float(target_info.find('xc').text)
      y_c = float(target_info.find('yc').text)
      if is_mat:
        x_c = x_c - 1
        y_c = y_c - 1
      brightness = float(target_info.find('brightness').text)
      targets_GT.append([x_c, y_c, brightness])
  return targets_GT


def read_targets_from_xml_list(xml_file_path_list, is_mat=False):
  batch_anns = []
  for xml_file_path in xml_file_path_list:
    batch_anns.append(read_targets_from_xml(xml_file_path, is_mat))
  return batch_anns


def xml_to_C_position(xml_file_path, c=3):
  tree = ET.parse(xml_file_path)
  root = tree.getroot()
  targets_GT = []
  for object_info in root.findall('object'):
    target_info = object_info.find('coordinate')
    if target_info is not None:
      x_c = float(target_info.find('xc').text)
      y_c = float(target_info.find('yc').text)
      brightness = float(target_info.find('brightness').text)
      targets_GT.append([x_c * c + (c - 1) // 2,
                         y_c * c + (c - 1) // 2,
                         brightness])
  return targets_GT


# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points

def save_image_infos(image_infos, image_id, w, h, sigma=0.5, depth="1",
                     xml_folder_location="Annotations"):
  root = ET.Element("annotation")
  image_name = "CSO" + f"{image_id}"

  file_name = ET.SubElement(root, "filename")
  file_name.text = image_name
  size = ET.SubElement(root, "size")
  width = ET.SubElement(size, "width")
  width.text = f"{w}"
  height = ET.SubElement(size, "height")
  height.text = f"{h}"
  height = ET.SubElement(size, "depth")
  height.text = depth
  argument = ET.SubElement(root, "sigma")
  argument.text = f"{sigma}"

  for image_info in image_infos:
    object_info = ET.SubElement(root, "object")
    name = ET.SubElement(object_info, "name")
    name.text = "Target"
    coordinate = ET.SubElement(object_info, "coordinate")
    xc = ET.SubElement(coordinate, "xc")
    xc.text = image_info["xc"]
    yc = ET.SubElement(coordinate, "yc")
    yc.text = image_info["yc"]
    brightness = ET.SubElement(coordinate, "brightness")
    brightness.text = image_info["brightness"]
  xml_filename = os.path.join(xml_folder_location, image_name + f".xml")
  xml_str = ET.tostring(root, encoding="utf-8")
  dom = minidom.parseString(xml_str)
  pretty_xml_str = dom.toprettyxml()

  with open(xml_filename, "w", encoding="utf-8") as xml_file:
    xml_file.write(pretty_xml_str)


def read_bounding_boxes_from_xml(xml_file_path):
  """parse xml file and return bounding boxes"""
  tree = ET.parse(xml_file_path)
  root = tree.getroot()
  targets_GT = []
  sigma = float(root.find('sigma').text)
  width = float(root.find('size').find('width').text)
  height = float(root.find('size').find('height').text)
  for obj in root.findall('object'):
    target_info = obj.find('coordinate')
    if target_info is not None:
      xc = float(target_info.find('xc').text)
      yc = float(target_info.find('yc').text)
      brightness = float(target_info.find('brightness').text)
      targets_GT.append([xc, yc, brightness])

  return targets_GT, sigma, width, height


def xml_path_2_matrix(xml_path, c=3):
  """Convert XML file to C coordinates"""
  tree = ET.parse(xml_path)
  root = tree.getroot()
  A = np.zeros((11 * c, 11 * c))
  count = 0
  for object_info in root.findall('object'):
    target_info = object_info.find('coordinate')
    if target_info is not None:
      count += 1
      x_c = float(target_info.find('xc').text)
      y_c = float(target_info.find('yc').text)
      brightness = float(target_info.find('brightness').text)
      A[int(round(c * x_c + (c - 1) // 2, 0)),
        int(round(c * y_c + (c - 1) // 2, 0))] = brightness
  return A, count-1





