import copy
from collections import OrderedDict
from typing import List, Optional

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS
from mmengine.logging import print_log


@METRICS.register_module()
class Similarity(BaseMetric):

  default_prefix: Optional[str] = 'cso_metric'

  def __init__(self,
               c=3,
               collect_device: str = 'cpu',
               prefix: Optional[str] = None) -> None:
    super().__init__(collect_device=collect_device, prefix=prefix)
    self.c = c
    self.psnr = lambda x, y: 10 * np.log10(255**2 / np.mean((x - y) ** 2))
    self.ssim = lambda x, y: 1 - np.mean((x - y) ** 2) / (np.mean(x ** 2) * np.mean(y ** 2))

  def process(self, data_batch: dict, outputs) -> None:
    output = outputs[0]
    x_output = output['x_final'].cpu().numpy()  # 获取网络输出，并转为 numpy 格式
    ann_list = output['img_gt'].cpu().numpy()  # 获取真实标注
    for idx in range(x_output.shape[0]):
      pred_img = x_output[idx].reshape(11*self.c, 11*self.c)
      gt_img = ann_list[idx].reshape(11*self.c, 11*self.c)
      # 计算psnr和ssim
      psnr = self.psnr(pred_img, gt_img)
      ssim = self.ssim(pred_img, gt_img)
      self.results.append([psnr, ssim])

  def compute_metrics(self, results: list) -> dict:
    logger: MMLogger = MMLogger.get_current_instance()
    psnrs = [x[0] for x in results]
    ssims = [x[1] for x in results]
    eval_results = OrderedDict()
    eval_results['PSNR'] = np.mean(psnrs)
    eval_results['SSIM'] = np.mean(ssims)
    # 平均精度
    print_log(f'Avg_PSNR: {eval_results["PSNR"]:.4f}--------Avg_SSIM:'
              f'{eval_results["SSIM"]:.4f}', logger)
    return eval_results



