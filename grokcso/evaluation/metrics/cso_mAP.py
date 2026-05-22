# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional

import os
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS
from grokcso.evaluation.functional.mean_mAP import eval_map

@METRICS.register_module()
class CSO_mAP(BaseMetric):

    default_prefix: Optional[str] = 'cso_metric'

    def __init__(self,
                 iou_thrs=(0.05, 0.1, 0.15, 0.2, 0.25),
                 brightness_threshold: float = 70,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # iou_thrs是一个列表
        self.iou_thrs = list(iou_thrs)
        self.brightness_threshold = brightness_threshold

    def process(self, data_batch: dict, outputs) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            outputs : A batch of data samples that
                contain annotations and x_output.
                x_output: torch.Tensor, shape (N, 1089)
                ann_list: list , shape (N, K, 3)
                N: batch size
                K: number of objects
                3: (x, y, brightness)
        """

        x_output = outputs[0].cpu().numpy()
        ann_list = outputs[2]
        # 将小于亮度阈值的值设置为0
        x_output[x_output < self.brightness_threshold] = 0

        for idx in range(x_output.shape[0]):
            # 一个样本的标注信息
            # ann = dict(targets=ann_list[idx+2])       
            ann = ann_list[idx+2]   # 3帧+1，5帧+2，7帧+3

            dets = []
            matrix = x_output[idx].reshape(33, 33)

            non_zero_indices = np.nonzero(matrix)
            # if non_zero_indices.shape[0] > 0:
            for i in range(len(non_zero_indices[0])):
                row = non_zero_indices[0][i]
                col = non_zero_indices[1][i]
                value = matrix[row, col]
                dets.append([(row - 1) / 3,
                            (col - 1) / 3,
                            value])
        
            # result_dir = 'SeqCSIST/work_dirs/temp'
            # if not os.path.exists(result_dir):
            #         os.makedirs(result_dir)
            # txt_name = img_name[idx+2][:-len(".png")] + ".txt"
            # txt_path = os.path.join(result_dir, txt_name)
            # with open(txt_path, "w") as f:
            #     for i in range(len(non_zero_indices[0])):
            #         f.write(str((non_zero_indices[0][i] - 1) / 3) + " " + str(
            #         (non_zero_indices[1][i] - 1) / 3) + " " +
            #         str(matrix[non_zero_indices[0][i], non_zero_indices[1][i]]) + "\n")
            self.results.append((ann, dets))

    def compute_metrics(self, results: list) -> dict:

        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)
        eval_results = OrderedDict()

        mean_aps = []
        for iou_thr in self.iou_thrs:
            logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            mean_ap, _ = eval_map(
                preds,
                gts,
                iou_thr=iou_thr,
                logger=logger,
                )
            mean_aps.append(mean_ap)
            eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

        print('eval_results:', eval_results)

        # 计算mAP
        eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        # 将字典中mAP放在第一个位置
        eval_results.move_to_end('mAP', last=False)
        return eval_results