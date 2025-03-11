from collections import OrderedDict
from typing import List, Optional
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS
from ..functional import eval_map2


@METRICS.register_module()
class CSO_Metrics(BaseMetric):
    """
      The CSO_Metrics class is used to calculate the metrics of the CSO dataset.

      This class evaluates the performance of the model in the super-resolution
      task by calculating the average precision (AP) at different distance_thr
      distance thresholds.

      Attributes:
          default_prefix (str): The default prefix for this metric,
                       used for identification.
          distance_thrs (list): distance_thrs A threshold list representing
                       different distance thresholds used during evaluation.
          brightness_threshold (float): Brightness threshold, predicted points
                       below this value will be filtered.
          c (int): Pixel division ratio
          collect_device (str): Specify the device (CPU or GPU) used to collect data。
          prefix (str, optional): Prefix name, optional
    """
    default_prefix: Optional[str] = 'cso_metric'

    def __init__(self,
                 distance_thrs=(0.05, 0.1, 0.15, 0.2, 0.25),
                 brightness_threshold: float = 50,
                 c=3,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.distance_thrs = list(distance_thrs)
        self.brightness_threshold = brightness_threshold
        self.c = c

    def process(self, data_batch: dict, outputs) -> None:
        """
        preds:The output result is the output result under c super-resolution multiples, which needs to be converted to the position in the 11 * 11 image.
        And compare it with the annotation information to calculate the index.
        The position (i, j) in the super-resolution ratio c corresponds to the position in the 11 * 11 image
        ((i - c // 2)/c, (j - c // 2)/c)

        Args:
            data_batch (dict): A batch of data samples, obtained from the dataloader.
            outputs: A batch of prediction results, including `x_output` and annotation information `ann_list`.
                - x_output: torch.Tensor, (N, 1089)
                - ann_list: list, (N, K, 3)
                    - N: batch size
                    - K: Number of targets
                    - 3: (x, y, brightness)
        """
        output = outputs[0]
        x_output = output['x_final'].cpu().numpy()
        ann_list = output['targets_GT']
        x_output[x_output < self.brightness_threshold] = 0

        for idx in range(x_output.shape[0]):
          ann = ann_list[idx]

          dets = []
          matrix = x_output[idx].reshape(11 * self.c, 11 * self.c)

          non_zero_indices = np.nonzero(matrix)

          for i in range(len(non_zero_indices[0])):
            row = non_zero_indices[0][i]
            col = non_zero_indices[1][i]
            value = matrix[row, col]
            dets.append([float(1.0 * (row - (self.c-1)//2) / self.c),
                         float(1.0 * (col - (self.c-1)//2) / self.c),
                        value])

          self.results.append((ann, dets))

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)
        eval_results = OrderedDict()

        mean_aps = []
        for distance_thr in self.distance_thrs:
            logger.info(f'\n{"-" * 15}distance_thr: {distance_thr}{"-" * 15}')
            mean_ap, _ = eval_map2(
                preds,
                gts,
                distance_thr=distance_thr,
                logger=logger,
                )
            mean_aps.append(mean_ap)
            eval_results[f'AP{int(distance_thr * 100):02d}'] = round(mean_ap, 5)

        print('eval_results:', eval_results)

        eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        eval_results.move_to_end('mAP', last=False)
        return eval_results

