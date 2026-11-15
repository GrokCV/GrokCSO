from collections import OrderedDict
import math
from typing import Optional

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS

from ..functional import eval_map2


def format_sig4(x):
    """Format to 4 significant figures"""
    if x == 0:
        return 0.0
    if abs(x) >= 1:
        return float(f'{x:.4g}')
    x_abs = abs(x)
    exponent = int(math.floor(math.log10(x_abs)))
    decimals = -exponent + 3
    return float(f'{x:.{decimals}f}')


@METRICS.register_module()
class CSO_Metrics(BaseMetric):

    default_prefix: Optional[str] = 'cso_metric'

    def __init__(self,
                 distance_thrs=(0.05, 0.1, 0.15, 0.2, 0.25),
                 brightness_threshold: float = 50,
                 c=3,
                 postprocess: bool = False,
                 nms_threshold: float = 0.4,
                 pd: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 use_loc_sort: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.distance_thrs = list(distance_thrs)
        self.brightness_threshold = brightness_threshold
        self.c = c
        self.postprocess = postprocess
        self.nms_threshold = nms_threshold
        self.pd = pd
        self.use_loc_sort = use_loc_sort

    def process(self, data_batch: dict, outputs) -> None:
        output = outputs[0]
        x_output = output['x_final'].cpu().numpy()
        ann_list = output['targets_GT']

        if self.postprocess:
            if output.get('x_num') is not None:
                count_logits = output['x_num']
                if isinstance(count_logits, torch.Tensor):
                    count_logits = count_logits.cpu().numpy()
                count_list = np.argmax(count_logits, axis=1) + 1
            else:
                gt_counts = data_batch['count_1']
                if isinstance(gt_counts, torch.Tensor):
                    count_list = gt_counts.cpu().numpy()
                elif isinstance(gt_counts, list):
                    count_list = np.array(gt_counts)

        use_offset = 'offset_pred' in output and output['offset_pred'] is not None
        offset_pred = output['offset_pred'].cpu().numpy() if use_offset else None

        x_output[x_output < self.brightness_threshold] = 0

        for idx in range(x_output.shape[0]):
            ann = ann_list[idx]
            gt_count = len(ann)
            dets = []
            matrix = x_output[idx].reshape(11 * self.c, 11 * self.c)
            non_zero_indices = np.nonzero(matrix)

            for i in range(len(non_zero_indices[0])):
                row = non_zero_indices[0][i]
                col = non_zero_indices[1][i]
                value = matrix[row, col]

                if use_offset:
                    d_row = offset_pred[idx, 0, row, col]
                    d_col = offset_pred[idx, 1, row, col]

                    sort_score = value
                    refined_row = row + d_row
                    refined_col = col + d_col
                else:
                    sort_score = value
                    refined_row = row
                    refined_col = col

                coord_row = float(1.0 * (refined_row - (self.c - 1) // 2) / self.c)
                coord_col = float(1.0 * (refined_col - (self.c - 1) // 2) / self.c)
                dets.append([coord_row, coord_col, value, sort_score])

            dets = sorted(dets, key=lambda x: x[3], reverse=True)

            if self.postprocess:
                dets_tmp = np.array([[d[0], d[1], d[2]] for d in dets])
                best_dets = []
                while len(dets_tmp) > 0:
                    best_dets.append(dets_tmp[0])
                    if len(dets_tmp) == 1:
                        break
                    distances = np.sqrt(np.sum((dets_tmp[1:, :2] - dets_tmp[0, :2]) ** 2, axis=1))
                    indices = np.where(distances < self.nms_threshold)[0] + 1
                    dets_tmp = np.delete(dets_tmp, indices, axis=0)
                    dets_tmp = np.delete(dets_tmp, 0, axis=0)
                dets = best_dets

                count_val = int(count_list[idx])
                if len(dets) > count_val:
                    dets = dets[:count_val]

                final_dets = dets
            else:
                final_dets = [[d[0], d[1], d[2]] for d in dets]

            pred_count = len(final_dets)

            self.results.append({
                'ann': ann,
                'dets': final_dets,
                'gt_count': gt_count,
                'pred_count': pred_count
            })

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()

        gts = [r['ann'] for r in results]
        preds = [r['dets'] for r in results]
        gt_counts = np.array([r['gt_count'] for r in results])
        pred_counts = np.array([r['pred_count'] for r in results])

        eval_results = OrderedDict()

        mean_aps = []
        tp_errors = []
        for distance_thr in self.distance_thrs:
            mean_ap, pd_val = eval_map2(preds, gts, distance_thr=distance_thr, logger=logger)
            mean_aps.append(mean_ap)

            eval_results[f'AP{int(distance_thr * 100):02d}'] = format_sig4(mean_ap * 100)

            errors = []
            for det_points, gt_points in zip(preds, gts):
                gt_points = np.array(gt_points)
                det_points = np.array(det_points)
                num_dets = det_points.shape[0]
                num_gts = gt_points.shape[0]
                if num_dets == 0 or num_gts == 0:
                    continue

                gt_coords = gt_points[:, :2].astype(np.float64)
                det_coords = det_points[:, :2].astype(np.float64)
                dist_matrix = np.linalg.norm(
                    det_coords[:, np.newaxis, :] - gt_coords[np.newaxis, :, :], axis=2)

                min_dists = dist_matrix.min(axis=1)
                min_indices = dist_matrix.argmin(axis=1)
                scores = det_points[:, -1]

                sort_inds = np.argsort(-scores)
                gt_covered = np.zeros(num_gts, dtype=bool)

                for i in sort_inds:
                    if min_dists[i] <= distance_thr:
                        matched_gt = min_indices[i]
                        if not gt_covered[matched_gt]:
                            gt_covered[matched_gt] = True
                            errors.append(min_dists[i])

            if errors:
                tp_errors.extend(errors)

                eval_results[f'TPPos_RMSE{int(distance_thr * 100):02d}'] = format_sig4(np.sqrt(np.mean(np.array(errors)**2)))
            else:
                eval_results[f'TPPos_RMSE{int(distance_thr * 100):02d}'] = 0.0

        mAP = pd_val * sum(mean_aps) / len(mean_aps) if self.pd else sum(mean_aps) / len(mean_aps)

        eval_results['mAP'] = format_sig4(mAP * 100)

        tp_pos_rmse = np.sqrt(np.mean(np.array(tp_errors)**2)) if tp_errors else 0.0
        eval_results['TPPos_RMSE'] = format_sig4(tp_pos_rmse)

        mae = np.mean(np.abs(gt_counts - pred_counts))
        acc = np.mean(gt_counts == pred_counts)

        eval_results['Count_MAE'] = format_sig4(mae)

        eval_results['Count_Acc'] = format_sig4(acc * 100)

        eval_results.move_to_end('mAP', last=False)

        logger.info(f"Final Eval Results: {eval_results}")

        return eval_results
