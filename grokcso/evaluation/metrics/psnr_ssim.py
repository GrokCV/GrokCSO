from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmengine.logging import MMLogger, print_log

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def format_sig4(x):
    """Format number to 4 significant figures"""
    if x == 0:
        return 0.0
    return float(f'{x:.4g}')


@METRICS.register_module()
class Similarity(BaseMetric):

    default_prefix: Optional[str] = 'cso_metric'

    def __init__(self,
                 c=3,
                 brightness_threshold: float = 50,
                 postprocess: bool = False,
                 nms_threshold: float = 0.4,
                 use_loc_sort: bool = False,
                 local_patch_size: int = 3,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.c = c
        self.brightness_threshold = brightness_threshold
        self.postprocess = postprocess
        self.nms_threshold = nms_threshold
        self.use_loc_sort = use_loc_sort
        self.local_patch_size = local_patch_size
        self._local_patch_upsampled = int(local_patch_size * c)

    def _apply_nms(self, dets):
        """Apply Non-Maximum Suppression to detections."""
        if len(dets) == 0:
            return []
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
        return [[d[0], d[1], d[2]] for d in best_dets]

    def _reconstruct_image_from_dets(self, dets, shape):
        """Reconstruct an image from detection results (dets: list of [row, col, value])."""
        recon = np.zeros(shape, dtype=np.float32)
        for det in dets:
            row_coord, col_coord, value = det
            row_pixel = int(round(row_coord * self.c + (self.c - 1) // 2))
            col_pixel = int(round(col_coord * self.c + (self.c - 1) // 2))
            row_pixel = max(0, min(row_pixel, shape[0] - 1))
            col_pixel = max(0, min(col_pixel, shape[1] - 1))
            recon[row_pixel, col_pixel] = value
        return recon

    def _get_gt_coords_from_img(self, gt_img):
        """Extract non-zero (row, col) coordinates from the GT image."""
        coords = np.nonzero(gt_img > 0)
        return list(zip(coords[0], coords[1]))

    def _compute_local_metrics(self, gt_img, pred_img, gt_coords, patch_size):
        """Compute PSNR and SSIM only within local patches around GT points.

        Args:
            patch_size: size in the ORIGINAL 11x11 space (will be scaled by self.c internally)
        """
        half = int(patch_size // 2)
        psnr_list = []
        ssim_list = []

        gt_padded = np.pad(gt_img, half, mode='constant', constant_values=0)
        pred_padded = np.pad(pred_img, half, mode='constant', constant_values=0)

        for row, col in gt_coords:
            gt_patch = gt_padded[row:row + patch_size, col:col + patch_size]
            pred_patch = pred_padded[row:row + patch_size, col:col + patch_size]

            if gt_patch.size == 0:
                continue

            gt_patch_u8 = np.clip(gt_patch, 0, 255).astype(np.uint8)
            pred_patch_u8 = np.clip(pred_patch, 0, 255).astype(np.uint8)

            if gt_patch_u8.max() == gt_patch_u8.min():
                continue
            if pred_patch_u8.max() == pred_patch_u8.min():
                psnr_list.append(0.0)
                ssim_list.append(0.0)
                continue

            try:
                psnr_patch = compute_psnr(gt_patch_u8, pred_patch_u8, data_range=255)
            except Exception:
                psnr_patch = 0.0
            try:
                ssim_patch = compute_ssim(gt_patch_u8, pred_patch_u8, data_range=255)
            except Exception:
                ssim_patch = 0.0

            psnr_list.append(psnr_patch)
            ssim_list.append(ssim_patch)

        if not psnr_list:
            return 0.0, 0.0
        return float(np.mean(psnr_list)), float(np.mean(ssim_list))

    def process(self, data_batch: dict, outputs) -> None:
        output = outputs[0]
        x_output = output['x_final'].detach().cpu().numpy()
        ann_list = output['img_gt'].detach().cpu().numpy()

        use_offset = 'offset_pred' in output and output['offset_pred'] is not None
        offset_pred = output['offset_pred'].cpu().numpy() if use_offset else None

        if self.postprocess:
            count_list = torch.argmax(output['x_num'], dim=1) + 1

        x_output_copy = x_output.copy()
        x_output_copy[x_output_copy < self.brightness_threshold] = 0

        for idx in range(x_output.shape[0]):
            gt_img = ann_list[idx].reshape(11 * self.c, 11 * self.c)

            if not self.postprocess:
                pred_img = x_output[idx].reshape(11 * self.c, 11 * self.c)
            else:
                dets = []
                raw_pred = x_output_copy[idx]

                if raw_pred.ndim == 1:
                    matrix = raw_pred.reshape(11 * self.c, 11 * self.c)
                elif raw_pred.ndim == 2:
                    matrix = raw_pred
                elif raw_pred.ndim == 3:

                    matrix = raw_pred.squeeze(0)
                else:

                    matrix = raw_pred.reshape(11 * self.c, 11 * self.c)
                non_zero_indices = np.nonzero(matrix)

                for i in range(len(non_zero_indices[0])):
                    row = non_zero_indices[0][i]
                    col = non_zero_indices[1][i]
                    value = matrix[row, col]

                    if use_offset and offset_pred is not None:

                        try:
                            if offset_pred.ndim == 4 and offset_pred.shape[1] >= 2:
                                d_row = offset_pred[idx, 0, row, col]
                                d_col = offset_pred[idx, 1, row, col]
                            else:
                                d_row = 0.0
                                d_col = 0.0
                        except (IndexError, TypeError):
                            d_row = 0.0
                            d_col = 0.0
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

                dets = self._apply_nms(dets)
                count_val = int(count_list[idx])
                if len(dets) > count_val:
                    dets = dets[:count_val]

                shape = (11 * self.c, 11 * self.c)
                pred_img = self._reconstruct_image_from_dets(dets, shape)

            if pred_img.max() <= 1.0:
                pred_img_uint8 = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)
                gt_img_uint8 = (np.clip(gt_img, 0, 1) * 255).astype(np.uint8)
            else:
                pred_img_uint8 = np.clip(pred_img, 0, 255).astype(np.uint8)
                gt_img_uint8 = np.clip(gt_img, 0, 255).astype(np.uint8)

            psnr = compute_psnr(gt_img_uint8, pred_img_uint8, data_range=255)
            ssim = compute_ssim(gt_img_uint8, pred_img_uint8, data_range=255)

            gt_coords = self._get_gt_coords_from_img(gt_img)
            local_psnr, local_ssim = self._compute_local_metrics(
                gt_img, pred_img, gt_coords, self._local_patch_upsampled
            )

            self.results.append([psnr, ssim, local_psnr, local_ssim])

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        psnrs = np.array([x[0] for x in results])
        ssims = np.array([x[1] for x in results])
        local_psnrs = np.array([x[2] for x in results])
        local_ssims = np.array([x[3] for x in results])

        inf_mask = np.isinf(psnrs)
        if np.any(inf_mask):
            psnrs[inf_mask] = 100.0

        eval_results = OrderedDict()
        eval_results['PSNR'] = format_sig4(float(np.mean(psnrs)))
        eval_results['SSIM'] = format_sig4(float(np.mean(ssims)))
        eval_results['Local_PSNR'] = format_sig4(float(np.mean(local_psnrs)))
        eval_results['Local_SSIM'] = format_sig4(float(np.mean(local_ssims)))

        print_log(
            f'Global | PSNR: {eval_results["PSNR"]:.4f} | SSIM: {eval_results["SSIM"]:.4f} | '
            f'Local(orig={self.local_patch_size}, up={self._local_patch_upsampled}) | '
            f'PSNR: {eval_results["Local_PSNR"]:.4f} | SSIM: {eval_results["Local_SSIM"]:.4f}',
            logger
        )

        return eval_results
