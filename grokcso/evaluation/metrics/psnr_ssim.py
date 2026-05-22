import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmengine.logging import MMLogger, print_log
from collections import OrderedDict
from typing import Optional

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

@METRICS.register_module()
class Similarity(BaseMetric):
    default_prefix: Optional[str] = 'cso_metric'

    def __init__(self,
                 c=3,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.c = c

    def process(self, data_batch: dict, outputs) -> None:
        output = outputs[0]
        x_output = output['x_final'].detach().cpu().numpy()
        ann_list = output['img_gt'].detach().cpu().numpy()

        for idx in range(x_output.shape[0]):
            pred_img = x_output[idx].reshape(11*self.c, 11*self.c)
            gt_img = ann_list[idx].reshape(11*self.c, 11*self.c)

            if pred_img.max() <= 1.0:
                pred_img_uint8 = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)
                gt_img_uint8 = (np.clip(gt_img, 0, 1) * 255).astype(np.uint8)
            else:
                pred_img_uint8 = np.clip(pred_img, 0, 255).astype(np.uint8)
                gt_img_uint8 = np.clip(gt_img, 0, 255).astype(np.uint8)

            psnr = compute_psnr(gt_img_uint8, pred_img_uint8, data_range=255)
            ssim = compute_ssim(gt_img_uint8, pred_img_uint8, data_range=255)

            self.results.append([psnr, ssim])

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        psnrs = np.array([x[0] for x in results]) 
        ssims = np.array([x[1] for x in results])
        
        inf_mask = np.isinf(psnrs)
        if np.any(inf_mask):
            count_inf = np.sum(inf_mask)
            # print_log(f"Warning: Found {count_inf} images with infinite PSNR (perfect match). Replacing inf with 100.0 dB.", logger)
            psnrs[inf_mask] = 100.0

        eval_results = OrderedDict()
        eval_results['PSNR'] = float(np.mean(psnrs))
        eval_results['SSIM'] = float(np.mean(ssims))

        print_log(f'Avg_PSNR: {eval_results["PSNR"]:.4f} | Avg_SSIM: {eval_results["SSIM"]:.4f}', logger)
        
        return eval_results

