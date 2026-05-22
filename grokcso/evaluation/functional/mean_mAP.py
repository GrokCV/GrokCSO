from multiprocessing import Pool

import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def points_distances(bboxes1, bboxes2):
  # 如果bboxes1不为空, 则取前两列, 即坐标
  if bboxes1.shape[0] > 0:
    bboxes1 = bboxes1[:, :2]
  # 如果bboxes2不为空, 则取前两列, 即坐标
  if bboxes2.shape[0] > 0:
    bboxes2 = bboxes2[:, :2]

  rows = bboxes1.shape[0]
  cols = bboxes2.shape[0]

  if rows == 0 or cols == 0:
    max_value = np.finfo(np.float64).max
    return np.full((rows, cols), max_value)

  # 计算两个矩阵的距离
  diff = bboxes1[:, np.newaxis, :] - bboxes2[np.newaxis, :, :]

  # 计算距离
  distances = np.linalg.norm(diff, axis=2)

  return distances


# todo: 修改tp_ap定义
def CSO_tpfp(det_bboxes,
             gt_bboxes,
             iou_thr,
             ):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 3).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 3).
        iou_thr (float): IoU threshold to be considered as matched.
            Defaults to 0.5.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    gt_bboxes = np.array(gt_bboxes)[:, :-1]
    det_bboxes = np.array(det_bboxes)

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros(num_dets, dtype=np.float32)
    fp = np.zeros(num_dets, dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
      fp[...] = 1
      return tp, fp
    ious = points_distances(
        det_bboxes, gt_bboxes)
    # for each det, the min iou with all gts
    ious_min = ious.min(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmin = ious.argmin(axis=1)

    if det_bboxes.size == 0:
      fp[...] = 1
    else:
      # sort all dets in descending order by scores
      sort_inds = np.argsort(-det_bboxes[:, -1])
      gt_covered = np.zeros(num_gts, dtype=bool)
      for i in sort_inds:
        if ious_min[i] <= iou_thr:
          matched_gt = ious_argmin[i]
          if not gt_covered[matched_gt]:
            gt_covered[matched_gt] = True
            tp[i] = 1
          else:
            fp[i] = 1
        else:
          fp[i] = 1
    return tp, fp


def eval_map(det_results,
             annotations,
             iou_thr,
             logger=None,
             nproc=4,
             eval_mode='area'):

    assert len(det_results) == len(annotations)

    num_imgs = len(annotations) # 图片数量

    # There is no need to use multi processes to process
    # when num_imgs = 1 .
    if num_imgs > 1:
        assert nproc > 0, 'nproc must be at least one.'
        nproc = min(nproc, num_imgs)
        pool = Pool(nproc)
    # 处理的进程数

    cls_dets = det_results
    cls_gts = annotations

    # 多个图像并行处理
    tpfp = pool.starmap(
        CSO_tpfp,
        zip(cls_dets, cls_gts,
            [iou_thr for _ in range(num_imgs)],
            ))

    tp, fp = tuple(zip(*tpfp))

    # calculate gt number of each scale
    # 计算每个尺度范围内的真实边界框数量num_gts
    # ignored gts or gts beyond the specific scale are not counted
    num_gts = 0

    for j, bbox in enumerate(cls_gts):
        num_gts += np.array(bbox).shape[0]


    # sort all det bboxes by score, also sort tp and fp
    cls_dets = tuple(arr for arr in cls_dets if arr)  # 去除空数组
    cls_dets = np.vstack(cls_dets)
    # 将检测结果堆叠成一个二维数组，每一行是一个检测结果
    num_dets = cls_dets.shape[0]
    sort_inds = np.argsort(-cls_dets[:, -1])

    # 根据检测结果的置信度得分对检测结果进行降序排序，返回的是排序后的索引数组 sort_inds
    tp = np.hstack(tp)[sort_inds]
    fp = np.hstack(fp)[sort_inds]

    # 根据排序后的索引重新排列
    # calculate recall and precision with tp and fp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    eps = np.finfo(np.float32).eps
    recalls = tp / np.maximum(np.full(tp.shape, num_gts), eps)
    precisions = tp / np.maximum((tp + fp), eps)


    # calculate AP

    eval_results = []

    ap = average_precision(recalls, precisions, eval_mode)
    eval_results.append({
        'num_gts': num_gts,
        'num_dets': num_dets,
        'recall': recalls,
        'precision': precisions,
        'ap': ap
    })

    if num_imgs > 1:
        pool.close()

    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, logger=logger)

    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmengine.logging.print_log()` for details.
            Defaults to None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

 
# if __name__ == '__main__':
#   preds = [[[2.3, 0, 120], [1, 1.3, 130], [2.1, 2.1, 136], [0, 2, 100]],
#            [[2.5, 0.1, 115], [1.3, 1.3, 120], [2.3, 2.3, 136]],
#            [[2, 0, 120], [1, 1, 130], [2, 2, 136]]]
#   gts = [[[2, 0, 120], [1, 1, 130], [2, 2, 136]],
#          [[2, 0, 120], [1, 1, 130], [2, 2, 136]],
#          [[2, 0, 120], [1, 1, 130], [2, 2, 136]]]
#   iou_thrs = (0.05, 0.1, 0.15, 0.2, 0.25)
#   mean_aps = []
#   for iou_thr in iou_thrs:
#     mean_ap, _ = eval_map2(
#                 preds,
#                 gts,
#                 iou_thr=iou_thr,
#                 )
#     mean_aps.append(mean_ap)
#   print(mean_aps)

