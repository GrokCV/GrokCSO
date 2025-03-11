from multiprocessing import Pool

import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable
import matplotlib.pyplot as plt


def average_precision(recalls, precisions, mode='area'):
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


def points_distances(points1, points2):
  if points1.shape[0] > 0:
    points1 = points1[:, :2]
  if points2.shape[0] > 0:
    points2 = points2[:, :2]
  points1 = points1.astype(np.float32)
  points2 = points2.astype(np.float32)

  rows = points1.shape[0]
  cols = points2.shape[0]

  if rows == 0 or cols == 0:
    max_value = np.finfo(np.float64).max
    return np.full((rows, cols), max_value)

  diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]

  distances = np.linalg.norm(diff, axis=2)

  return distances


def CSO_tpfp(det_points,
             gt_points,
             distance_thr,
             ):
    """Checks whether the detection point is true positive or false positive.

    Args:
         det_points (ndarray): The detected points of the current image, with a
               shape of (m, 3), where each point contains (x, y, lightness).
         gt_points (ndarray): The true target points of the current image,
               with a shape of (n, 3), where each point contains (x, y,
               lightness).
         distance_thr (float): When the distance between the detected point and
               the true target point is less than distance_thr,
         the detected point is considered to be a true positive.

    Returns:
         tuple[np.ndarray]: (tp, fp)
         - tp: An array of shape (m,) where 1 represents true positive and 0
               represents false.
         - fp: An array of shape (m,) where 1 represents false positive and 0
               represents true.
    """
    gt_points = np.array(gt_points)[:, :-1]
    det_points = np.array(det_points)

    num_dets = det_points.shape[0]
    num_gts = gt_points.shape[0]
    tp = np.zeros(num_dets, dtype=np.float32)
    fp = np.zeros(num_dets, dtype=np.float32)

    if gt_points.shape[0] == 0:
      fp[...] = 1
      return tp, fp

    ious = points_distances(
        det_points, gt_points)

    ious_min = ious.min(axis=1)

    ious_argmin = ious.argmin(axis=1)

    if det_points.size == 0:
      fp[...] = 1
    else:
      sort_inds = np.argsort(-det_points[:, -1])
      gt_covered = np.zeros(num_gts, dtype=bool)
      for i in sort_inds:
        if ious_min[i] <= distance_thr:
          matched_gt = ious_argmin[i]
          if not gt_covered[matched_gt]:
            gt_covered[matched_gt] = True
            tp[i] = 1
          else:
            fp[i] = 1
        else:
          fp[i] = 1

    return tp, fp


def eval_map2(det_results,
             annotations,
             distance_thr,
             logger=None,
             nproc=4,
             eval_mode='area'):
    """
      The mean average precision (mAP) of the detection results was evaluated.

      Args:
          det_results (list[ndarray]): Detection result list, each element is a
           detection point of the current image, and the shape is (m, 3).
          annotations (list[ndarray]): A list of true target points, each
           element of which is a true target point of the current image,
           with a shape of (n, 3).
          distance_thr (float): IoU threshold, when the distance between the
           detection point and the true target point is less than this value,
           it is considered a match.
          logger (logging.Logger, optional): The logger to use for logging.
           Defaults to None.
          nproc (int, optional): The number of parallel processing processes,
           the default is 4.
          eval_mode (str, optional): Evaluation mode, the default is 'area',
           which means the evaluation mode is based on area calculation.

      Returns:
          tuple: return (mean_ap, eval_results)
              - mean_ap (float): mean average precision（mAP）。
              - eval_results (list[dict]): evaluation results.
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(annotations)

    if num_imgs > 1:
        assert nproc > 0, 'nproc at least 1'
        nproc = min(nproc, num_imgs)
        pool = Pool(nproc)

    cls_dets = det_results
    cls_gts = annotations

    tpfp = pool.starmap(
        CSO_tpfp,
        zip(cls_dets, cls_gts,
            [distance_thr for _ in range(num_imgs)],
            ))

    tp, fp = tuple(zip(*tpfp))

    num_gts = 0
    for j, bbox in enumerate(cls_gts):
        num_gts += np.array(bbox).shape[0]

    # 将所有检测结果按分数降序排序
    cls_dets = tuple(arr for arr in cls_dets if arr)  # 去除空数组
    if len(cls_dets) == 0:
        print("--------------No detections found.-------------------------")
        return 0.0, []
    cls_dets = np.vstack(cls_dets)
    num_dets = cls_dets.shape[0]
    sort_inds = np.argsort(-cls_dets[:, -1])

    tp = np.hstack(tp)[sort_inds]
    fp = np.hstack(fp)[sort_inds]

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    eps = np.finfo(np.float32).eps

    recalls = tp / np.maximum(np.full(tp.shape, num_gts), eps)
    precisions = tp / np.maximum((tp + fp), eps)

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
        table_data.append(['AP', '', '', '', f'{mean_ap[i]:.3f}'])

        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

