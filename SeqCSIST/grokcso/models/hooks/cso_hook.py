# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer
# from grokcso.models.visualization import Visualizer

from mmdet.datasets.samplers import TrackImgSampler
from mmengine.registry import HOOKS
from mmdet.structures import DetDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes
from mmdet.visualization.palette import _get_adaptive_scales

from typing import Dict, Optional, Sequence, Union
from tools.utils import show_contrast

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class CSOVisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 draw: bool = False,
                 show: bool = False,
                 c=3,
                 image_name="",
                 wait_time: float = 0,
                 test_out_dir: Optional[str] = None,
                 backend_args: dict = None):
        self._visualizer: Visualizer = Visualizer()
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self._test_index = 0
        self.c = c
        self.image_name = image_name

    def before_test(self, runner) -> None:
      print(runner)
      try:
        x = 1 / 0
      except ZeroDivisionError:
        print("除以0错误")
      finally:
        print("finally语句块")

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test iteration.

        Args:
            runner (Runner): The runner of the training  process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='test')

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[Sequence, dict]] = None,
                    mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict or Sequence, optional): Outputs from model.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        if mode == 'test':
            x_output = outputs[0]
            #aligned_png_name = outputs[3]
            png_name = outputs[1]
            ann_list = outputs[2]
            x_output[x_output < 70] = 0
            gt=[]
            if batch_idx == 5:
              for idx in range(15):
                pred = x_output[idx].reshape(11*self.c, 11*self.c)
                name = png_name[idx+2]
                targets_GT = ann_list[idx+2]
                gt = np.zeros((11 * self.c, 11 * self.c))
                print("targets_GT:", targets_GT)
                for i in range(len(targets_GT)):
                  x, y, lightness = targets_GT[i][0], targets_GT[i][1], \
                                    targets_GT[i][2]
                  print("lightness:", lightness)
                  gt[int(round(self.c * x + 1, 0)), int(
                    round(self.c * y + 1, 0))] = lightness
                idx = idx + 20 * batch_idx
                show_contrast(gt, pred, batch_idx, idx, self.image_name, name, self.c)


