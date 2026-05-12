# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.autograd as autograd
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import is_distributed
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.functional import conv2d

from mmengine.registry import MODELS


@MODELS.register_module()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge', 'l1'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type: str,
                 real_label_val: float = 1.0,
                 fake_label_val: float = 0.0,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan' or self.gan_type == 'smgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        elif self.gan_type == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input: torch.Tensor, target: bool) -> torch.Tensor:
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """

        return -input.mean() if target else input.mean()

    def get_target_label(self, input: torch.Tensor,
                         target_is_real: bool) -> Union[bool, torch.Tensor]:
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type == 'wgan':
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self,
                input: torch.Tensor,
                target_is_real: bool,
                is_disc: bool = False,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the target is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
            mask (Tensor): The mask tensor. Default: None.

        Returns:
            Tensor: GAN loss value.
        """

        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        elif self.gan_type == 'smgan':

            input_height, input_width = input.shape[2:]
            mask_height, mask_width = mask.shape[2:]

            # Handle inconsistent size between outputs and masks
            if input_height != mask_height or input_width != mask_width:
                input = F.interpolate(
                    input,
                    size=(mask_height, mask_width),
                    mode='bilinear',
                    align_corners=True)

                target_label = self.get_target_label(input, target_is_real)

            if is_disc:
                if target_is_real:
                    target_label = target_label
                else:
                    target_label = self.gaussian_blur(mask).detach().cuda(
                    ) if mask.is_cuda else self.gaussian_blur(
                        mask).detach().cpu()
                    # target_label = self.gaussian_blur(mask).detach().cpu()
                loss = self.loss(input, target_label)
            else:
                loss = self.loss(input, target_label) * mask / mask.mean()
                loss = loss.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight