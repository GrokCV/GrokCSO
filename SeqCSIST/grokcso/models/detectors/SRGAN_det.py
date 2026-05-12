# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from mmengine.optim import OptimWrapperDict

from mmagic.models.base_models import BaseEditModel
from mmagic.models.utils import set_requires_grad
from mmengine.registry import MODELS
from tools.utils import read_targets_from_xml_list
import scipy.io as sio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@MODELS.register_module()
class SRGAN_gai(BaseEditModel):
    """SRGAN model for single image super-resolution.

    Ref:
    Photo-Realistic Single Image Super-Resolution Using a Generative
    Adversarial Network.

    Args:
        generator (dict): Config for the generator.
        discriminator (dict): Config for the discriminator. Default: None.
        gan_loss (dict): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict): Config for the pixel loss. Default: None.
        perceptual_loss (dict): Config for the perceptual loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`. Default: None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Default: None.
    """

    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):

        super().__init__(
            generator=generator,
            pixel_loss=pixel_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        # discriminator
        self.discriminator = MODELS.build(
            discriminator) if discriminator else None

        # loss
        self.gan_loss = MODELS.build(gan_loss) if gan_loss else None
        self.perceptual_loss = MODELS.build(
            perceptual_loss) if perceptual_loss else None

        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_repeat = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_repeat', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))

        self.register_buffer('step_counter', torch.tensor(0), False)

        if self.discriminator is None or self.gan_loss is None:
            # No GAN model or loss.
            self.disc_repeat = 0

    def forward_train(self, inputs, data_samples=None, **kwargs):
        """Forward training. Losses of training is calculated in train_step.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Tensor: Result of ``forward_tensor`` with ``training=True``.
        """

        return self.forward_tensor(
            inputs, data_samples, training=True, **kwargs)

    def forward_tensor(self, inputs, data_samples=None, training=False):
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            training (bool): Whether is training. Default: False.

        Returns:
            Tensor: result of simple forward.
        """

        feats = self.generator(inputs)

        return feats

    def if_run_g(self):
        """Calculates whether need to run the generator step."""

        return (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps)

    def if_run_d(self):
        """Calculates whether need to run the discriminator step."""

        return self.discriminator and self.gan_loss

    def g_step(self, batch_outputs: torch.Tensor, batch_gt_data: torch.Tensor):
        """G step of GAN: Calculate losses of generator.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.

        Returns:
            dict: Dict of losses.
        """

        losses = dict()

        # pix loss
        if self.pixel_loss:
            losses['loss_pix'] = self.pixel_loss(batch_outputs.to('cuda:0') , batch_gt_data.to('cuda:0') )

        # perceptual loss
        if self.perceptual_loss:
            loss_percep, loss_style = self.perceptual_loss(
                batch_outputs.to('cuda:0'), batch_gt_data.to('cuda:0'))
            if loss_percep is not None:
                losses['loss_perceptual'] = loss_percep
            if loss_style is not None:
                losses['loss_style'] = loss_style

        # gan loss for generator
        if self.gan_loss and self.discriminator:
            fake_g_pred = self.discriminator(batch_outputs)
            losses['loss_gan'] = self.gan_loss(
                fake_g_pred, target_is_real=True, is_disc=False)

        return losses

    def d_step_real(self, batch_outputs, batch_gt_data: torch.Tensor):
        """Real part of D step.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.

        Returns:
            Tensor: Real part of gan_loss for discriminator.
        """

        # real
        real_d_pred = self.discriminator(batch_gt_data)
        loss_d_real = self.gan_loss(
            real_d_pred, target_is_real=True, is_disc=True)

        return loss_d_real

    def d_step_fake(self, batch_outputs: torch.Tensor, batch_gt_data):
        """Fake part of D step.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.

        Returns:
            Tensor: Fake part of gan_loss for discriminator.
        """

        # fake
        fake_d_pred = self.discriminator(batch_outputs.detach())
        loss_d_fake = self.gan_loss(
            fake_d_pred, target_is_real=False, is_disc=True)

        return loss_d_fake

    def g_step_with_optim(self, batch_outputs: torch.Tensor,
                          batch_gt_data: torch.Tensor,
                          optim_wrapper: OptimWrapperDict):
        """G step with optim of GAN: Calculate losses of generator and run
        optim.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.
            optim_wrapper (OptimWrapperDict): Optim wrapper dict.

        Returns:
            dict: Dict of parsed losses.
        """

        g_optim_wrapper = optim_wrapper['generator']

        with g_optim_wrapper.optim_context(self):
            losses_g = self.g_step(batch_outputs, batch_gt_data)

        parsed_losses_g, log_vars_g = self.parse_losses(losses_g)
        g_optim_wrapper.update_params(parsed_losses_g)

        return log_vars_g

    def d_step_with_optim(self, batch_outputs: torch.Tensor,
                          batch_gt_data: torch.Tensor,
                          optim_wrapper: OptimWrapperDict):
        """D step with optim of GAN: Calculate losses of discriminator and run
        optim.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.
            optim_wrapper (OptimWrapperDict): Optim wrapper dict.

        Returns:
            dict: Dict of parsed losses.
        """

        log_vars = dict()
        d_optim_wrapper = optim_wrapper['discriminator']

        with d_optim_wrapper.optim_context(self):
            loss_d_real = self.d_step_real(batch_outputs, batch_gt_data)

        parsed_losses_dr, log_vars_dr = self.parse_losses(
            dict(loss_d_real=loss_d_real))
        log_vars.update(log_vars_dr)
        loss_dr = d_optim_wrapper.scale_loss(parsed_losses_dr)
        d_optim_wrapper.backward(loss_dr)

        with d_optim_wrapper.optim_context(self):
            loss_d_fake = self.d_step_fake(batch_outputs, batch_gt_data)

        parsed_losses_df, log_vars_df = self.parse_losses(
            dict(loss_d_fake=loss_d_fake))
        log_vars.update(log_vars_df)
        loss_df = d_optim_wrapper.scale_loss(parsed_losses_df)
        d_optim_wrapper.backward(loss_df)

        if d_optim_wrapper.should_update():
            d_optim_wrapper.step()
            d_optim_wrapper.zero_grad()

        return log_vars

    def extract_gt_data(self, data_samples):
        """extract gt data from data samples.

        Args:
            data_samples (list): List of DataSample.

        Returns:
            Tensor: Extract gt data.
        """

        batch_gt_data = data_samples.gt_img

        return batch_gt_data
    
    def forward(self,**kwargs)-> Dict[str, torch.Tensor]:
        mode = kwargs['mode']
        if mode == 'loss':
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            Phi_lrs_Name = 'SeqCSIST/data/phi_0.5.mat'
            Phi_lrs = sio.loadmat(Phi_lrs_Name)
            Phi_input = Phi_lrs['phi']
            self.Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor).to(device)

            Qinit_Name = 'SeqCSIST/data/track_5000_20/train/qinit.mat'
            Qinit_lrs = sio.loadmat(Qinit_Name)
            Qinit = Qinit_lrs['Qinit']
            self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

            batch = torch.stack(kwargs["batch_x"])
            batch_gt_data = batch.squeeze(dim=1)
            Phi_x = torch.stack(kwargs["gt_img_11"])
            batch_inputs = Phi_x.squeeze(dim=1)

        elif mode == 'predict':
            image_name = []
            Phi_x = torch.stack(kwargs["gt_img_11"])
            batch_inputs = Phi_x.squeeze(dim=1)
            ann_paths = kwargs["ann_path"]
            targets_GT = read_targets_from_xml_list(ann_paths)
            image_name = kwargs["image_name"]
            
        else:
            print("Invalid mode:", mode)
            return None

        
        if mode == 'tensor':
            batch_outputs = self.forward_train(batch_inputs)
            return batch_outputs[2:18,:]
        elif mode == 'predict':
            batch_outputs = self.forward_train(batch_inputs)
            return [batch_outputs[2:18,:], image_name, targets_GT]

        
    def train_step(self, data,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, torch.Tensor]:
        """Train step of GAN-based method.

        Args:
            data (List[dict]): Data sampled from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        g_optim_wrapper = optim_wrapper['generator']

        Phi_lrs_Name = 'SeqCSIST/data/phi_0.5.mat'
        Phi_lrs = sio.loadmat(Phi_lrs_Name)
        Phi_input = Phi_lrs['phi']
        self.Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor).to(device)

        Qinit_Name = 'SeqCSIST/data/track_5000_20/train/qinit.mat'
        Qinit_lrs = sio.loadmat(Qinit_Name)
        Qinit = Qinit_lrs['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

        batch = torch.stack(data["batch_x"])
        batch_gt_data = batch.squeeze(dim=1)
        Phi_x = torch.stack(data["gt_img_11"])
        batch_inputs = Phi_x.squeeze(dim=1)

        # batch_inputs, batch_gt_data, image_name= self.data_preprocessor_1(data)
        # batch_inputs = data['inputs']
        # data_samples = data['data_samples']
        # batch_gt_data = self.extract_gt_data(data_samples)

        log_vars = dict()

        with g_optim_wrapper.optim_context(self):
            batch_outputs = self.forward_train(batch_inputs)

        if self.if_run_g():
            set_requires_grad(self.discriminator, False)

            log_vars_d = self.g_step_with_optim(
                batch_outputs=batch_outputs,
                batch_gt_data=batch_gt_data,
                optim_wrapper=optim_wrapper)

            log_vars.update(log_vars_d)

        if self.if_run_d():
            set_requires_grad(self.discriminator, True)

            for _ in range(self.disc_repeat):
                # detach before function call to resolve PyTorch2.0 compile bug
                log_vars_d = self.d_step_with_optim(
                    batch_outputs=batch_outputs.detach(),
                    batch_gt_data=batch_gt_data,
                    optim_wrapper=optim_wrapper)

            log_vars.update(log_vars_d)

        if 'loss' in log_vars:
            log_vars.pop('loss')

        self.step_counter += 1
        return log_vars
    