# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from functools import partial
import torch
import torch.nn.functional as F

from stainaire.losses import (GANLoss, PerceptualLoss, GaussianKLLoss)
from stainaire.trainers.base import BaseTrainer
from stainaire.utils.distributed import master_only_print as print
from stainaire.utils.meters import Meter


class MSLoss(torch.nn.Module):
    """
    Multi-Scale Loss.
    """
    def __init__(self, loss=F.l1_loss, use_in=False, weights=None):
        super(MSLoss, self).__init__()
        self.use_in = use_in
        self.weights = weights
        self.loss = loss

    def forward(self, input, target):
        assert len(input) == len(target), (len(input), len(target))
        assert isinstance(input, list) and isinstance(target, list)
        N = len(input)
        l = 0
        if self.weights is None:
            weights = [1.0 / N] * N
        else:
            weights = self.weights
        for x, t, w in zip(input, target, weights):
            if self.use_in:
                x = F.instance_norm(x)
                t = F.instance_norm(t)
            l += self.loss(x, t) * w
        return l


class Trainer(BaseTrainer):

    def __init__(self, cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D,
                 train_data_loader, val_data_loader):
        super().__init__(cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D,
                         train_data_loader, val_data_loader)
        self.gan_recon = getattr(cfg.trainer, 'gan_recon', False)

    def _init_tensorboard(self):
        r"""Initialize the tensorboard."""
        # Logging frequency: self.cfg.logging_iter
        self.meters = {}
        names = ['optim/gen_lr', 'optim/dis_lr',
                 'time/iteration', 'time/epoch']
        for name in names:
            self.meters[name] = Meter(name)

        # Logging frequency: self.cfg.image_display_iter
        self.image_meter = Meter('images')

    def _init_loss(self, cfg):
        self.criteria['gan'] = GANLoss(cfg.trainer.gan_mode)
        self.criteria['image_recon'] = MSLoss(weights=cfg.trainer.recon_loss.weights,
                                              loss=self._get_loss_func(cfg.trainer.recon_loss.loss))
        self.criteria['kl'] = GaussianKLLoss()
        self.criteria['latent'] = torch.nn.L1Loss()
        self.criteria['struct'] = MSLoss(use_in=True)
        self.criteria['ms'] = MSLoss()

        if getattr(cfg.trainer.loss_weight, 'perceptual', 0) > 0:
            # self.criteria['perceptual'] = \
            #     PerceptualLoss(cfg=cfg,
            #                    network=cfg.trainer.perceptual_loss.mode,
            #                    layers=cfg.trainer.perceptual_loss.layers,
            #                    weights=cfg.trainer.perceptual_loss.weights,
            #                    instance_normalized=True)
            self.criteria['perceptual'] = MSLoss(weights=cfg.trainer.recon_loss.weights,
                                                 loss=self._get_loss_func('perceptual'))

        for loss_name, loss_weight in cfg.trainer.loss_weight.__dict__.items():
            if loss_weight > 0:
                self.weights[loss_name] = loss_weight

    def _get_loss_func(self, loss):
        if loss == 'l1':
            return F.l1_loss
        elif loss == 'mse':
            return F.mse_loss
        elif loss == 'smoothl1':
            return partial(F.smooth_l1_loss, beta=self.cfg.trainer.recon_loss.beta)
        elif loss == 'perceptual':
            return PerceptualLoss(cfg=self.cfg,
                                  network=self.cfg.trainer.perceptual_loss.mode,
                                  layers=self.cfg.trainer.perceptual_loss.layers,
                                  weights=self.cfg.trainer.perceptual_loss.weights,
                                  instance_normalized=True)

    def gen_forward(self, data):
        perceptual = 'perceptual' in self.weights

        if self.current_iteration < getattr(
                self.cfg.gen_opt.lr_policy, 'n_steps_before_decay', 1000):
            self.net_G.mix_mode = 'shuffle'
        else:
            self.net_G.mix_mode = 'random'
        net_G_output = self.net_G(data, use_vae='kl' in self.weights)
        net_D_output = self.net_D(data, net_G_output, real=False)

        self._time_before_loss()

        # GAN loss
        self.gen_losses['gan'] = self.criteria['gan'](
            net_D_output['out_v'], True, dis_update=False)

        # Cycle reconstruction loss
        self.gen_losses['cycle_recon'] = \
            self.criteria['image_recon'](net_G_output['g_u_cyc'],
                                         net_G_output['g_u'])

        # Identity reconstruction loss
        self.gen_losses['idt_recon'] = \
            self.criteria['image_recon'](net_G_output['g_u_idt'],
                                         net_G_output['g_u'])

        # KL loss
        self.gen_losses['kl'] = \
            self.criteria['kl'](net_G_output['z_s_u'],
                                net_G_output['z_s_logvar_u'])

        # Latent mapping loss
        self.gen_losses['latent'] = \
            self.criteria['latent'](
                net_G_output['z_s_r'], net_G_output['z_s_v'])

        # Structure mapping loss
        self.gen_losses['struct'] = \
            self.criteria['struct'](net_G_output['ft_u'], net_G_output['ft_v'])

        # Mode seeking loss
        nomin = self.criteria['ms'](net_G_output['g_v2'], net_G_output['g_v'])
        denom = torch.mean(
            torch.abs(net_G_output['z_s_r'] - net_G_output['z_s_r2']))
        self.gen_losses['ms'] = 1 / (nomin / denom + 1e-5)

        # Perceptual loss
        if perceptual:
            # Inputs should be [-1, 1]
            self.gen_losses['perceptual'] = \
                self.criteria['perceptual'](net_G_output['g_v'],
                                            net_G_output['g_u'])
        # Compute total loss
        total_loss = self._get_total_loss(gen_forward=True)
        return total_loss

    def dis_forward(self, data):
        with torch.no_grad():
            net_G_output = self.net_G(data,
                                      cycle_recon=False,
                                      idt_recon=False,
                                      mode_seek=False,
                                      )
        net_G_output['v'].requires_grad = True
        net_D_output = self.net_D(data, net_G_output, real=True)

        self._time_before_loss()

        # GAN loss.
        self.dis_losses['gan'] = self.criteria['gan'](net_D_output['out_u'], True) + \
            self.criteria['gan'](net_D_output['out_v'], False)

        # Compute total loss
        total_loss = self._get_total_loss(gen_forward=False)
        return total_loss

    def _get_visualizations(self, data):
        r"""Compute visualization image.

        Args:
            data (dict): The current batch.
        """
        if self.cfg.trainer.model_average:
            net_G_for_evaluation = self.net_G.module.averaged_model
        else:
            net_G_for_evaluation = self.net_G
        with torch.no_grad():
            net_G_output = net_G_for_evaluation(data)
            vis_images = [data['images'],
                          net_G_output['u_idt'],
                          net_G_output['v'],
                          net_G_output['u_cyc']]
            return vis_images

    def write_metrics(self):
        r"""Compute metrics and save them to tensorboard"""
        pass
