# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved
import functools
import warnings

import numpy as np
import torch
import torch.nn as nn

from stainaire.layers import Conv2dBlock
from stainaire.utils.distributed import master_only_print as print


class Discriminator(nn.Module):
    r"""Multi-resolution patch discriminator.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config
            file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, dis_cfg, data_cfg):
        super(Discriminator, self).__init__()
        print('Multi-resolution patch discriminator initialization.')
        # We assume the first datum is the ground truth image.
        image_channels = 3
        # Calculate number of channels in the input label.
        num_labels = 0

        # Build the discriminator.
        kernel_size = getattr(dis_cfg, 'kernel_size', 3)
        num_filters = getattr(dis_cfg, 'num_filters', 128)
        max_num_filters = getattr(dis_cfg, 'max_num_filters', 512)
        num_discriminators = getattr(dis_cfg, 'num_discriminators', 2)
        num_layers = getattr(dis_cfg, 'num_layers', 5)
        activation_norm_type = getattr(dis_cfg, 'activation_norm_type', 'none')
        weight_norm_type = getattr(dis_cfg, 'weight_norm_type', 'spectral')
        print('\tBase filter number: %d' % num_filters)
        print('\tNumber of discriminators: %d' % num_discriminators)
        print('\tNumber of layers in a discriminator: %d' % num_layers)
        print('\tWeight norm type: %s' % weight_norm_type)
        num_input_channels = image_channels + num_labels
        self.model = MultiResPatchDiscriminator(num_discriminators,
                                                kernel_size,
                                                num_input_channels,
                                                num_filters,
                                                num_layers,
                                                max_num_filters,
                                                activation_norm_type,
                                                weight_norm_type)
        print('Done with the Multi-resolution patch '
              'discriminator initialization.')

    def forward(self, data, net_G_output, real=True):
        out_v, fea_v = self.model(net_G_output['g_v'])
        output = dict(out_v=out_v, fea_v=fea_v)
        if real:
            out_u, fea_u = self.model(net_G_output['g_u'])
            output.update(dict(out_u=out_u, fea_u=fea_u))
        return output


class MultiResPatchDiscriminator(nn.Module):
    r"""Multi-resolution patch discriminator.

    Args:
        num_discriminators (int): Num. of discriminators (one per scale).
        kernel_size (int): Convolution kernel size.
        num_image_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        num_layers (int): Num. of layers for the patch discriminator.
        max_num_filters (int): Maximum num. of filters in a layer.
        activation_norm_type (str): batch_norm/instance_norm/none/....
        weight_norm_type (str): none/spectral_norm/weight_norm
    """

    def __init__(self,
                 num_discriminators=3,
                 kernel_size=3,
                 num_image_channels=3,
                 num_filters=64,
                 num_layers=4,
                 max_num_filters=512,
                 activation_norm_type='',
                 weight_norm_type='',
                 **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type' and key != 'patch_wise':
                warnings.warn(
                    "Discriminator argument {} is not used".format(key))

        self.discriminators = nn.ModuleList()
        for i in range(num_discriminators):
            net_discriminator = NLayerPatchDiscriminator(
                kernel_size,
                num_image_channels,
                num_filters,
                num_layers,
                max_num_filters,
                activation_norm_type,
                weight_norm_type)
            self.discriminators.append(net_discriminator)
        print('Done with the Multi-resolution patch '
              'discriminator initialization.')

    def forward(self, input_x):
        r"""Multi-resolution patch discriminator forward.

        Args:
            input_x (list) : Input images.
        Returns:
            (tuple):
              - output_list (list): list of output tensors produced by
                individual patch discriminators.
              - features_list (list): list of lists of features produced by
                individual patch discriminators.
              - input_list (list): list of downsampled input images.
        """
        output_list = []
        features_list = []
        for i, d in enumerate(self.discriminators):
            output, features = d(input_x[i])
            output_list.append(output)
            features_list.append(features)
        return output_list, features_list


class NLayerPatchDiscriminator(nn.Module):
    r"""Patch Discriminator constructor.

    Args:
        kernel_size (int): Convolution kernel size.
        num_input_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        num_layers (int): Num. of layers for the patch discriminator.
        max_num_filters (int): Maximum num. of filters in a layer.
        activation_norm_type (str): batch_norm/instance_norm/none/....
        weight_norm_type (str): none/spectral_norm/weight_norm
    """

    def __init__(self,
                 kernel_size,
                 num_input_channels,
                 num_filters,
                 num_layers,
                 max_num_filters,
                 activation_norm_type,
                 weight_norm_type):
        super(NLayerPatchDiscriminator, self).__init__()
        self.num_layers = num_layers
        padding = int(np.floor((kernel_size - 1.0) / 2))
        nonlinearity = 'leakyrelu'
        base_conv2d_block = \
            functools.partial(Conv2dBlock,
                              kernel_size=kernel_size,
                              padding=padding,
                              weight_norm_type=weight_norm_type,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity,
                              # inplace_nonlinearity=True,
                              order='CNA')
        layers = [[base_conv2d_block(
            num_input_channels, num_filters, stride=2)]]
        for n in range(num_layers):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            stride = 2 if n < (num_layers - 1) else 1
            layers += [[base_conv2d_block(num_filters_prev, num_filters,
                                          stride=stride)]]
        layers += [[Conv2dBlock(num_filters, 1,
                                3, 1,
                                padding,
                                weight_norm_type=weight_norm_type)]]
        for n in range(len(layers)):
            setattr(self, 'layer' + str(n), nn.Sequential(*layers[n]))

    def forward(self, input_x):
        r"""Patch Discriminator forward.

        Args:
            input_x (N x C x H1 x W2 tensor): Concatenation of images and
                semantic representations.
        Returns:
            (tuple):
              - output (N x 1 x H2 x W2 tensor): Discriminator output value.
                Before the sigmoid when using NSGAN.
              - features (list): lists of tensors of the intermediate
                activations.
        """
        res = [input_x]
        for n in range(self.num_layers + 2):
            layer = getattr(self, 'layer' + str(n))
            x = res[-1]
            res.append(layer(x))
        output = res[-1]
        features = res[1:-1]
        return output, features
