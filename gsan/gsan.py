from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F

from gsan.layers import Conv2dBlock, Res2dBlock, Lap_Pyramid_Conv, get_nonlinearity_layer


class GSAN(nn.Module):
    sigma_init = torch.FloatTensor([[0.04020832, 0.03567401, 0.02935611],
                                    [0.05103851, 0.04890475, 0.03961657],
                                    [0.05615773, 0.05953120, 0.04856632],
                                    [0.05966155, 0.06538393, 0.05243944],
                                    [0.05236926, 0.05979212, 0.04683833],
                                    [0.06130973, 0.07831845, 0.05987991]])

    def __init__(self, K):
        super().__init__()
        self.K = K
        # The pyramid operator
        self.P = Lap_Pyramid_Conv(K)
        # Style mapping network
        self.S = StyleMappingNetwork()
        # Learnable scales
        self.V = nn.ParameterList()
        # Mask blocks
        self.M = self.get_bp_blocks()
        # The low-level encoder/decoder
        self.E = self.get_encoder_residual()
        self.D = self.get_decoder_residual()

    def encode(self, x, input_level=0, training=False):
        # Do pyramid decompose first
        if training:
            p_a, g_a = self.P.pyramid_decom(x, input_level, training)
        else:
            p_a = self.P.pyramid_decom(x, input_level)
        tbn_a = p_a[-1]
        # low-level encoding
        ft_a = self.E(tbn_a)
        # Encode style
        sty_a, logvar_a = self.S.encode(ft_a)
        # Outputs
        outputs = {
            'p': p_a, 'ft': ft_a, 'sty': sty_a,
            'logvar': logvar_a
        }
        if training:
            outputs['g'] = g_a
        return outputs

    def decode(self, inputs, training=False):
        sty_b = inputs['sty']
        # Projection happens within activation norm
        adain_b = sty_b[..., 0, 0]
        # Do low-level decoding
        low_b = self.cond_forward(self.D, inputs['ft'], adain_b)
        tbn_b = torch.sigmoid(low_b[:, :3])
        bdp_b = low_b[:, 3:]
        # Progressively generate the high-freq details
        p_b = self.bp_forward(inputs['p'], bdp_b, tbn_b, adain_b)
        # Pyramid reconstruction
        if training:
            g_b = self.P.pyramid_recons(p_b, training)
            return g_b
        x_b = self.P.pyramid_recons(p_b)
        return x_b

    def get_encoder_residual(self):
        # Encoding half of the residual pathway
        model = [Conv2dBlock(3, 16, 3, padding=1,
                             padding_mode='reflect', nonlinearity='leakyrelu')]
        model += [Conv2dBlock(16, 64, 3, padding=1,
                              padding_mode='reflect', nonlinearity='leakyrelu')]
        model += [Res2dBlock(64, 128, padding=1, padding_mode='reflect',
                             activation_norm_type='layer_2d', nonlinearity='leakyrelu')]
        model += [Res2dBlock(128, 256, 3, padding=1, padding_mode='reflect',
                             activation_norm_type='layer_2d', nonlinearity='leakyrelu')]
        return nn.Sequential(*model)

    def get_decoder_residual(self):
        # Decoding half of the residual pathway
        model = []
        # Apply new style
        model += [
            Conv2dBlock(
                256, 256, 3,
                padding=1, padding_mode='reflect',
                activation_norm_type='adaptive',
                activation_norm_params=SimpleNamespace(
                    activation_norm_type='instance',
                    cond_dims=64,
                    projection=True),
                nonlinearity='leakyrelu'
            )]
        model += [Res2dBlock(256, 128, padding=1, padding_mode='reflect',
                             activation_norm_type='layer_2d', nonlinearity='leakyrelu')]
        model += [Res2dBlock(128, 64, padding=1, padding_mode='reflect',
                             activation_norm_type='layer_2d', nonlinearity='leakyrelu')]
        model += [Conv2dBlock(64, 16, 3, padding=1, padding_mode='reflect',
                              nonlinearity='leakyrelu')]
        model += [Conv2dBlock(16, 6, 3, padding=1, padding_mode='reflect')]
        return nn.Sequential(*model)

    def get_bp_blocks(self):
        mask_blocks = nn.ModuleList()
        n_channels = 16
        for k in range(self.K):
            mask_block = nn.Sequential(
                Conv2dBlock(6, n_channels, 3, padding=1,
                            padding_mode='reflect', nonlinearity='leakyrelu'),
                Conv2dBlock(
                    n_channels, n_channels * 2, 3,
                    padding=1, padding_mode='reflect',
                    activation_norm_type='adaptive',
                    activation_norm_params=SimpleNamespace(
                        activation_norm_type='instance',
                        cond_dims=64,
                        projection=True),
                    nonlinearity='leakyrelu'
                ),
                Conv2dBlock(n_channels * 2, n_channels, 3, padding=1,
                            padding_mode='reflect', nonlinearity='leakyrelu'),
                Conv2dBlock(n_channels, 3, 3, padding=1,
                            padding_mode='reflect'),
            )
            mask_blocks.append(mask_block)
            n_channels += 16
            self.V.append(nn.Parameter(self.sigma_init[k]))
        self.V.append(nn.Parameter(self.sigma_init[k + 1]))
        return mask_blocks

    def bp_forward(self, p_a, p_b_K, tbn_b, z_s_b):
        p_a = p_a[:-1]
        p_b = []
        p_b_k = p_b_K * self.V[self.K].view(1, 3, 1, 1)
        # input level might be > 0
        for i in range(len(p_a)):
            k = self.K - i - 1
            # Upsample p_b_i
            p_b_k_up = nn.functional.interpolate(
                p_b_k, size=(p_a[-1 - i].shape[2], p_a[-1 - i].shape[3]))
            # Do current level
            p_a_k = p_a[-1 - i]
            # With progressive feedback
            p_b_k = self.cond_forward(self.M[k], torch.cat(
                (p_a_k, p_b_k_up), dim=1), z_s_b)
            # Apply learnable scalar
            p_b_k = F.instance_norm(p_b_k) * self.V[k].view(1, 3, 1, 1)
            p_b = [p_b_k] + p_b
        p_b.append(tbn_b)
        return p_b

    @staticmethod
    def cond_forward(net, x, s):
        for block in net:
            if getattr(block, 'conditional', False):
                x = block(x, s)
            else:
                x = block(x)
        return x


class StyleMappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoding
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(256, 128, 1)
        self.act = get_nonlinearity_layer('leakyrelu', True)
        self.fc_vae_mu = nn.Conv2d(128, 64, 1)
        self.fc_vae_var = nn.Conv2d(128, 64, 1)

    def encode(self, x):
        x = self.avgpool(x)
        # Normalize latent input (across channels of each sample)
        sums = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        x = x / sums
        # Produce encoded style vector
        x = self.fc1(x)
        x = self.act(x)
        mu = self.fc_vae_mu(x)
        logvar = self.fc_vae_var(x)
        return mu, logvar

    def decode(self, z):
        raise NotImplementedError(
            "Decoder is implemeted with projection=True in AdaIN.")
