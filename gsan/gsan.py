import torch
from torch import nn
import torch.nn.functional as F

from gsan.layers import Conv2dBlock, Res2dBlock, Lap_Pyramid_Conv, get_nonlinearity_layer, AdaptiveNorm

class GSAN(nn.Module):
    # Values from the max abs values
    bypass_in_scale = torch.FloatTensor([[0.37773720, 0.35912505, 0.30737990],
                                         [0.37801176, 0.37293965, 0.31109694],
                                         [0.38694460, 0.38137160, 0.33248110],
                                         [0.39483570, 0.38033080, 0.33549930],
                                         [0.30094090, 0.32920223, 0.27291647]]).cuda()
    bypass_in_scale = 1.0 / bypass_in_scale

    bypass_out_scale = torch.FloatTensor([[0.37773720, 0.35912505, 0.30737990],
                                          [0.37801176, 0.37293965, 0.31109694],
                                          [0.38694460, 0.38137160, 0.33248110],
                                          [0.39483570, 0.38033080, 0.33549930],
                                          [0.30094090, 0.32920223, 0.27291647]]).cuda()

    def __init__(self, K):
        super().__init__()
        self.K = K
        # The pyramid operator
        self.P = Lap_Pyramid_Conv(K)
        # Style mapping network
        self.S = StyleMappingNetwork()
        # Learnable scales
        self.V_in = []
        self.V_out = []
        # Mask blocks
        self.M_enc, self.M_dec = self.get_mask_blocks()
        # The low-level encoder/decoder
        self.E = self.get_encoder_low()
        self.D = self.get_decoder_low()

    def encode(self, x, input_level=0, training=False):
        # Input: x_a
        # Output: [p_a], ft_low_a, ft_mid_a and sty_a
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
        # Input: outputs_a with sty_b
        # Output: x_b
        # Decode style
        sty_b = inputs['sty']
        # adain_b = self.S.decode(sty_b)
        # Projection happens within activation norm
        adain_b = sty_b[..., 0, 0]
        # Do low-level decoding
        low_b = self.cond_forward(self.D, inputs['ft'], adain_b)
        tbn_b = torch.sigmoid(low_b)
        # Progressively generate the high-freq details
        p_b, ft_hf = self.mask_forward(inputs['p'], tbn_b, adain_b)
        # Pyramid reconstruction
        g_b = self.P.pyramid_recons(p_b, True)
        ft_a = ft_hf + [inputs['ft']]
        if training:
            outputs = {
                'x': g_b[0], 'g': g_b, 'ft': ft_a, 'p': p_b,
            }
            return outputs
        return g_b[0]

    def get_decoder_low(self):
        # Decoding half of low-res
        # Input: ft_low_a and adain_b
        # Output: tbn_b
        # Remove style first
        # No need for IN since AdaIN has IN
        model = []
        # Apply new style
        model += [
            AdaptiveNorm(256, 64, projection=True),
            get_nonlinearity_layer('leakyrelu', inplace=False)
        ]
        model += [Res2dBlock(256, 128, padding=1,
                             activation_norm_type='layer_2d',
                             padding_mode='reflect', nonlinearity='leakyrelu')]
        model += [Res2dBlock(128, 64, padding=1,
                             activation_norm_type='layer_2d',
                             padding_mode='reflect', nonlinearity='leakyrelu')]
        model += [Conv2dBlock(64, 16, 3, padding=1, padding_mode='reflect',
                              nonlinearity='leakyrelu')]
        model += [Conv2dBlock(16, 3, 3, padding=1, padding_mode='reflect')]
        return nn.Sequential(*model)

    def get_encoder_low(self):
        # Encoding half of mid-res
        # Input: p_n_a and tbn_a
        # Ouput: ft_mid_a
        model = [Conv2dBlock(3, 16, 3, padding=1,
                             padding_mode='reflect', nonlinearity='leakyrelu')]
        model += [Conv2dBlock(16, 64, 3, padding=1,
                              padding_mode='reflect', nonlinearity='leakyrelu')]
        model += [Res2dBlock(64, 128, padding=1,
                             activation_norm_type='layer_2d',
                             padding_mode='reflect', nonlinearity='leakyrelu')]
        model += [Res2dBlock(128, 256, 3, padding=1,
                             activation_norm_type='layer_2d',
                             padding_mode='reflect',  nonlinearity='leakyrelu')]
        model += [Conv2dBlock(256, 256, 3, padding=1, padding_mode='reflect')]
        return nn.Sequential(*model)

    def get_mask_blocks(self):
        mask_enc_blocks = nn.ModuleList()
        mask_dec_blocks = nn.ModuleList()
        n_channels = 16
        for k in range(self.K):
            # Separate encoder and decoder blocks
            mask_enc_block = nn.Sequential(
                Conv2dBlock(3,
                            n_channels, 3, padding=1,
                            padding_mode='reflect', nonlinearity='leakyrelu'),
                Conv2dBlock(n_channels, n_channels * 2, 3, padding=1,
                            padding_mode='reflect', nonlinearity='leakyrelu'),
                Conv2dBlock(
                    n_channels * 2, n_channels * 2, 3,
                    padding=1, padding_mode='reflect'),
            )
            mask_enc_blocks.append(mask_enc_block)

            mask_dec_block = nn.Sequential(
                AdaptiveNorm(n_channels * 2, 64, projection=True),
                get_nonlinearity_layer('leakyrelu', inplace=False),
                Conv2dBlock(n_channels * 2, n_channels, 3, padding=1,
                            padding_mode='reflect', nonlinearity='leakyrelu'),
                Conv2dBlock(n_channels, 3, 3, padding=1,
                            padding_mode='reflect'),
            )
            mask_dec_blocks.append(mask_dec_block)
            n_channels += 16
            self.V_in.append(self.bypass_in_scale[k])
            self.V_out.append(self.bypass_out_scale[k])
        return mask_enc_blocks, mask_dec_blocks

    def mask_forward(self, p_a, tbn_b, z_s_b):
        # Input: [p_a] and sp_mask
        # Output: [p_b]
        p_a = p_a[:-1]
        p_b = []
        ft = []
        # input level might be > 0
        for i in range(len(p_a)):
            k = self.K - i - 1
            # Do current level
            p_a_k = p_a[-1-i]
            p_a_k_in = p_a_k * self.V_in[k].view(1, 3, 1, 1)
            ft_k = self.M_enc[k](p_a_k_in)
            ft = [ft_k] + ft
            p_b_k = self.cond_forward(self.M_dec[k], ft_k, z_s_b)
            # Let's call this lin2 for now
            # A custom activation function that applies higher gradient at large input
            p_b_k = torch.where(torch.abs(p_b_k) < 1.0,
                                p_b_k, p_b_k * 4)
            p_b_k = p_b_k * self.V_out[k].view(1, 3, 1, 1)
            p_b = [p_b_k] + p_b
        p_b = p_b + [tbn_b]
        return p_b, ft

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