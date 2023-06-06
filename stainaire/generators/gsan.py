import torch
from torch import nn
import torch.nn.functional as F

from stainaire.layers import Conv2dBlock, Res2dBlock
from stainaire.layers.laplacian import Lap_Pyramid_Conv
from stainaire.layers.nonlinearity import get_nonlinearity_layer
from stainaire.layers.activation_norm import AdaptiveNorm


class Generator(nn.Module):
    def __init__(self, gen_cfg, data_cfg, **kwargs):
        super().__init__()
        self.mix_mode = getattr(gen_cfg, 'mix_mode', 'shuffle')
        self.cfg = gen_cfg
        self.G = LaplacianAutoEncoder(self.cfg)

    def forward(self, data, cycle_recon=True, idt_recon=True,
                mode_seek=True, use_vae=True):
        net_G_output = dict()
        u = data['images']
        batch_size = u.shape[0]

        # Encode batch into morph and style
        outputs_u = self.G.encode(u, training=True)
        z_s_u = outputs_u['sty']
        z_s_logvar_u = outputs_u['logvar']
        g_u = outputs_u['g']
        p_u = outputs_u['p']
        net_G_output.update(dict(p_u=p_u, g_u=g_u,
                            z_s_u=z_s_u, z_s_logvar_u=z_s_logvar_u))
        # Perform identity reconstruction (using VAE)
        if idt_recon:
            if use_vae:
                # Get VAE version of adain parameters
                std = torch.exp(0.5 * z_s_logvar_u)
                eps = torch.randn_like(std)
                z_s_vae_u = eps.mul(std) + z_s_u
            else:
                z_s_vae_u = self.S.decode(z_s_u)
            inputs_idt = outputs_u
            inputs_idt['sty'] = z_s_vae_u
            g_u_idt = self.G.decode(inputs_idt, training=True)['g']
            u_idt = g_u_idt[0]
            net_G_output.update(dict(u_idt=u_idt, g_u_idt=g_u_idt))

        # Augmentations
        # Draw styles randomly
        z_s_r = torch.randn_like(z_s_u)
        inputs_v = outputs_u
        inputs_v['sty'] = z_s_r
        outputs_v = self.G.decode(inputs_v, training=True)
        g_v = outputs_v['g']
        v = g_v[0]
        ft_u = outputs_v['ft']
        net_G_output.update(dict(z_s_r=z_s_r, v=v, g_v=g_v, ft_u=ft_u))

        # Mode seeking
        if mode_seek:
            # Draw styles randomly
            z_s_r2 = torch.randn_like(z_s_r)
            inputs_v2 = outputs_u
            inputs_v2['sty'] = z_s_r2
            # Regenerate with the same condition
            outputs_v2 = self.G.decode(inputs_v2, training=True)
            v2 = outputs_v2['g'][0]
            g_v2 = outputs_v2['g']
            net_G_output.update(dict(v2=v2, z_s_r2=z_s_r2, g_v2=g_v2))

        # Cross-cycle reconstruction
        if cycle_recon:
            # Encode augmentation
            outputs_v = self.G.encode(v)
            z_s_v = outputs_v['sty']
            p_v = outputs_v['p']
            ft_v = outputs_v['ft']
            # Reconstruct
            inputs_cyc = outputs_v
            inputs_cyc['sty'] = z_s_u
            outputs_cyc = self.G.decode(inputs_cyc, training=True)
            g_u_cyc = outputs_cyc['g']
            u_cyc = g_u_cyc[0]
            ft_v = outputs_cyc['ft']
            net_G_output.update(
                dict(ft_v=ft_v, p_v=p_v, u_cyc=u_cyc, z_s_v=z_s_v, g_u_cyc=g_u_cyc))

        return net_G_output

    def inference(self, data, style_mode='random', theta=1.0, input_level=0):
        assert style_mode in ['random', 'bypass',
                              'shuffle', 'jitter', 'identity', 'cyclic']
        input_key = 'images'
        u = data[input_key]
        batch_size = u.shape[0]

        # Encode into morph and style
        outputs_u = self.G.encode(u, input_level)
        z_s_u = outputs_u['sty']
        z_s_logvar_u = outputs_u['logvar']

        # y is the output image
        inputs_y = outputs_u

        if style_mode == 'random':
            # Random gaussian sampling
            z_s_y = torch.randn_like(z_s_u) * theta
            # z_s_y = torch.rand_like(z_s_u) * 4 - 2
            # Get corresponding style params
            file_names = data['key'][input_key]['filename']
        elif style_mode == 'shuffle':
            # Randomly shuffle tensor
            rnd_idx = torch.randperm(batch_size)
            # Content is the same, style is shuffled
            z_s_y = z_s_u[rnd_idx]
            file_names = data['key'][input_key]['filename']
        elif style_mode == 'jitter':
            # Sample around the current style
            z_s_y = z_s_u + torch.randn_like(z_s_u) * theta
            file_names = data['key'][input_key]['filename']
        elif style_mode == 'identity':
            z_s_y = z_s_u
            file_names = data['key'][input_key]['filename']
        elif style_mode == 'cyclic':
            # Map to a random style then back to original
            z_s_r = torch.randn_like(z_s_u)
            inputs_v = outputs_u
            inputs_v['sty'] = z_s_r
            v = self.G.decode(inputs_v)
            outputs_v = self.G.decode(v)
            inputs_y = outputs_v
            z_s_y = z_s_u
            file_names = data['key'][input_key]['filename']
        elif style_mode == 'bypass':
            file_names = data['key'][input_key]['filename']
            return u, file_names
        else:
            raise NotImplementedError

        inputs_y['sty'] = z_s_y
        y = self.G.decode(inputs_y)
        return y, file_names


class LaplacianAutoEncoder(nn.Module):
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

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # The pyramid operator
        self.P = Lap_Pyramid_Conv(cfg.arch.num_levels)
        # Style mapping network
        self.S = StyleMappingNetwork(**cfg.smn)
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
                             activation_norm_type=self.cfg.arch.activation_norm_type,
                             padding_mode='reflect', nonlinearity='leakyrelu')]
        model += [Res2dBlock(128, 64, padding=1,
                             activation_norm_type=self.cfg.arch.activation_norm_type,
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
                             activation_norm_type=self.cfg.arch.activation_norm_type,
                             padding_mode='reflect', nonlinearity='leakyrelu')]
        model += [Res2dBlock(128, 256, 3, padding=1,
                             activation_norm_type=self.cfg.arch.activation_norm_type,
                             padding_mode='reflect',  nonlinearity='leakyrelu')]
        model += [Conv2dBlock(256, 256, 3, padding=1, padding_mode='reflect')]
        return nn.Sequential(*model)

    def get_mask_blocks(self):
        mask_enc_blocks = nn.ModuleList()
        mask_dec_blocks = nn.ModuleList()
        n_channels = self.cfg.arch.num_mask_channels
        for k in range(self.cfg.arch.num_levels):
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
        K = self.cfg.arch.num_levels
        # input level might be > 0
        for i in range(len(p_a)):
            k = K - i - 1
            # Do current level
            p_a_k = p_a[-1-i]
            p_a_k_in = p_a_k * self.V_in[k].view(1, 3, 1, 1)
            ft_k = self.M_enc[k](p_a_k_in)
            ft = [ft_k] + ft
            p_b_k = self.cond_forward(self.M_dec[k], ft_k, z_s_b)
            # Let's call this lin2 for now
            p_b_k = torch.where(torch.abs(p_b_k) < self.cfg.arch.beta,
                                p_b_k, p_b_k * self.cfg.arch.alpha)
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
    def __init__(self, in_channels, num_filters, out_channels, inner_channels, nonlinearity,
                 **kwargs):
        super().__init__()
        self._out_channels = out_channels
        # Encoding
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, inner_channels, 1)
        self.act = get_nonlinearity_layer(nonlinearity, True)
        self.fc_vae_mu = nn.Conv2d(inner_channels, num_filters, 1)
        self.fc_vae_var = nn.Conv2d(inner_channels, num_filters, 1)
        # Decoding
        self.fc2 = nn.Conv2d(num_filters, inner_channels, 1)
        self.fc_adain_mu = nn.Conv2d(inner_channels, out_channels, 1)
        self.fc_adain_var = nn.Conv2d(inner_channels, out_channels, 1)

    def forward(self, x):
        # use_vae controls whether z and adain stuff is controlled by vae
        vae_mu, vae_logvar = self.encode(x)
        z = vae_mu
        adain_beta, adain_gamma = self.decode(z)
        return adain_beta, adain_gamma

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
        x = self.fc2(z)
        x = self.act(x)
        beta = self.fc_adain_mu(x)
        logvar = self.fc_adain_var(x)
        gamma = torch.exp(0.5 * logvar)
        z_adain = torch.cat([beta, gamma], 1)
        return z_adain
