import torch
from torch import nn

# Modified from https://github.com/csjliang/LPTN/blob/main/codes/models/archs/LPTN_arch.py
class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(
            x.shape[0], x.shape[1], x.shape[2], x.shape[3],
            device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(
            x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2,
            device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img, input_level=0, return_images=False):
        current = img
        pyr = []
        images = [img]
        for _ in range(self.num_high - input_level):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(
                    up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
            images.append(down)
        pyr.append(current)
        if return_images:
            return pyr, images
        return pyr

    def pyramid_recons(self, pyr, return_images=False):
        image = pyr[-1]
        images = []
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(
                    up, size=(level.shape[2], level.shape[3]))
            image = up + level
            images.append(image)
        if return_images:
            images.reverse()
            images.append(pyr[-1])
            return images
        return image
