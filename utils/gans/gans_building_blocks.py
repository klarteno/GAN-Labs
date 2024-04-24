import torch
from torch import nn


# the following techniques are partly base on https://github.com/soumith/ganhacks

# simple addition layer of gaussian instance noise
# suggested in https://arxiv.org/abs/1906.04612

class GaussianNoise(nn.Module):
    # sigma: sigma*pixel value = stdev of added noise from normal distribution
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            # scale of noise = stdev of gaussian noise = sigma * pixel value
            scale = self.sigma * x.detach()
            sampled_noise = self.noise.expand(
                *x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x

# From Progressively Growing GANs https://arxiv.org/abs/1710.10196
# For every pixel in a feature map, divide that pixel
# by the L2 norm over that pixel's channels
# theoretically goes after batchnorm only in generator layers
"""
    ------------------------------------------------------------------------------------
    Pixelwise feature vector normalization.
    reference:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
    ------------------------------------------------------------------------------------
"""
# the model was trained with this layer, but it produces images with a lot of pixels and not clear images
class PixelwiseNorm(nn.Module):
    def __init__(self, alpha=1e-8):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(self.alpha).sqrt()
        y = x / y
        return y
    
# From Progressively Growing GANs https://arxiv.org/abs/1710.10196
# Standard deviation of each feature in the activation map is calculated 
# and then averaged over the minibatch. 
# goes on the final layer of discriminator, just before activation
class MinibatchStdDev(nn.Module):
    def __init__(self, alpha=1e-8):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        batch_size, _, height, width = x.shape
        y = x - x.mean(dim=0, keepdim=True)
        y = y.pow(2.).mean(dim=0, keepdim=False).add(self.alpha).sqrt()
        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(batch_size, 0, height, width)
        y = torch.cat([x, y], 1)
        return y
