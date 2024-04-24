import torch
from torch import nn


# We initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init_ver2(m):
    """Initialize the weights  to the kaiming_normal distribution

    Initialize the weights in the different layers of
    the network.

    Parameters
    ----------
    m : :py:class:`torch.nn.*`
      The layer to initialize

    """
    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
        torch.nn.init.normal_(m.weight, mean=1.0, std=0.02)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# We initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    """Initialize the weights  to the normal distribution

    Initialize the weights in the different layers of
    the network.

    Parameters
    ----------
    m : :py:class:`torch.nn.*`
      The layer to initialize

    """

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
        # 1-centered normal distribution, stdev==0.02 as specified in Radford et al, 2015
        torch.nn.init.normal_(m.weight, mean=1.0, std=0.02)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
