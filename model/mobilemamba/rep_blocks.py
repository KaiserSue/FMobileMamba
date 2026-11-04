"""Linear, three-branch depthwise convolutions with explicit deployment fusion."""

import torch
from torch import nn
from torch.nn import functional as F


class RepDepthwise3x3(nn.Module):
    """Sum DW3x3/BN, DW1x1/BN and identity/BN without an activation."""

    def __init__(self, channels, deploy=False):
        super().__init__()
        if type(channels) is not int or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if type(deploy) is not bool:
            raise ValueError("deploy must be bool")
        self.channels = channels
        self.deploy = deploy
        if deploy:
            self.reparam = nn.Conv2d(channels, channels, 3, padding=1,
                                     groups=channels, bias=True)
            return
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.rbr_identity = nn.BatchNorm2d(channels)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels or not x.is_floating_point():
            raise ValueError("expected floating NCHW input with matching channels")
        if self.deploy:
            return self.reparam(x)
        return self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x)

    @torch.no_grad()
    def get_equivalent_kernel_bias(self):
        if self.training:
            raise ValueError("Rep fusion requires eval() with fixed BN statistics")
        if self.deploy:
            return self.reparam.weight, self.reparam.bias
        kernels, biases = [], []
        identity = torch.zeros_like(self.rbr_dense[0].weight)
        identity[:, 0, 1, 1] = 1
        for kernel, bn in (
                (self.rbr_dense[0].weight, self.rbr_dense[1]),
                (F.pad(self.rbr_1x1[0].weight, (1, 1, 1, 1)), self.rbr_1x1[1]),
                (identity, self.rbr_identity)):
            if bn.training or bn.running_mean is None or bn.running_var is None:
                raise ValueError("Rep fusion requires fixed BN running statistics")
            if (not all(torch.isfinite(t).all() for t in
                        (kernel, bn.weight, bn.bias, bn.running_mean, bn.running_var))
                    or (bn.running_var < 0).any()):
                raise ValueError("Rep fusion requires finite parameters and nonnegative variance")
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            kernels.append(kernel * scale[:, None, None, None])
            biases.append(bn.bias - bn.running_mean * scale)
        kernel, bias = sum(kernels), sum(biases)
        if not torch.isfinite(kernel).all() or not torch.isfinite(bias).all():
            raise ValueError("nonfinite fused Rep parameters")
        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.training:
            raise ValueError("Rep fusion requires eval()")
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        conv = nn.Conv2d(self.channels, self.channels, 3, padding=1,
                         groups=self.channels, bias=True).to(kernel)
        conv.weight.copy_(kernel)
        conv.bias.copy_(bias)
        self.reparam = conv.eval()
        del self.rbr_dense, self.rbr_1x1, self.rbr_identity
        self.deploy = True
