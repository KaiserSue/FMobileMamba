"""T2 MobileMamba with a depthwise 3x3 spectral FFT branch."""

import torch.nn as nn

from model import MODEL
from model.mobilemamba.mobilemamba import (
    CFG_MobileMamba_T2,
    FFTConvLayer,
    MBFFTConv2d,
    MobileMamba,
    normalize_global_mode,
    replace_batchnorm,
)


class SpectralBottleneckFFTConvLayer(nn.Module):
    """Process each real and imaginary spectral channel independently."""

    def __init__(self, in_channel):
        if type(in_channel) is not int or in_channel <= 0:
            raise ValueError("FFT input channels must be a positive integer")
        super().__init__()
        spectral_channels = 2 * in_channel
        self.spectral = FFTConvLayer(in_channel)
        self.spectral.conv = nn.Sequential(
            nn.Conv2d(spectral_channels, spectral_channels, kernel_size=3,
                      stride=1, padding=1, groups=spectral_channels),
            nn.BatchNorm2d(spectral_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.spectral(x)


class FMobileMamba1x1ConvModel(MobileMamba):
    """Keep the original backbone while replacing every FFT convolution."""

    def __init__(self, **model_options):
        if normalize_global_mode(model_options.get("global_mode", "fft")) != "fft":
            raise ValueError("FMobileMamba_1x1Conv only supports FFT mode")
        model_options["global_mode"] = "fft"
        super().__init__(**model_options)
        count = 0
        for module in self.modules():
            if isinstance(module, MBFFTConv2d):
                module.fft_conv = SpectralBottleneckFFTConvLayer(module.in_channels)
                count += 1
        if count == 0:
            raise RuntimeError("No FFT modules found to replace")


@MODEL.register_module
def FMobileMamba_1x1Conv_T2(num_classes=1000, pretrained=False,
                            distillation=False, fuse=False, pretrained_cfg=None,
                            model_cfg=CFG_MobileMamba_T2, global_mode=None,
                            local_mode=None):
    if global_mode is not None and normalize_global_mode(global_mode) != "fft":
        raise ValueError("FMobileMamba_1x1Conv_T2 only supports FFT mode")
    options = dict(model_cfg)
    options["global_mode"] = "fft"
    if local_mode is not None:
        options["local_mode"] = local_mode
    model = FMobileMamba1x1ConvModel(
        num_classes=num_classes, distillation=distillation, **options)
    if fuse:
        replace_batchnorm(model)
    return model
