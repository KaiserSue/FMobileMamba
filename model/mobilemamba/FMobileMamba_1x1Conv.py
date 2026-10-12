"""T2 MobileMamba with a channel-mixing spectral bottleneck."""

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


class SpectralBottleneckFFTConvLayer(FFTConvLayer):
    """Apply 1x1, depthwise 3x3, 1x1 to stacked real and imaginary spectra."""

    def __init__(self, in_channel):
        if type(in_channel) is not int or in_channel < 2 or in_channel % 2:
            raise ValueError("FFT input channels must be an even integer >= 2")
        super().__init__(in_channel)
        middle = in_channel // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channel, middle, kernel_size=1),
            nn.BatchNorm2d(middle),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle, middle, kernel_size=3, padding=1, groups=middle),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle, 2 * in_channel, kernel_size=1),
        )


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
