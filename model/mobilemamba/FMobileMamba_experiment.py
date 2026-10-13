"""T2 MobileMamba experiment with selectable depthwise spectral kernels."""

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


class SpectralExperimentFFTConvLayer(nn.Module):
    """Apply one depthwise convolution to the concatenated FFT components."""

    def __init__(self, in_channel, kernel_size):
        if type(in_channel) is not int or in_channel <= 0:
            raise ValueError("FFT input channels must be a positive integer")
        if type(kernel_size) is not int or kernel_size not in (3, 5, 7):
            raise ValueError("spectral kernel_size must be 3, 5, or 7")
        super().__init__()
        spectral_channels = 2 * in_channel
        self.spectral = FFTConvLayer(in_channel)
        self.spectral.conv = nn.Sequential(
            nn.Conv2d(spectral_channels, spectral_channels, kernel_size=kernel_size,
                      stride=1, padding=kernel_size // 2, groups=spectral_channels),
            nn.BatchNorm2d(spectral_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.spectral(x)


class FMobileMambaExperimentModel(MobileMamba):
    """Install a specified FFT kernel in each T2 stage."""

    def __init__(self, fft_kernel_size=3, **model_options):
        if type(fft_kernel_size) is int and fft_kernel_size in (3, 5, 7):
            stage_kernel_sizes = (fft_kernel_size,) * 3
        elif type(fft_kernel_size) is str and fft_kernel_size == "k":
            stage_kernel_sizes = (7, 5, 3)
        else:
            raise ValueError("fft_kernel_size must be integer 3, 5, 7, or string 'k'")
        if normalize_global_mode(model_options.get("global_mode", "fft")) != "fft":
            raise ValueError("FMobileMamba_experiment only supports FFT mode")
        model_options["global_mode"] = "fft"
        super().__init__(**model_options)

        for stage, kernel_size, expected_count in zip(
                (self.blocks1, self.blocks2, self.blocks3),
                stage_kernel_sizes, (1, 2, 2)):
            fft_nodes = [module for module in stage.modules()
                         if isinstance(module, MBFFTConv2d)]
            if len(fft_nodes) != expected_count:
                raise RuntimeError(
                    f"T2 FFT stage structure mismatch: expected {expected_count} nodes, "
                    f"found {len(fft_nodes)}")
            for module in fft_nodes:
                module.fft_conv = SpectralExperimentFFTConvLayer(
                    module.in_channels, kernel_size)


@MODEL.register_module
def FMobileMamba_experiment_T2(num_classes=1000, pretrained=False,
                               distillation=False, fuse=False, pretrained_cfg=None,
                               model_cfg=CFG_MobileMamba_T2, global_mode=None,
                               local_mode=None, fft_kernel_size=3):
    if global_mode is not None and normalize_global_mode(global_mode) != "fft":
        raise ValueError("FMobileMamba_experiment_T2 only supports FFT mode")
    options = dict(model_cfg)
    options["global_mode"] = "fft"
    if local_mode is not None:
        options["local_mode"] = local_mode
    model = FMobileMambaExperimentModel(
        fft_kernel_size=fft_kernel_size, num_classes=num_classes,
        distillation=distillation, **options)
    if fuse:
        replace_batchnorm(model)
    return model
