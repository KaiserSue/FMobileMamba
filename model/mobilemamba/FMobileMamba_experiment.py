"""MobileMamba experiments with selectable spectral and local kernels."""

import torch
import torch.nn as nn

from model import MODEL
from model.mobilemamba.mobilemamba import (
    CFG_MobileMamba_B1,
    CFG_MobileMamba_B2,
    CFG_MobileMamba_B4,
    CFG_MobileMamba_S6,
    CFG_MobileMamba_T2,
    CFG_MobileMamba_T4,
    FFTConvLayer,
    LayerNorm,
    MBFFTConv2d,
    MobileMamba,
    MobileMambaModule,
    normalize_global_mode,
    replace_batchnorm,
)


def _resolve_fft_stage_specs(fft_kernel_size):
    if type(fft_kernel_size) is int and fft_kernel_size in (3, 5, 7):
        return ((fft_kernel_size, 1),) * 3
    if type(fft_kernel_size) is str:
        if fft_kernel_size in ("k", "k1"):
            return ((7, 1), (5, 1), (3, 1))
        if fft_kernel_size == "k2":
            return ((3, 3), (3, 2), (3, 1))
    raise ValueError(
        "fft_kernel_size must be integer 3, 5, 7, or string 'k', 'k1', 'k2'"
    )


def _resolve_local_kernel_paddings(local_kernel_set):
    kernel_sets = {
        "original": ((11, 5), (7, 3), (5, 2)),
        "reduced": ((9, 4), (7, 3), (5, 2)),
    }
    if type(local_kernel_set) is not str or local_kernel_set not in kernel_sets:
        raise ValueError("local_kernel_set must be 'original' or 'reduced'")
    return kernel_sets[local_kernel_set]


def _load_cuda_dcnv3():
    from pkg_resources import DistributionNotFound

    try:
        from model.ops_dcnv3.modules import DCNv3
    except (ImportError, ModuleNotFoundError, DistributionNotFound) as error:
        raise RuntimeError(
            "CUDA DCNv3 is unavailable; compile and install model/ops_dcnv3 "
            "with the selected Python environment before building local_branch_new"
        ) from error
    return DCNv3


class LocalBranchNew(nn.Module):
    """Large-kernel local branch paired with a channel-wise CUDA DCNv3 branch."""

    def __init__(self, in_channel, kernel_padding):
        if type(in_channel) is not int or in_channel < 2:
            raise ValueError("in_channel must be an integer >= 2")
        if (not isinstance(kernel_padding, (tuple, list))
                or len(kernel_padding) != 2
                or any(type(value) is not int for value in kernel_padding)
                or kernel_padding[0] <= 0
                or kernel_padding[0] % 2 != 1
                or kernel_padding[1] != kernel_padding[0] // 2):
            raise ValueError(
                "kernel_padding must contain an odd positive kernel and "
                "size-preserving padding"
            )
        super().__init__()
        big_channels = in_channel // 2
        small_channels = in_channel - big_channels
        self.in_channel = in_channel
        self.split_channels = (big_channels, small_channels)
        self.kernel = kernel_padding[0]
        self.padding = kernel_padding[1]
        self.norm1 = LayerNorm(
            in_channel, eps=1e-6, data_format="channels_first"
        )
        self.big_conv = nn.Sequential(
            nn.Conv2d(big_channels, big_channels, 1),
            nn.GELU(),
            nn.Conv2d(
                big_channels,
                big_channels,
                kernel_size=self.kernel,
                stride=1,
                padding=self.padding,
                dilation=1,
                groups=big_channels,
            ),
        )
        self.conv1x1_1 = nn.Conv2d(big_channels, big_channels, 1)
        self.conv1x1_2 = nn.Conv2d(big_channels, big_channels, 1)
        self.conv1x1_3 = nn.Conv2d(small_channels, small_channels, 1)
        dcnv3 = _load_cuda_dcnv3()
        self.small_conv = dcnv3(
            channels=small_channels,
            kernel_size=3,
            dw_kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=small_channels,
        )

    def forward(self, x):
        normalized = self.norm1(x)
        big_input, small_input = torch.split(
            normalized, self.split_channels, dim=1
        )
        big_output = self.conv1x1_2(
            self.big_conv(big_input) + self.conv1x1_1(big_input)
        )
        small_nchw = self.conv1x1_3(small_input)
        small_nhwc = small_nchw.permute(0, 2, 3, 1).contiguous()
        small_output = self.small_conv(small_nhwc)
        small_output = small_output.permute(0, 3, 1, 2).contiguous()
        return torch.cat((big_output, small_output), dim=1)


class SpectralExperimentFFTConvLayer(nn.Module):
    """Apply depthwise Conv-BN-ReLU units between a single FFT and IFFT."""

    def __init__(self, in_channel, kernel_size, num_blocks=1):
        if type(in_channel) is not int or in_channel <= 0:
            raise ValueError("FFT input channels must be a positive integer")
        if type(kernel_size) is not int or kernel_size not in (3, 5, 7):
            raise ValueError("spectral kernel_size must be 3, 5, or 7")
        if type(num_blocks) is not int or num_blocks not in (1, 2, 3):
            raise ValueError("spectral num_blocks must be 1, 2, or 3")
        if num_blocks > 1 and kernel_size != 3:
            raise ValueError("stacked spectral blocks require kernel_size=3")
        super().__init__()
        spectral_channels = 2 * in_channel
        self.spectral = FFTConvLayer(in_channel)
        layers = []
        for _ in range(num_blocks):
            layers.extend((
                nn.Conv2d(spectral_channels, spectral_channels, kernel_size=kernel_size,
                          stride=1, padding=kernel_size // 2, groups=spectral_channels),
                nn.BatchNorm2d(spectral_channels),
                nn.ReLU(inplace=True),
            ))
        self.spectral.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.spectral(x)


class FMobileMambaExperimentModel(MobileMamba):
    """Select FFT or WT; FFT accepts integers 3/5/7, k (k1 alias), k1, or k2."""

    def __init__(self, fft_kernel_size=3, local_kernel_set="reduced",
                 **model_options):
        global_mode = normalize_global_mode(model_options.get("global_mode", "fft"))
        if global_mode == "fft":
            stage_specs = _resolve_fft_stage_specs(fft_kernel_size)
        stage_depths = tuple(model_options.get("depth", (1, 2, 2)))
        if (len(stage_depths) != 3
                or any(type(depth) is not int or depth <= 0 for depth in stage_depths)):
            raise ValueError("model depth must contain three positive integers")
        stage_local_kernels = _resolve_local_kernel_paddings(local_kernel_set)
        requested_local_mode = model_options.get("local_mode", "layeroperator")
        if requested_local_mode not in (
                "layeroperator", "dwconv", "local_branch_new"):
            raise ValueError(
                "local_mode must be layeroperator, dwconv, or local_branch_new"
            )
        if requested_local_mode == "local_branch_new":
            model_options["local_mode"] = "layeroperator"
            model_options["kernels_padding"] = stage_local_kernels
        model_options["global_mode"] = global_mode
        super().__init__(**model_options)

        if requested_local_mode == "local_branch_new":
            self._install_local_branch_new(stage_local_kernels, stage_depths)

        if global_mode != "fft":
            return

        for stage, (kernel_size, num_blocks), expected_count in zip(
                (self.blocks1, self.blocks2, self.blocks3),
                stage_specs, stage_depths):
            fft_nodes = [module for module in stage.modules()
                         if isinstance(module, MBFFTConv2d)]
            if len(fft_nodes) != expected_count:
                raise RuntimeError(
                    f"FFT stage structure mismatch: expected {expected_count} nodes, "
                    f"found {len(fft_nodes)}")
            for module in fft_nodes:
                module.fft_conv = SpectralExperimentFFTConvLayer(
                    module.in_channels, kernel_size, num_blocks)

    def _install_local_branch_new(self, stage_kernel_paddings, stage_depths):
        replacements = []
        for stage_index, (stage, kernel_padding, expected_count) in enumerate(zip(
                (self.blocks1, self.blocks2, self.blocks3),
                stage_kernel_paddings,
                stage_depths), start=1):
            local_nodes = [
                module for module in stage.modules()
                if isinstance(module, MobileMambaModule)
            ]
            if len(local_nodes) != expected_count:
                raise RuntimeError(
                    f"local branch stage {stage_index} structure mismatch: "
                    f"expected {expected_count} nodes, found {len(local_nodes)}"
                )
            for module in local_nodes:
                if type(module.local_channels) is not int or module.local_channels <= 0:
                    raise RuntimeError(
                        f"local branch stage {stage_index} has invalid channel count: "
                        f"{module.local_channels}"
                    )
                replacements.append((
                    module,
                    LocalBranchNew(module.local_channels, kernel_padding),
                ))

        for module, local_op in replacements:
            module.local_op = local_op


def _build_fmobilemamba_experiment(num_classes, distillation, fuse, model_cfg,
                                   global_mode, local_mode, fft_kernel_size,
                                   local_kernel_set):
    options = dict(model_cfg)
    options["global_mode"] = (
        global_mode if global_mode is not None else options.get("global_mode", "fft")
    )
    if local_mode is not None:
        options["local_mode"] = local_mode
    model = FMobileMambaExperimentModel(
        fft_kernel_size=fft_kernel_size, local_kernel_set=local_kernel_set,
        num_classes=num_classes,
        distillation=distillation, **options)
    if fuse:
        replace_batchnorm(model)
    return model


@MODEL.register_module
def FMobileMamba_experiment_T2(num_classes=1000, pretrained=False,
                               distillation=False, fuse=False, pretrained_cfg=None,
                               model_cfg=CFG_MobileMamba_T2, global_mode=None,
                               local_mode=None, fft_kernel_size=3,
                               local_kernel_set="reduced"):
    """Build T2; fft_kernel_size accepts 3/5/7, k (alias of k1), k1, k2."""
    return _build_fmobilemamba_experiment(
        num_classes, distillation, fuse, model_cfg, global_mode, local_mode,
        fft_kernel_size, local_kernel_set)


@MODEL.register_module
def FMobileMamba_experiment_T4(num_classes=1000, pretrained=False,
                               distillation=False, fuse=False, pretrained_cfg=None,
                               model_cfg=CFG_MobileMamba_T4, global_mode=None,
                               local_mode=None, fft_kernel_size=3,
                               local_kernel_set="reduced"):
    """Build T4; fft_kernel_size accepts 3/5/7, k (alias of k1), k1, k2."""
    return _build_fmobilemamba_experiment(
        num_classes, distillation, fuse, model_cfg, global_mode, local_mode,
        fft_kernel_size, local_kernel_set)


@MODEL.register_module
def FMobileMamba_experiment_S6(num_classes=1000, pretrained=False,
                               distillation=False, fuse=False, pretrained_cfg=None,
                               model_cfg=CFG_MobileMamba_S6, global_mode=None,
                               local_mode=None, fft_kernel_size=3,
                               local_kernel_set="reduced"):
    """Build S6; fft_kernel_size accepts 3/5/7, k (alias of k1), k1, k2."""
    return _build_fmobilemamba_experiment(
        num_classes, distillation, fuse, model_cfg, global_mode, local_mode,
        fft_kernel_size, local_kernel_set)


@MODEL.register_module
def FMobileMamba_experiment_B1(num_classes=1000, pretrained=False,
                               distillation=False, fuse=False, pretrained_cfg=None,
                               model_cfg=CFG_MobileMamba_B1, global_mode=None,
                               local_mode=None, fft_kernel_size=3,
                               local_kernel_set="reduced"):
    """Build B1; fft_kernel_size accepts 3/5/7, k (alias of k1), k1, k2."""
    return _build_fmobilemamba_experiment(
        num_classes, distillation, fuse, model_cfg, global_mode, local_mode,
        fft_kernel_size, local_kernel_set)


@MODEL.register_module
def FMobileMamba_experiment_B2(num_classes=1000, pretrained=False,
                               distillation=False, fuse=False, pretrained_cfg=None,
                               model_cfg=CFG_MobileMamba_B2, global_mode=None,
                               local_mode=None, fft_kernel_size=3,
                               local_kernel_set="reduced"):
    """Build B2; fft_kernel_size accepts 3/5/7, k (alias of k1), k1, k2."""
    return _build_fmobilemamba_experiment(
        num_classes, distillation, fuse, model_cfg, global_mode, local_mode,
        fft_kernel_size, local_kernel_set)


@MODEL.register_module
def FMobileMamba_experiment_B4(num_classes=1000, pretrained=False,
                               distillation=False, fuse=False, pretrained_cfg=None,
                               model_cfg=CFG_MobileMamba_B4, global_mode=None,
                               local_mode=None, fft_kernel_size=3,
                               local_kernel_set="reduced"):
    """Build B4; fft_kernel_size accepts 3/5/7, k (alias of k1), k1, k2."""
    return _build_fmobilemamba_experiment(
        num_classes, distillation, fuse, model_cfg, global_mode, local_mode,
        fft_kernel_size, local_kernel_set)
