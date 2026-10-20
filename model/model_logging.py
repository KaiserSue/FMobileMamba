"""Write configuration provenance and actual block settings to the training log."""

from torch import nn

from model.mobilemamba.mobilemamba import (
    MBFFTConv2d,
    MBWTConv2d,
    MobileMamba,
    MobileMambaBlock,
    MobileMambaModule,
)
from util.util import log_msg


def log_model_configuration(logger, net, cfg_path, model_name):
    """Log a constructed, unwrapped model; cfg_path is the loaded config module."""
    if logger is None:
        return

    lines = [
        '==> MODEL CONFIGURATION',
        'config_file: ' + cfg_path.replace('.', '/') + '.py',
        'registered_model: ' + model_name,
    ]
    if isinstance(net, MobileMamba):
        lines.extend(_mobilemamba_structure(net))
    log_msg(logger, '\n'.join(lines))


def _mobilemamba_structure(net):
    stages = (net.blocks1, net.blocks2, net.blocks3)
    stage_nodes = [[module for module in stage.modules()
                    if isinstance(module, MobileMambaModule)] for stage in stages]
    depths = [sum(isinstance(module, MobileMambaBlock) for module in stage)
              for stage in stages]
    lines = [
        'actual MobileMamba structure:',
        'stage_depths: ' + str(depths),
    ]
    for stage_index, nodes in enumerate(stage_nodes, start=1):
        for block_index, node in enumerate(nodes, start=1):
            fft_kernels = []
            wavelet_kernels = []
            if isinstance(node.global_op, MBFFTConv2d):
                fft_kernels = _convolution_kernels(node.global_op.fft_conv)
            elif isinstance(node.global_op, MBWTConv2d):
                wavelet_kernels = _convolution_kernels(node.global_op.wavelet_convs)
            local_kernels = [kernel for kernel in _convolution_kernels(node.local_op)
                             if kernel != (1, 1)]
            lines.append(
                f'stage{stage_index}.block{block_index}: '
                f'global={type(node.global_op).__name__}, '
                f'local={type(node.local_op).__name__}, '
                f'fft_kernels={fft_kernels}, wavelet_kernels={wavelet_kernels}, '
                f'local_kernels={local_kernels}'
            )
    return lines


def _convolution_kernels(module):
    return [node.kernel_size for node in module.modules() if isinstance(node, nn.Conv2d)]
