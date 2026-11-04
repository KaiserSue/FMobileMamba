"""Explicit Rep conversion, FP32 verification and strict artifact reconstruction."""

from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
import math

import torch

from model.mobilemamba.rep_blocks import RepDepthwise3x3


@dataclass(frozen=True)
class VerificationReport:
    input_shape: list
    output_shape: list
    atol: float
    rtol: float
    max_abs_error: float
    max_rel_error: float
    passed: bool


def _rep_units(model):
    units = [m for m in model.modules() if isinstance(m, RepDepthwise3x3)]
    if not units:
        raise ValueError("model contains no Rep units")
    return units


def _require_eval_fp32(model):
    if any(m.training for m in model.modules()):
        raise ValueError("all modules must be in eval mode")
    if any(t.is_floating_point() and t.dtype != torch.float32
           for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Rep deployment verification requires FP32")


def convert_rep_model(model):
    """Return a deployment copy; preserve the source model and its state."""
    from model.mobilemamba.FMobileMamba_experiment import FMobileMambaExperimentModel

    if not isinstance(model, FMobileMambaExperimentModel):
        raise ValueError("expected an unwrapped FMobileMambaExperimentModel")
    _require_eval_fp32(model)
    units = _rep_units(model)
    if any(unit.deploy != model.rep_deploy for unit in units):
        raise ValueError("inconsistent Rep deployment state")
    converted = deepcopy(model)
    for unit in _rep_units(converted):
        unit.switch_to_deploy()
    converted.rep_deploy = True
    return converted


def verify_rep_equivalence(reference, candidate, inputs, atol, rtol):
    """Check one FP32 input numerically, raising before any artifact is published."""
    _require_eval_fp32(reference)
    _require_eval_fp32(candidate)
    if inputs.ndim != 4 or inputs.dtype != torch.float32 or not torch.isfinite(inputs).all():
        raise ValueError("verification requires finite FP32 NCHW inputs")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool)
           or not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError("tolerances must be finite and nonnegative")
    for model in (reference, candidate):
        if any(t.device != inputs.device for t in list(model.parameters()) + list(model.buffers())):
            raise ValueError("models and verification input must share a device")
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        with torch.inference_mode(), torch.autocast(device_type=inputs.device.type, enabled=False):
            expected, actual = reference(inputs), candidate(inputs)
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            raise ValueError("verification requires a single Tensor output")
        if expected.shape != actual.shape or expected.numel() == 0:
            raise ValueError("verification output shapes differ or are empty")
        if expected.dtype != torch.float32 or actual.dtype != torch.float32:
            raise ValueError("verification outputs must be FP32")
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise ValueError("verification outputs must be finite")
        difference = (expected - actual).abs()
        max_abs = difference.max().item()
        max_rel = (difference / expected.abs().clamp_min(torch.finfo(torch.float32).eps)).max().item()
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            raise ValueError(f"Rep equivalence failed: max_abs={max_abs}, max_rel={max_rel}, atol={atol}, rtol={rtol}")
        return VerificationReport(list(inputs.shape), list(actual.shape), float(atol), float(rtol), max_abs, max_rel, True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.backends.cudnn.benchmark = old_benchmark


def _validate_report(report):
    if not isinstance(report, dict) or report.get('passed') is not True:
        raise ValueError("artifact requires a passed verification report")
    for key in ('input_shape', 'output_shape'):
        if (not isinstance(report.get(key), list) or not report[key]
                or any(type(v) is not int or v <= 0 for v in report[key])):
            raise ValueError(f"invalid verification {key}")
    if len(report['input_shape']) != 4:
        raise ValueError('verification input_shape must be NCHW')
    for key in ('atol', 'rtol', 'max_abs_error', 'max_rel_error'):
        value = report.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid verification {key}")


def load_rep_deploy(path, device='cpu'):
    """Rebuild a trusted local deployment artifact and strictly load its weights."""
    from model import get_model

    payload = torch.load(path, map_location='cpu')
    if (not isinstance(payload, dict) or type(payload.get('format_version')) is not int
            or payload['format_version'] != 1
            or payload.get('artifact_type') != 'fmobilemamba_rep_deploy'
            or payload.get('deploy') is not True):
        raise ValueError("unsupported Rep deployment artifact format")
    name = payload.get('model_name')
    names = {f'FMobileMamba_experiment_{v}' for v in ('T2', 'T4', 'S6', 'B1', 'B2', 'B4')}
    if not isinstance(name, str) or name not in names:
        raise ValueError("unknown experiment factory in Rep artifact")
    kwargs = payload.get('model_kwargs')
    required = {'model_cfg', 'num_classes', 'global_mode', 'local_mode', 'fft_kernel_size',
                'local_kernel_set', 'distillation', 'deploy', 'fuse'}
    if not isinstance(kwargs, dict) or not required.issubset(kwargs):
        raise ValueError("artifact is missing effective construction parameters")
    if (kwargs['deploy'] is not True or kwargs['fuse'] is not False
            or {'checkpoint_path', 'ema', 'strict'} & kwargs.keys()
            or (kwargs['fft_kernel_size'] != 'rep' and kwargs['local_mode'] != 'MultiRep')
            or (kwargs['fft_kernel_size'] == 'rep' and kwargs['global_mode'] != 'fft')):
        raise ValueError("conflicting Rep construction metadata")
    source = payload.get('source')
    if (not isinstance(source, dict) or source.get('weights') not in ('net', 'net_E', 'state_dict')
            or any(not isinstance(source.get(k), str) for k in ('checkpoint_path', 'cfg_path'))
            or not isinstance(source.get('opts'), list)
            or any(not isinstance(v, str) for v in source['opts'])):
        raise ValueError("invalid artifact source metadata")
    verification = payload.get('verification')
    if not isinstance(verification, dict) or 'reload' not in verification:
        raise ValueError("missing artifact verification")
    _validate_report(verification.get('conversion'))
    if verification['reload'] is not None:
        _validate_report(verification['reload'])
    state = payload.get('net')
    if (not isinstance(state, dict) or not state
            or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                   or (v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all()))
                   for k, v in state.items())):
        raise ValueError("invalid FP32 deployment state_dict")
    options = deepcopy(kwargs)
    options.update(checkpoint_path='', ema=False, strict=True, pretrained=False)
    model = get_model(Namespace(name=name, model_kwargs=options))
    model.load_state_dict(state, strict=True)
    if not model.rep_deploy or any(not m.deploy or list(m._modules) != ['reparam'] for m in _rep_units(model)):
        raise ValueError("artifact did not reconstruct a deployed Rep model")
    return model.to(device=device, dtype=torch.float32).eval()
