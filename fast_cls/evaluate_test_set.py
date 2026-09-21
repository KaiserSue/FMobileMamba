"""Evaluate the configured test split with one explicit normal-model checkpoint."""
import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import logging
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .config import load_config
from .data import build_loaders
from .metrics import DeviceMetrics


@dataclass(frozen=True)
class CheckpointMetadata:
    class_to_idx: dict
    epoch: object = None
    pipeline_version: object = None


@dataclass(frozen=True)
class EvaluationResult:
    top1: float
    top5: float
    loss: float
    average_inference_seconds: float
    forward_seconds: float
    samples: int
    timed_samples: int


@dataclass(frozen=True)
class EvaluationContext:
    started_at: str
    finished_at: str
    checkpoint_path: str
    cfg_path: str
    checkpoint_epoch: object
    pipeline_version: object
    model_name: str
    global_mode: str
    local_mode: str
    effective_config: Mapping


def _positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def _nonnegative_integer(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--cfg-path', default='configs/mobilemamba/mobilemamba_b1.py')
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--global-mode', choices=('fft', 'wt'), default='wt')
    parser.add_argument('--local-mode', choices=('layeroperator', 'dwconv'), default='dwconv')
    parser.add_argument('--image-size', type=_positive_integer, default=192)
    parser.add_argument('--nb-classes', type=_positive_integer, default=20)
    parser.add_argument('--batch-size-per-gpu', type=_positive_integer, default=125)
    parser.add_argument('--num-workers-per-gpu', type=_nonnegative_integer, default=8)
    parser.add_argument('--dist-url', default='env://')
    parser.add_argument('--logger-rank', type=_nonnegative_integer, default=0)
    return parser


def _rooted_path(value, project_root):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_evaluation_config(args, project_root):
    project_root = Path(project_root).resolve()
    checkpoint_path = _rooted_path(args.checkpoint_path, project_root)
    log_dir = _rooted_path(args.log_dir, project_root)
    if not args.dist_url:
        raise ValueError('dist_url must not be empty')
    if args.global_mode not in ('fft', 'wt'):
        raise ValueError('global_mode must be fft or wt')
    if args.local_mode not in ('layeroperator', 'dwconv'):
        raise ValueError('local_mode must be layeroperator or dwconv')
    for name in ('image_size', 'nb_classes', 'batch_size_per_gpu'):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise ValueError(name + ' must be a positive integer')
    if type(args.num_workers_per_gpu) is not int or args.num_workers_per_gpu < 0:
        raise ValueError('num_workers_per_gpu must be a nonnegative integer')
    if type(args.logger_rank) is not int or args.logger_rank < 0:
        raise ValueError('logger_rank must be a nonnegative integer')

    opts = [
        'shared.size={}'.format(args.image_size),
        'shared.nb_classes={}'.format(args.nb_classes),
        'model.model_kwargs.checkpoint_path={}'.format(repr(str(checkpoint_path))),
        'model.model_kwargs.ema=False',
        'model.model_kwargs.strict=True',
        'model.model_kwargs.global_mode={}'.format(repr(args.global_mode)),
        'model.model_kwargs.local_mode={}'.format(repr(args.local_mode)),
        'trainer.data.batch_size_test=None',
        'trainer.data.batch_size_per_gpu_test={}'.format(args.batch_size_per_gpu),
        'trainer.data.num_workers_per_gpu_eval={}'.format(args.num_workers_per_gpu),
    ]
    config_args = argparse.Namespace(
        cfg_path=args.cfg_path,
        mode='test',
        sleep=-1,
        memory=-1,
        dist_url=args.dist_url,
        logger_rank=args.logger_rank,
        synthetic_smoke=False,
        opts=opts,
    )
    cfg = load_config(config_args, project_root)
    if cfg.mode != 'test' or cfg.model.model_kwargs['ema'] is not False:
        raise RuntimeError('Evaluation configuration must select test mode and normal weights')
    cfg.evaluation_checkpoint_path = str(checkpoint_path)
    cfg.evaluation_log_dir = str(log_dir)
    return cfg


def initialize_runtime(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; test-set evaluation requires a CUDA GPU')
    from util.net import init_training
    init_training(cfg)
    cfg.nnodes = max(1, cfg.world_size // int(os.environ.get('LOCAL_WORLD_SIZE', cfg.world_size)))
    cfg.device = torch.device('cuda', cfg.local_rank)
    return cfg.device


def prepare_file_logger(log_dir, is_master):
    if not is_master:
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'test_eval.log'
    logger = logging.Logger('fast_cls.test_eval.{}'.format(id(log_dir)), level=logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(str(log_path), mode='a', encoding='utf-8', delay=False)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    return logger


def _validate_optional_nonnegative_integer(value, name):
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(name + ' must be a nonnegative integer when present')


def load_checkpoint_metadata(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ValueError('Checkpoint is not a file: ' + str(checkpoint_path))
    if not os.access(str(checkpoint_path), os.R_OK):
        raise ValueError('Checkpoint is not readable: ' + str(checkpoint_path))
    state = torch.load(str(checkpoint_path), map_location='cpu')
    if not isinstance(state, Mapping):
        raise ValueError('Checkpoint must contain a top-level mapping')
    if 'net' not in state or not isinstance(state['net'], Mapping) or not state['net']:
        raise ValueError("Checkpoint must contain a nonempty 'net' weight mapping")
    mapping = state.get('class_to_idx')
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("Checkpoint must contain a nonempty 'class_to_idx' mapping")
    if (any(not isinstance(name, str) or type(index) is not int for name, index in mapping.items())
            or sorted(mapping.values()) != list(range(len(mapping)))):
        raise ValueError('class_to_idx values must be continuous integers starting at zero')
    _validate_optional_nonnegative_integer(state.get('epoch'), 'epoch')
    _validate_optional_nonnegative_integer(state.get('pipeline_version'), 'pipeline_version')
    return CheckpointMetadata(dict(mapping), state.get('epoch'), state.get('pipeline_version'))


def transfer_batch(value, device, non_blocking):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return type(value)((key, transfer_batch(item, device, non_blocking))
                           for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(transfer_batch(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [transfer_batch(item, device, non_blocking) for item in value]
    return value


def _model_logits(outputs):
    logits = outputs['out'] if isinstance(outputs, Mapping) else outputs
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] <= 0:
        raise ValueError('Model output must be a nonempty [batch, classes] tensor')
    return logits


@torch.no_grad()
def evaluate_test_set(cfg, loader, model):
    device = cfg.device
    expected_samples = cfg.data.test_length
    previous_mode = model.training
    metrics = DeviceMetrics(
        ('loss', 'top1', 'top5', 'invalid', 'forward_seconds', 'timed_samples'),
        device,
    )
    model.eval()
    from util.net import get_autocast
    amp_autocast = get_autocast(cfg.trainer.scaler)
    try:
        for batch_index, source_batch in enumerate(loader):
            batch = transfer_batch(source_batch, device, cfg.trainer.data.non_blocking)
            for key in ('img', 'target', 'valid'):
                if key not in batch or not isinstance(batch[key], torch.Tensor):
                    raise ValueError('Test batch must contain tensor field: ' + key)
            with amp_autocast():
                if batch_index:
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                outputs = _model_logits(model(batch['img']))
                if batch_index:
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                else:
                    elapsed = 0.0

            targets = batch['target']
            valid = batch['valid'].bool()
            if targets.ndim != 1 or valid.ndim != 1 or outputs.shape[0] != targets.shape[0] or valid.shape != targets.shape:
                raise ValueError('Test img, target, valid and output batch dimensions must match')
            losses = F.cross_entropy(outputs, targets, reduction='none')
            finite = torch.isfinite(outputs).all(dim=1) & torch.isfinite(losses)
            correct = outputs.topk(min(5, outputs.shape[1]), dim=1).indices.eq(targets[:, None])
            count = valid.sum()
            timed_count = count if batch_index else count.new_zeros(())
            elapsed_tensor = outputs.new_tensor(elapsed, dtype=torch.float64)
            sums = {
                'loss': torch.where(valid & finite, losses, 0.0).sum(),
                'top1': (correct[:, 0] & valid & finite).sum(),
                'top5': (correct.any(dim=1) & valid & finite).sum(),
                'invalid': (valid & ~finite).sum(),
                'forward_seconds': elapsed_tensor,
                'timed_samples': timed_count,
            }
            one = count.new_ones(())
            counts = {
                'loss': count,
                'top1': count,
                'top5': count,
                'invalid': one,
                'forward_seconds': one,
                'timed_samples': one,
            }
            metrics.update(sums, counts)

        snapshot = metrics.flush()
        samples = int(snapshot.get('top1', {}).get('count', 0))
        timed_samples = int(snapshot.get('timed_samples', {}).get('sum', 0))
        invalid = snapshot.get('invalid', {}).get('sum', 0)
        forward_seconds = snapshot.get('forward_seconds', {}).get('sum', 0.0)
        if samples != expected_samples:
            raise RuntimeError('Test effective sample count mismatch: expected {}, got {}'.format(expected_samples, samples))
        if invalid:
            raise FloatingPointError('Model output or loss is non-finite for {} test samples'.format(int(invalid)))
        if timed_samples <= 0:
            raise RuntimeError('Test data has no valid samples after the warm-up batch')
        average = forward_seconds / timed_samples
        values = (snapshot['top1']['mean'], snapshot['top5']['mean'],
                  snapshot['loss']['mean'], forward_seconds, average)
        if not all(math.isfinite(value) for value in values) or average <= 0:
            raise FloatingPointError('Evaluation produced invalid metrics or inference time')
        return EvaluationResult(
            top1=snapshot['top1']['mean'] * 100,
            top5=snapshot['top5']['mean'] * 100,
            loss=snapshot['loss']['mean'],
            average_inference_seconds=average,
            forward_seconds=forward_seconds,
            samples=samples,
            timed_samples=timed_samples,
        )
    finally:
        model.train(previous_mode)


def _isoformat(value):
    return value.isoformat() if isinstance(value, datetime) else str(value)


def build_evaluation_context(cfg, metadata, result, started_at, finished_at):
    model_kwargs = cfg.model.model_kwargs
    return EvaluationContext(
        started_at=_isoformat(started_at),
        finished_at=_isoformat(finished_at),
        checkpoint_path=cfg.evaluation_checkpoint_path,
        cfg_path=cfg.cfg_path,
        checkpoint_epoch=metadata.epoch,
        pipeline_version=metadata.pipeline_version,
        model_name=cfg.model.name,
        global_mode=model_kwargs['global_mode'],
        local_mode=model_kwargs['local_mode'],
        effective_config={
            'image_size': cfg.size,
            'nb_classes': cfg.data.nb_classes,
            'world_size': cfg.world_size,
            'batch_size_per_gpu': cfg.trainer.data.batch_size_per_gpu_test,
            'num_workers_per_gpu': cfg.trainer.data.num_workers_per_gpu_eval,
            'amp_mode': cfg.trainer.scaler,
            'test_samples': result.samples,
            'timed_samples': result.timed_samples,
        },
    )


def write_evaluation_log(logger, context, result):
    lines = [
        '================ TEST EVALUATION ================',
        'started_at: ' + context.started_at,
        'finished_at: ' + context.finished_at,
        'cfg_path: ' + context.cfg_path,
        'checkpoint_path: ' + context.checkpoint_path,
    ]
    if context.checkpoint_epoch is not None:
        lines.append('checkpoint_epoch: {}'.format(context.checkpoint_epoch))
    if context.pipeline_version is not None:
        lines.append('pipeline_version: {}'.format(context.pipeline_version))
    lines.extend([
        'model_name: ' + context.model_name,
        'global_mode: ' + context.global_mode,
        'local_mode: ' + context.local_mode,
    ])
    lines.extend('{}: {}'.format(key, value) for key, value in context.effective_config.items())
    lines.extend([
        'Top-1: {:.3f}%'.format(result.top1),
        'Top-5: {:.3f}%'.format(result.top5),
        'Average inference time: {:.6f} s/img'.format(result.average_inference_seconds),
        'Loss: {:.6f}'.format(result.loss),
        '',
    ])
    payload = '\n'.join(lines) + '\n'
    if len(logger.handlers) != 1 or not isinstance(logger.handlers[0], logging.FileHandler):
        raise ValueError('Evaluation logger must have exactly one file handler')
    stream = logger.handlers[0].stream
    stream.write(payload)
    stream.flush()


def format_terminal_result(result):
    return '\n'.join((
        'Top-1: {:.3f}%'.format(result.top1),
        'Top-5: {:.3f}%'.format(result.top5),
        'Average inference time: {:.6f} s/img'.format(result.average_inference_seconds),
        'Loss: {:.6f}'.format(result.loss),
    ))


def _close_logger(logger):
    if logger is None:
        return
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()


def main(argv=None):
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    cfg = load_evaluation_config(args, project_root)
    logger = None
    try:
        device = initialize_runtime(cfg)
        logger = prepare_file_logger(Path(cfg.evaluation_log_dir), cfg.master)
        metadata = load_checkpoint_metadata(Path(cfg.evaluation_checkpoint_path))
        bundle = build_loaders(cfg, ('test',), metadata.class_to_idx)
        cfg.data.test_length = bundle.lengths['test']
        from model import get_model
        model = get_model(cfg.model).to(device)
        started_at = datetime.now().astimezone()
        result = evaluate_test_set(cfg, bundle.loaders['test'], model)
        finished_at = datetime.now().astimezone()
        if cfg.master:
            context = build_evaluation_context(cfg, metadata, result, started_at, finished_at)
            write_evaluation_log(logger, context, result)
            print(format_terminal_result(result))
    finally:
        _close_logger(logger)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
