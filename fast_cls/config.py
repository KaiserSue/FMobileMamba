"""Configuration copies and explicit Namespace/dict overrides."""
import ast
import copy
import importlib
import math
import os
import re
import shlex
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import ModuleType


_SHARED_TYPES = {
    'seed': int,
    'size': int,
    'epoch_full': int,
    'warmup_epochs': int,
    'test_start_epoch': int,
    'batch_size': int,
    'lr': float,
    'weight_decay': float,
    'nb_classes': int,
    'ft': bool,
}


def split_shared_overrides(opts):
    shared_params = {}
    remaining_opts = []
    for opt in opts:
        key, separator, value = opt.partition('=')
        if not separator or not key:
            raise ValueError('Expected key=value: ' + opt)
        if not key.startswith('shared.'):
            remaining_opts.append(opt)
            continue
        name = key[len('shared.'):]
        if name not in _SHARED_TYPES:
            raise ValueError('Unknown shared key: ' + key)
        expected_type = _SHARED_TYPES[name]
        if expected_type is bool:
            if value not in ('true', 'false'):
                raise ValueError(key + ' must be true or false')
            parsed = value == 'true'
        elif expected_type is int:
            if re.fullmatch(r'-?[0-9]+', value) is None:
                raise ValueError(key + ' must be an integer')
            parsed = int(value)
        else:
            if re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', value) is None:
                raise ValueError(key + ' must be a finite number')
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(key + ' must be a finite number')
        shared_params[name] = parsed
    return shared_params, remaining_opts


def _unique_transform(transforms, transform_type, field_name):
    matches = [transform for transform in transforms
               if isinstance(transform, dict) and transform.get('type') == transform_type]
    if len(matches) != 1:
        raise ValueError(field_name + ' requires exactly one ' + transform_type + ' transform')
    return matches[0]


def synchronize_shared_config(cfg, shared_params):
    values = {name: shared_params.get(name, getattr(cfg, name)) for name in _SHARED_TYPES}
    if values['ft']:
        values['epoch_full'] = 30
        values['warmup_epochs'] = 5
        values['lr'] /= 6
        values['weight_decay'] = 1e-8

    for name, value in values.items():
        setattr(cfg, name, value)

    cfg.trainer.epoch_full = values['epoch_full']
    cfg.trainer.test_start_epoch = values['test_start_epoch']
    cfg.trainer.data.batch_size = values['batch_size']
    cfg.optim.lr = values['lr']
    cfg.optim.optim_kwargs['weight_decay'] = values['weight_decay']
    scheduler = cfg.trainer.scheduler_kwargs
    scheduler['warmup_epochs'] = values['warmup_epochs']
    scheduler['lr_min'] = values['lr'] / 100
    scheduler['warmup_lr'] = values['lr'] / 1000

    train_resize = _unique_transform(
        cfg.data.train_transforms, 'timm_create_transform', 'shared.size')
    test_resize = _unique_transform(cfg.data.test_transforms, 'Resize', 'shared.size')
    center_crop = _unique_transform(cfg.data.test_transforms, 'CenterCrop', 'shared.size')
    train_resize['input_size'] = values['size']
    test_resize['size'] = int(values['size'] / 0.875)
    center_crop['size'] = values['size']
    cfg.trainer.scale_kwargs['base_h'] = values['size']
    cfg.trainer.scale_kwargs['base_w'] = values['size']

    cfg.data.nb_classes = values['nb_classes']
    cfg.model.model_kwargs['num_classes'] = values['nb_classes']
    cfg.trainer.mixup_kwargs['num_classes'] = values['nb_classes']
    cfg.tea_model.model_kwargs['num_classes'] = values['nb_classes']


def apply_overrides(cfg, opts):
    for opt in opts:
        key, separator, value = opt.partition('=')
        if not separator or not all(key.split('.')):
            raise ValueError('Expected key=value: ' + opt)
        # Parsing failure deliberately means an unquoted string, as in the old CLI.
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            if not isinstance(node, (dict, Namespace)):
                raise ValueError('Non-container configuration path: ' + key)
            fields = node if isinstance(node, dict) else vars(node)
            if part not in fields:
                raise ValueError('Unknown configuration key: ' + key)
            node = fields[part]
        if not isinstance(node, (dict, Namespace)):
            raise ValueError('Non-container configuration path: ' + key)
        fields = node if isinstance(node, dict) else vars(node)
        if parts[-1] not in fields and '.'.join(parts[:-1]) not in ('model.model_kwargs', 'optim.optim_kwargs'):
            raise ValueError('Unknown configuration key: ' + key)
        fields[parts[-1]] = value


def load_config(args, project_root):
    root = Path(project_root).resolve()
    path = Path(args.cfg_path)
    path = (root / path).resolve()
    relative = path.relative_to(root)
    if path.suffix != '.py' or not path.is_file():
        raise ValueError('Configuration must be a project Python file: ' + str(path))
    module_name = '.'.join(relative.with_suffix('').parts)
    module = importlib.import_module(module_name)
    cfg = Namespace(**copy.deepcopy({k: v for k, v in vars(module).items()
                                    if not k.startswith('_') and not isinstance(v, ModuleType) and not callable(v)}))
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    cfg.cfg_path = module_name
    cfg.mode = 'test' if cfg.mode == 'test_net' else cfg.mode
    cfg.ft = cfg.mode == 'ft'
    cfg.trainer.name = 'FastCLSTrainer'
    cfg.model.model_kwargs.setdefault('global_mode', 'fft')
    cfg.model.model_kwargs.setdefault('local_mode', 'layeroperator')
    cfg.trainer.iter_full = None
    td = cfg.trainer.data
    td.persistent_workers = True
    td.num_workers_per_gpu_eval = td.num_workers_per_gpu
    td.prefetch_factor = td.prefetch_factor_eval = 2
    td.non_blocking = True
    for split in ('train', 'val', 'test'):
        vars(cfg.data).setdefault(split + '_subdir', split)
    production_root = Path(cfg.trainer.checkpoint).resolve()
    cfg.trainer.checkpoint = str(production_root / 'fast_pipeline')
    if cfg.synthetic_smoke:
        cfg.trainer.checkpoint = str(root / 'runs/smoke_fast_cls')
        td.batch_size = 2 * int(os.environ.get('WORLD_SIZE', '1'))
        td.batch_size_test = None
        td.batch_size_per_gpu_test = 2
        td.num_workers_per_gpu = td.num_workers_per_gpu_eval = 2
        cfg.trainer.epoch_full = 2
        cfg.trainer.scheduler_kwargs['warmup_epochs'] = 0
        cfg.trainer.test_per_epoch = cfg.trainer.save_per_epoch = 1
        cfg.logging.train_log_per = 3
    shared_params, remaining_opts = split_shared_overrides(args.opts or [])
    if shared_params:
        synchronize_shared_config(cfg, shared_params)
    apply_overrides(cfg, remaining_opts)
    if cfg.synthetic_smoke != args.synthetic_smoke:
        raise ValueError('Select synthetic data with --synthetic-smoke, not a config override')
    if cfg.trainer.name != 'FastCLSTrainer':
        raise ValueError('trainer.name must be FastCLSTrainer')
    if cfg.synthetic_smoke and (Path(cfg.trainer.checkpoint).resolve() == production_root or production_root in Path(cfg.trainer.checkpoint).resolve().parents):
        raise ValueError('Synthetic output must be separate from the production checkpoint root')
    validate_config(cfg)
    cfg.task_start_time = time.perf_counter()
    cfg.command = shlex.join([sys.executable] + sys.argv)
    return cfg


def validate_config(cfg):
    if cfg.mode not in ('train', 'ft', 'val', 'test'):
        raise ValueError('Unsupported mode: ' + cfg.mode)
    td = cfg.trainer.data
    for name in ('num_workers_per_gpu', 'num_workers_per_gpu_eval'):
        value = vars(td)[name]
        if type(value) is not int or value < 0:
            raise ValueError(name + ' must be a nonnegative integer')
    for name in ('prefetch_factor', 'prefetch_factor_eval'):
        if type(vars(td)[name]) is not int or vars(td)[name] <= 0:
            raise ValueError(name + ' must be positive')
    for name, value in (
            ('size', cfg.size), ('data.nb_classes', cfg.data.nb_classes),
            ('logging.train_log_per', cfg.logging.train_log_per),
            ('trainer.test_per_epoch', cfg.trainer.test_per_epoch),
            ('trainer.save_per_epoch', cfg.trainer.save_per_epoch)):
        if type(value) is not int or value <= 0:
            raise ValueError(name + ' must be a positive integer')
    if type(cfg.seed) is not int or cfg.seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(cfg.trainer.test_start_epoch) is not int or cfg.trainer.test_start_epoch < 0:
        raise ValueError('trainer.test_start_epoch must be a nonnegative integer')
    for name in ('persistent_workers', 'pin_memory', 'non_blocking', 'drop_last'):
        if type(vars(td)[name]) is not bool:
            raise ValueError(name + ' must be boolean')
    if cfg.trainer.epoch_full is not None and (type(cfg.trainer.epoch_full) is not int or cfg.trainer.epoch_full <= 0):
        raise ValueError('trainer.epoch_full must be a positive integer')
    if cfg.trainer.epoch_full is None and (type(cfg.trainer.iter_full) is not int or cfg.trainer.iter_full <= 0):
        raise ValueError('trainer.iter_full must be positive when epoch_full is None')
    warmup_epochs = cfg.trainer.scheduler_kwargs['warmup_epochs']
    if type(warmup_epochs) is not int or warmup_epochs < 0:
        raise ValueError('trainer.scheduler_kwargs.warmup_epochs must be a nonnegative integer')
    if cfg.trainer.epoch_full is not None and warmup_epochs > cfg.trainer.epoch_full:
        raise ValueError('trainer.scheduler_kwargs.warmup_epochs must not exceed trainer.epoch_full')
    if type(cfg.ft) is not bool:
        raise ValueError('ft must be boolean')
    if type(cfg.optim.lr) not in (int, float) or not math.isfinite(cfg.optim.lr) or cfg.optim.lr <= 0:
        raise ValueError('optim.lr must be a positive finite number')
    weight_decay = cfg.optim.optim_kwargs['weight_decay']
    if type(weight_decay) not in (int, float) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('optim.optim_kwargs.weight_decay must be a nonnegative finite number')
    if not isinstance(cfg.trainer.checkpoint, str) or not cfg.trainer.checkpoint:
        raise ValueError('trainer.checkpoint must be a nonempty string')
    if not isinstance(cfg.trainer.resume_dir, str):
        raise ValueError('trainer.resume_dir must be a string')
    if cfg.trainer.ema is not None and not 0 <= cfg.trainer.ema < 1:
        raise ValueError('EMA decay must be in [0, 1)')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world <= 0:
        raise ValueError('WORLD_SIZE must be positive')
    if 'logger_rank' in vars(cfg) and not 0 <= cfg.logger_rank < world:
        raise ValueError('logger_rank must identify a launched rank')
    for global_name, local_name in (('batch_size', 'batch_size_per_gpu'), ('batch_size_test', 'batch_size_per_gpu_test')):
        value = vars(td)[global_name]
        if value is not None:
            if type(value) is not int or value <= 0 or value % world:
                raise ValueError(global_name + ' must be positive and divisible by WORLD_SIZE')
        elif type(vars(td)[local_name]) is not int or vars(td)[local_name] <= 0:
            raise ValueError(local_name + ' must be positive')
    class_counts = (
        cfg.data.nb_classes,
        cfg.model.model_kwargs['num_classes'],
        cfg.trainer.mixup_kwargs['num_classes'],
        cfg.tea_model.model_kwargs['num_classes'],
    )
    if any(value != cfg.data.nb_classes for value in class_counts):
        raise ValueError('Model, dataset, Mixup and teacher class counts must match')
    if cfg.trainer.scaler not in ('none', 'native', 'apex') or cfg.trainer.sync_BN not in ('none', 'native', 'timm', 'apex'):
        raise ValueError('Unsupported AMP or SyncBN mode')
    paths = []
    root = Path(cfg.data.root_dir).resolve()
    for split in ('train', 'val', 'test'):
        subdir = Path(vars(cfg.data)[split + '_subdir'])
        path = (root / subdir).resolve()
        if subdir.is_absolute() or root not in path.parents:
            raise ValueError('Split must be a subdirectory of the data root')
        paths.append(path)
    if len(set(paths)) != 3:
        raise ValueError('train/val/test directories must be distinct')
