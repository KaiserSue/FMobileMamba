"""Sequential native five-fold training and epoch-boundary resume."""
import copy
import math
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

from .folds import load_or_create_fold_plan, fold_sample_ids
from .state import (activate_fold, complete_fold, create_or_load_state,
                    read_best_checkpoint, save_state, summarize)


def _plain(value):
    if isinstance(value, Namespace):
        return {key: _plain(item) for key, item in sorted(vars(value).items()) if not key.startswith('_')}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def config_snapshot(cfg):
    return dict(cfg_path=cfg.cfg_path, seed=cfg.seed, size=cfg.size, world_size=cfg.world_size,
                model=_plain(cfg.model), optim=_plain(cfg.optim), loss=_plain(cfg.loss),
                train_transforms=_plain(cfg.data.train_transforms),
                test_transforms=_plain(cfg.data.test_transforms),
                nb_classes=cfg.data.nb_classes, sampler=cfg.data.sampler,
                trainer=_plain(cfg.trainer), ft=cfg.ft)


def _broadcast(value, cfg):
    if cfg.world_size == 1:
        return value
    values = [value]
    dist.broadcast_object_list(values, src=cfg.logger_rank)
    return values[0]


def _barrier(cfg):
    if cfg.world_size > 1:
        dist.barrier()


def make_fold_config(cfg, plan, fold, run_dir, state):
    fold_cfg = copy.deepcopy(cfg)
    fold_index = plan.validation_order.index(fold) + 1
    fold_root = Path(run_dir) / ('fold_{:02d}_{}'.format(fold_index, fold))
    fold_cfg.data.type = 'NativeCVCLS'
    fold_cfg.trainer.checkpoint = str(fold_root)
    fold_cfg.trainer.resume_dir = ''
    old_logdir = state['fold_logdirs'].get(fold)
    if old_logdir is not None:
        old = Path(old_logdir)
        if old.parent.resolve() != fold_root.resolve():
            raise ValueError('Fold resume directory mismatch')
        fold_cfg.trainer.resume_dir = old.name
    fold_cfg.native_cv = Namespace(plan_path=str(Path(run_dir) / 'fold_plan.json'),
                                   validation_fold=fold, plan_digest=plan.plan_digest,
                                   expected_class_to_idx=plan.class_to_idx)
    fold_cfg.task_start_time = time.perf_counter()
    return fold_cfg


def _steps_per_epoch(cfg, plan, fold):
    train_ids, _ = fold_sample_ids(plan, fold)
    per_rank = math.ceil(len(train_ids) / cfg.world_size) if cfg.dist else len(train_ids)
    batch = cfg.trainer.data.batch_size_per_gpu
    steps = per_rank // batch if cfg.trainer.data.drop_last else math.ceil(per_rank / batch)
    if steps < 1:
        raise ValueError('Training fold has zero native batches')
    return steps


def run_cross_validation(cfg):
    from util.util import init_checkpoint
    from .trainer import NativeCVTrainer
    from evals.native_test import evaluate_checkpoint

    if cfg.ft:
        raise ValueError('Fine-tuning mode is incompatible with native CV resume')
    root = Path(cfg.data.root_dir).resolve()
    if not (root / cfg.data.test_subdir).is_dir():
        raise ValueError('Missing independent test split')
    requested = getattr(cfg, 'native_cv_resume_dir', '')
    run_dir = (Path(requested).resolve() if requested else
               Path(cfg.trainer.checkpoint).resolve() /
               ('native_cv_' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')))
    run_dir = Path(_broadcast(str(run_dir) if cfg.master else None, cfg))
    plan_path = run_dir / 'fold_plan.json'
    state_path = run_dir / 'state.json'
    if cfg.master:
        plan = load_or_create_fold_plan(root, plan_path, cfg.seed)
        state = create_or_load_state(state_path, plan.plan_digest, config_snapshot(cfg), bool(requested))
    else:
        plan = state = None
    _barrier(cfg)
    if not cfg.master:
        plan = load_or_create_fold_plan(root, plan_path, cfg.seed)
        state = create_or_load_state(state_path, plan.plan_digest, config_snapshot(cfg), True)
    if state['status'] == 'completed':
        return summarize(run_dir, state) if cfg.master else None
    for fold in plan.validation_order:
        if fold in state['completed_folds']:
            result = state['fold_results'][fold]
            if not Path(result['best_checkpoint']).is_file():
                raise ValueError('Completed fold checkpoint missing: ' + fold)
            continue
        steps = _steps_per_epoch(cfg, plan, fold)
        fold_cfg = make_fold_config(cfg, plan, fold, run_dir, state)
        if fold_cfg.trainer.resume_dir:
            fold_cfg.model.model_kwargs['checkpoint_path'] = ''
            prior_dir = Path(state['fold_logdirs'][fold])
            latest = torch.load(prior_dir / 'latest_ckpt.pth', map_location='cpu')
            if latest['iter'] != latest['epoch'] * steps:
                raise ValueError('Resume checkpoint is not at a complete epoch boundary')
            fold_cfg.total_time = latest['total_time']
            if latest['epoch'] >= cfg.trainer.epoch_full:
                read_best_checkpoint(prior_dir, steps, cfg.trainer.epoch_full)
            elif (any(epoch >= cfg.trainer.test_start_epoch or
                      epoch % cfg.trainer.test_per_epoch == 0
                      for epoch in range(1, latest['epoch'] + 1))
                  and not list(prior_dir.glob('best_epoch*.pth'))):
                raise ValueError('Missing best checkpoint from already validated fold')
        init_checkpoint(fold_cfg)
        fold_cfg.logdir = _broadcast(fold_cfg.logdir if cfg.master else None, cfg)
        if cfg.master:
            activate_fold(state_path, state, fold, fold_cfg.logdir)
        _barrier(cfg)
        trainer = NativeCVTrainer(fold_cfg)
        trainer.run()
        _barrier(cfg)
        if cfg.master:
            best_path, best_epoch, val_top1, val_top5, epoch = read_best_checkpoint(
                fold_cfg.logdir, steps, cfg.trainer.epoch_full)
        else:
            best_path = best_epoch = val_top1 = val_top5 = epoch = None
        best_path = Path(_broadcast(str(best_path) if cfg.master else None, cfg))
        test_started = time.perf_counter()
        test_result = evaluate_checkpoint(fold_cfg, best_path,
                                          Path(fold_cfg.logdir) / 'fixed_test', plan.class_to_idx)
        _barrier(cfg)
        if cfg.master:
            result = dict(fold_index=plan.validation_order.index(fold) + 1,
                          validation_fold=fold, best_val_top1=val_top1,
                          best_val_top5=val_top5, best_epoch=best_epoch,
                          best_checkpoint=str(best_path), epochs_completed=epoch,
                          test_top1=test_result.top1, test_top5=test_result.top5,
                          test_loss=test_result.loss, test_samples=test_result.samples,
                          fold_duration_seconds=fold_cfg.total_time + time.perf_counter() - test_started,
                          completed_at=datetime.now(timezone.utc).isoformat())
            complete_fold(state_path, state, fold, result)
        _barrier(cfg)
        if not cfg.master:
            state = create_or_load_state(state_path, plan.plan_digest, config_snapshot(cfg), True)
    if cfg.master:
        state['status'] = 'completed'
        save_state(state_path, state)
        return summarize(run_dir, state)
    return None
