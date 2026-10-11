"""Durable run state and checkpoint validation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from .folds import VALIDATION_ORDER, atomic_write_json


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(',', ':')).encode()).hexdigest()


def create_or_load_state(path, plan_digest, config_snapshot, resume):
    path = Path(path)
    config_digest = _digest(config_snapshot)
    if resume:
        if not path.is_file():
            raise ValueError('Missing resume state: ' + str(path))
        state = json.loads(path.read_text(encoding='utf-8'))
        if state['plan_digest'] != plan_digest or state['config_digest'] != config_digest:
            raise ValueError('Resume fingerprint mismatch: plan or configuration changed')
        if state['fold_order'] != list(VALIDATION_ORDER):
            raise ValueError('Resume fold order mismatch')
        return state
    if path.exists():
        raise ValueError('Refusing to overwrite state: ' + str(path))
    state = dict(schema_version=1, status='running', plan_digest=plan_digest,
                 config_digest=config_digest, seed=config_snapshot['seed'],
                 fold_order=list(VALIDATION_ORDER), active_fold=None, fold_logdirs={},
                 completed_folds=[], fold_results={}, started_at=_now(), updated_at=_now(),
                 accumulated_duration_seconds=0.0, last_error=None)
    atomic_write_json(path.with_name('config_fingerprint.json'), config_snapshot)
    atomic_write_json(path, state)
    return state


def save_state(path, state):
    if set(state['completed_folds']) != set(state['fold_results']):
        raise ValueError('Completed folds/results mismatch')
    state['updated_at'] = _now()
    atomic_write_json(path, state)


def activate_fold(path, state, fold, logdir):
    old = state['fold_logdirs'].get(fold)
    if old is not None and old != str(logdir):
        raise ValueError('Active fold logdir changed')
    state['active_fold'] = fold
    state['fold_logdirs'][fold] = str(logdir)
    state['status'] = 'running'
    save_state(path, state)


def complete_fold(path, state, fold, result):
    if fold in state['completed_folds'] or result['validation_fold'] != fold:
        raise ValueError('Invalid completed fold')
    state['fold_results'][fold] = result
    state['completed_folds'].append(fold)
    state['active_fold'] = None
    state['accumulated_duration_seconds'] += result.get('fold_duration_seconds', 0)
    save_state(path, state)


def read_best_checkpoint(logdir, steps_per_epoch, expected_epoch=None):
    logdir = Path(logdir)
    latest_path = logdir / 'latest_ckpt.pth'
    latest = torch.load(latest_path, map_location='cpu')
    epoch = latest['epoch']
    if (not isinstance(epoch, int) or epoch < 1 or steps_per_epoch < 1
            or latest['iter'] != epoch * steps_per_epoch):
        raise ValueError('Checkpoint is not at a complete epoch boundary')
    if expected_epoch is not None and epoch != expected_epoch:
        raise ValueError('Fold checkpoint epoch mismatch')
    best_paths = list(logdir.glob('best_epoch*.pth'))
    if len(best_paths) != 1:
        raise ValueError('Expected exactly one best checkpoint in ' + str(logdir))
    best = torch.load(best_paths[0], map_location='cpu')
    if not best.get('net'):
        raise ValueError('Best checkpoint has no normal model weights')
    scores = best['topk_recorder']
    best_epoch = best['epoch']
    if not 1 <= best_epoch <= epoch:
        raise ValueError('Invalid best checkpoint epoch')
    return best_paths[0], best_epoch, scores['net_top1'][-1], scores['net_top5'][-1], epoch


def summarize(run_dir, state):
    if state['completed_folds'] != list(VALIDATION_ORDER):
        raise ValueError('Five folds are not complete')
    results = [state['fold_results'][fold] for fold in VALIDATION_ORDER]
    summary = dict(fold_results=results,
                   mean_top1=sum(item['test_top1'] for item in results) / 5,
                   mean_top5=sum(item['test_top5'] for item in results) / 5,
                   total_duration_seconds=state['accumulated_duration_seconds'],
                   plan_digest=state['plan_digest'], config_digest=state['config_digest'],
                   completed_at=_now(), run_dir=str(run_dir))
    atomic_write_json(Path(run_dir) / 'summary.json', summary)
    return summary
