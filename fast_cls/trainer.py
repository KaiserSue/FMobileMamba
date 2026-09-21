"""Classification lifecycle with independent training and evaluation statistics."""
import copy
import datetime
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F
from timm.data import Mixup
from timm.optim.lookahead import Lookahead
from timm.utils import dispatch_clip_grad
from .best_metrics import BestValidationMetrics
from .data import build_loaders
from .metrics import DeviceMetrics


@dataclass
class EpochResult:
    losses: dict
    samples: int
    seconds: float


@dataclass
class EvalResult:
    split: str
    model: str
    top1: float
    top5: float
    loss: Optional[float]
    loss_valid: bool
    samples: int


class FastCLSTrainer:
    def __init__(self, cfg, fold_context=None, resume_checkpoint=None):
        from model import get_model
        from loss import get_loss_terms
        from optim import get_optim
        from optim.scheduler import get_scheduler
        from util.net import get_autocast, get_loss_scaler, save_network_stats
        from util.util import log_cfg
        self.cfg = cfg
        self.fold_context = fold_context
        self.device = torch.device('cuda', cfg.local_rank)
        self.master, self.logger, self.writer = cfg.master, cfg.logger, cfg.writer
        self.closed, self.recorders = False, {}
        self.training = cfg.mode in ('train', 'ft')
        checkpoint = str(resume_checkpoint) if resume_checkpoint else cfg.model.model_kwargs['checkpoint_path']
        if not self.training and not checkpoint and not cfg.synthetic_smoke:
            raise ValueError('Independent evaluation requires an explicit checkpoint')
        state = torch.load(checkpoint, map_location='cpu') if checkpoint else {}
        if 'synthetic_smoke' in state and state['synthetic_smoke'] != cfg.synthetic_smoke:
            raise ValueError('Synthetic and production checkpoints cannot be mixed')
        if checkpoint and 'class_to_idx' not in state:
            self._log('Legacy checkpoint: historical class mapping cannot be verified')
        splits = ('train', 'val', 'test') if self.training else (cfg.mode,)
        self.bundle = build_loaders(
            cfg, splits, state.get('class_to_idx'),
            getattr(cfg, 'fold_plan', None),
            fold_context.validation_fold if fold_context else None)
        self.net = get_model(cfg.model).to(self.device).eval()
        if self.master:
            model_stat_path = Path(cfg.logdir) / 'model_stat.txt'
            save_network_stats([self.net], cfg.size, model_stat_path)
            self._log('Model statistics saved to {}'.format(model_stat_path))
        self.ema = cfg.trainer.ema or 0
        self.net_E = copy.deepcopy(self.net).eval() if self.ema else None
        if self.net_E is not None and state.get('net_E') is not None:
            self.net_E.load_state_dict(state['net_E'], strict=cfg.model.model_kwargs['strict'])
        self.dist_BN = cfg.trainer.dist_BN
        if cfg.dist and cfg.trainer.sync_BN != 'none':
            self.dist_BN = ''
            if cfg.trainer.sync_BN == 'native':
                converter = torch.nn.SyncBatchNorm.convert_sync_batchnorm
            elif cfg.trainer.sync_BN == 'timm':
                from timm.layers.norm_act import convert_sync_batchnorm as converter
            else:
                from apex.parallel import convert_syncbn_model as converter
            self.net = converter(self.net)
        self.optim = self.scheduler = self.loss_scaler = self.mixup_fn = None
        self.amp_autocast = get_autocast(cfg.trainer.scaler)
        if self.training:
            cfg.data.train_size = len(self.bundle.loaders['train'])
            cfg.data.train_length = self.bundle.lengths['train']
            scale = cfg.trainer.data.batch_size / 512
            cfg.optim.lr *= scale
            cfg.trainer.scheduler_kwargs['lr_min'] *= scale
            cfg.trainer.scheduler_kwargs['warmup_lr'] *= scale
            self.optim = get_optim(cfg, self.net, lr=cfg.optim.lr)
            self.loss_scaler = get_loss_scaler(cfg.trainer.scaler)
            self.loss_terms = get_loss_terms(cfg.loss.loss_terms, device=str(self.device))
            self.mixup_fn = Mixup(**cfg.trainer.mixup_kwargs) if cfg.trainer.mixup_kwargs['prob'] > 0 else None
            if cfg.trainer.scaler == 'apex':
                from apex import amp
                self.net, self.optim = amp.initialize(self.net, self.optim, opt_level='O1')
            self.scheduler = get_scheduler(cfg, self.optim)
            self.metrics = DeviceMetrics(('CE', 'KD') if 'KD' in self.loss_terms else ('CE',), self.device)
        if cfg.dist:
            if cfg.trainer.scaler == 'apex':
                from apex.parallel import DistributedDataParallel
                self.net = DistributedDataParallel(self.net, delay_allreduce=True)
            else:
                self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[cfg.local_rank], find_unused_parameters=cfg.trainer.find_unused_parameters)
        self.iter = 0
        self.epoch = state.get('epoch', 0) if not self.training else 0
        self.nan_or_inf_cnt = 0
        self.topk_recorder = {name: [] for name in ('net_top1', 'net_top5', 'net_E_top1', 'net_E_top5')}
        self.legacy_topk_recorder = None
        self.is_best = False
        self.best_metrics = BestValidationMetrics()
        self.best_checkpoint = None
        if cfg.resume_checkpoint and self.training and not cfg.ft:
            if cfg.model.model_kwargs['ema']:
                raise ValueError('Training resume must load net, not select EMA weights')
            if self.loss_scaler is None and state['scaler'] is not None:
                raise ValueError('Checkpoint AMP scaler requires matching AMP mode')
            self.iter, self.epoch = state['iter'], state['epoch']
            self.optim.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            if self.loss_scaler is not None:
                if state['scaler'] is None:
                    raise ValueError('Checkpoint scaler is incompatible with selected AMP')
                self.loss_scaler.load_state_dict(state['scaler'])
            if self.ema and state.get('net_E') is None:
                raise ValueError('EMA resume requires net_E in checkpoint')
            cfg.task_start_time = time.perf_counter() - state['total_time']
            self.nan_or_inf_cnt = state.get('nan_or_inf_cnt', 0)
            if state.get('selection_split') == 'val':
                self.topk_recorder = state['topk_recorder']
                self.legacy_topk_recorder = state.get('legacy_topk_recorder')
                self.best_metrics = BestValidationMetrics.from_checkpoint(state)
                if self.best_metrics.epoch is not None:
                    resumed_best = Path(checkpoint).parent / 'best_epoch{:03d}.pth'.format(
                        self.best_metrics.epoch)
                    if resumed_best.is_file():
                        self.best_checkpoint = str(resumed_best)
            else:
                self.legacy_topk_recorder = state['topk_recorder']
                self.topk_recorder = {name: [0.] * self.epoch for name in self.topk_recorder}
                self._log('Legacy test history retained separately; val best selection starts fresh')
            if self.iter % cfg.data.train_size:
                raise ValueError('Resume requires a complete epoch boundary with compatible loader size')
        self._log('pipeline_version=1; synthetic_smoke={}; logs flush at global period OR epoch end; train_reset_log_per does not clear metrics'.format(cfg.synthetic_smoke))
        if self.fold_context is not None:
            from .folds import format_fold_split
            self._log(format_fold_split(self.fold_context))
        log_cfg(cfg)

    def _log(self, message):
        if self.master:
            prefix = '[Fold {}/{}] '.format(self.fold_context.fold_index, self.fold_context.fold_count) if self.fold_context else ''
            self.logger.info(prefix + message)

    def _log_training_modules(self):
        model_kwargs = self.cfg.model.model_kwargs
        global_mode = model_kwargs['global_mode'].strip().lower()
        global_name = 'wavelet' if global_mode == 'wt' else global_mode
        self._log('Training modules: global={}, local={}'.format(
            global_name, model_kwargs['local_mode']))

    def _log_training_time(self, total_time):
        total_time_text = str(datetime.timedelta(seconds=int(total_time)))
        remaining_epochs = self.cfg.trainer.epoch_full - self.epoch
        eta_seconds = total_time / self.epoch * remaining_epochs
        eta_text = str(datetime.timedelta(seconds=int(eta_seconds)))
        self._log("==> Total time: {}\t Eta: {} \tLogged in '{}'".format(
            total_time_text, eta_text, self.cfg.logdir))

    def run(self):
        if not self.training:
            result = self.evaluate(self.cfg.mode, self.net)
            self._log('Evaluation effective samples: {}'.format(result.samples))
            self.record_metrics(result.split, result.model, self.epoch, {'CE': result.loss}, {'top1': result.top1, 'top5': result.top5})
            return
        self._log_training_modules()
        while self.epoch < self.cfg.trainer.epoch_full and self.iter < self.cfg.trainer.iter_full:
            result = self.train_epoch()
            if self.iter % len(self.bundle.loaders['train']):
                self._log('Iteration limit reached within epoch; no partial checkpoint saved')
                break
            self.epoch += 1
            self.record_metrics('train', 'net', self.epoch, {k: v['mean'] for k, v in result.losses.items()})
            self._log('Epoch {}: samples={}, seconds={:.3f}, images/s={:.3f}'.format(self.epoch, result.samples, result.seconds, result.samples / result.seconds))
            if True:
                for name, net in (('net', self.net), ('net_E', self.net_E)):
                    if net is None:
                        continue
                    result = self.evaluate('val', net)
                    self.record_metrics('val', name, self.epoch, {'CE': result.loss}, {'top1': result.top1, 'top5': result.top5})
                    if name == 'net':
                        self.is_best = self.best_metrics.update(
                            self.epoch, result.top1, result.top5)
                        self._log(self.best_metrics.validation_log(
                            result.top1, self.is_best))
                        if self.master:
                            self.writer.add_scalar(
                                'Val/net/best_top1', self.best_metrics.top1,
                                self.epoch)
                            self.writer.add_scalar(
                                'Val/net/top5_at_best', self.best_metrics.top5,
                                self.epoch)
                    self.topk_recorder[name + '_top1'].append(result.top1)
                    self.topk_recorder[name + '_top5'].append(result.top5)
                if self.net_E is None:
                    self.topk_recorder['net_E_top1'].append(0.)
                    self.topk_recorder['net_E_top5'].append(0.)
            else:
                for values in self.topk_recorder.values():
                    values.append(0.)
            self._log_training_time(time.perf_counter() - self.cfg.task_start_time)
            self.save_checkpoint()
        self._log(self.best_metrics.summary_log(self.best_checkpoint))
        if self.best_checkpoint is None or not Path(self.best_checkpoint).is_file():
            raise RuntimeError('Training completed without a best validation checkpoint')
        from .cross_validation import TrainingResult
        return TrainingResult(self.best_metrics.top1, self.best_metrics.top5,
                              self.best_metrics.epoch, self.best_checkpoint,
                              self.epoch, time.perf_counter() - self.cfg.task_start_time)

    def evaluate_best_on_test(self, best_checkpoint):
        state = torch.load(best_checkpoint, map_location='cpu')
        if self.fold_context is not None:
            if state.get('validation_fold') != self.fold_context.validation_fold:
                raise ValueError('Best checkpoint belongs to a different fold')
            if state.get('plan_digest') != self.fold_context.plan_digest or state.get('config_digest') != self.fold_context.config_digest:
                raise ValueError('Best checkpoint fold fingerprint mismatch')
        from util.net import trans_state_dict
        target = self.net.module if self.cfg.dist else self.net
        target.load_state_dict(state['net'], strict=self.cfg.model.model_kwargs['strict'])
        result = self.evaluate('test', self.net)
        self.record_metrics('test', 'net', self.epoch, {'CE': result.loss},
                            {'top1': result.top1, 'top5': result.top5})
        return result

    def check_bn(self):
        # The original model hook is optional; inspect its declared class API only.
        net = self.net.module if self.cfg.dist else self.net
        if callable(getattr(type(net), 'check_bn', None)):
            net.check_bn()

    def train_epoch(self):
        from util.net import distribute_bn
        self.net.train(self.cfg.mode == 'train')
        self.check_bn()
        loader = self.bundle.loaders['train']
        if self.cfg.dist or self.cfg.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        totals, samples = {}, 0
        for index, batch in enumerate(loader):
            self.scheduler.step(self.iter)
            losses = self.train_step(self.transfer_batch(batch))
            self.iter += 1
            samples += batch['img'].shape[0] * self.cfg.world_size
            self.metrics.update(losses, {name: torch.ones((), device=self.device) for name in losses})
            last = index + 1 == len(loader) or self.iter >= self.cfg.trainer.iter_full
            if self.iter % self.cfg.logging.train_log_per == 0 or last:
                snapshot = self.metrics.flush()
                for name, metric in snapshot.items():
                    total = totals.setdefault(name, {'sum': 0., 'count': 0.})
                    total['sum'] += metric['sum']
                    total['count'] += metric['count']
                    total['mean'] = total['sum'] / total['count']
                self._log('Train iter {}: {}'.format(self.iter, snapshot))
                if self.master:
                    for name, metric in snapshot.items():
                        self.writer.add_scalar('Train/window_' + name, metric['mean'], self.iter)
                    self.writer.add_scalar('Train/lr', self.optim.param_groups[0]['lr'], self.iter)
            if last:
                break
        if self.cfg.dist and self.dist_BN:
            distribute_bn(self.net, self.cfg.world_size, self.dist_BN)
            if self.net_E is not None:
                distribute_bn(self.net_E, self.cfg.world_size, self.dist_BN)
        if isinstance(self.optim, Lookahead):
            self.optim.sync_lookahead()
        torch.cuda.synchronize(self.device)
        return EpochResult(totals, samples, time.perf_counter() - start)

    def transfer_batch(self, batch):
        return {name: batch[name].to(self.device, non_blocking=self.cfg.trainer.data.non_blocking)
                for name in ('img', 'target', 'valid') if name in batch}

    def train_step(self, batch):
        images, targets = batch['img'], batch['target']
        if self.mixup_fn is not None:
            images, targets = self.mixup_fn(images, targets)
        with self.amp_autocast():
            outputs = self.net(images)
            if not isinstance(outputs, dict):
                outputs = {'out': outputs, 'out_kd': outputs}
            # Preserve the old per-step global nonfinite decision and anomaly update.
            invalid = (~torch.isfinite(outputs['out'])).any().float()
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(invalid, op=dist.ReduceOp.SUM)
            bad = invalid.item() > 0
            if bad:
                self.nan_or_inf_cnt += 1
                self._log('NaN or Inf Found, total {} times'.format(self.nan_or_inf_cnt))
                self.check_bn()
            losses = {'CE': self.loss_terms['CE'](outputs['out'], targets) if not bad else outputs['out'].new_zeros(())}
            if 'KD' in self.loss_terms:
                losses['KD'] = self.loss_terms['KD'](outputs['out_kd'], images) if not bad else outputs['out'].new_zeros(())
        loss = 0 * outputs['out'][0, 0] if bad else sum(losses.values())
        self.optim.zero_grad()
        if self.loss_scaler is not None:
            self.loss_scaler(loss, self.optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(), create_graph=self.cfg.loss.create_graph)
        else:
            loss.backward(retain_graph=self.cfg.loss.retain_graph)
            if self.cfg.loss.clip_grad is not None:
                dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
            self.optim.step()
        self.update_ema()
        return {name: value.detach() for name, value in losses.items()}

    @torch.no_grad()
    def update_ema(self):
        if not self.ema:
            return
        for ema_value, value in zip(self.net_E.state_dict().values(), self.net.state_dict().values()):
            if ema_value.dtype == torch.int64:
                continue
            ema_value.mul_(self.ema).add_(value, alpha=1. - self.ema)

    @torch.no_grad()
    def evaluate(self, split, net):
        if split not in ('val', 'test'):
            raise ValueError('Evaluation requires val or test')
        previous_mode = net.training
        metrics = DeviceMetrics(('loss', 'top1', 'top5', 'invalid'), self.device)
        net.eval()
        try:
            for batch in self.bundle.loaders[split]:
                batch = self.transfer_batch(batch)
                outputs = net(batch['img'])
                outputs = outputs['out'] if isinstance(outputs, dict) else outputs
                losses = F.cross_entropy(outputs, batch['target'], reduction='none')
                valid = batch['valid']
                finite = torch.isfinite(outputs).all(dim=1) & torch.isfinite(losses)
                correct = outputs.topk(min(5, outputs.shape[1]), dim=1).indices.eq(batch['target'][:, None])
                count = valid.sum()
                sums = {'loss': torch.where(valid & finite, losses, 0.).sum(),
                        'top1': (correct[:, 0] & valid).sum(),
                        'top5': (correct.any(dim=1) & valid).sum(),
                        'invalid': (valid & ~finite).sum()}
                metrics.update(sums, {'loss': count, 'top1': count, 'top5': count, 'invalid': count.new_ones(())})
            snapshot = metrics.flush()
            samples = int(snapshot['top1']['count'])
            if samples != self.bundle.lengths[split]:
                raise RuntimeError('Evaluation effective sample count mismatch')
            loss_valid = snapshot['invalid']['sum'] == 0
            return EvalResult(split, 'net_E' if net is self.net_E or (self.cfg.mode in ('val', 'test') and self.cfg.model.model_kwargs['ema']) else 'net',
                              snapshot['top1']['mean'] * 100, snapshot['top5']['mean'] * 100,
                              snapshot['loss']['mean'] if loss_valid else None, loss_valid, samples)
        finally:
            net.train(previous_mode)

    def record_metrics(self, split, model_name, epoch, losses, scores=None):
        if not self.master:
            return
        from trainer.loss_recorder import LossRecorder
        prefix = split.capitalize()
        if scores is None:
            score_log = 'None'
        else:
            score_log = '{' + ', '.join(
                '{}: {}'.format(repr(name), format(value, '.3f')
                                if name in ('top1', 'top5') else repr(value))
                for name, value in scores.items()) + '}'
        self._log('{} ({}) epoch {}: losses={}, scores={}'.format(
            prefix, model_name, epoch, losses, score_log))
        if all(value is not None and math.isfinite(value) for value in losses.values()):
            suffix = '_ema' if model_name == 'net_E' else ''
            key = split + '_loss' + suffix
            if key not in self.recorders:
                self.recorders[key] = LossRecorder(str(Path(self.cfg.logdir) / ('show_' + split)), key + '.json', self.logger)
            self.recorders[key].append(epoch, sum(losses.values()))
            for name, value in losses.items():
                self.writer.add_scalar('{}/{}/{}'.format(prefix, model_name, name), value, epoch)
        else:
            self._log('Nonfinite loss: diagnostic only, no loss JSON entry')
        for name, value in (scores or {}).items():
            self.writer.add_scalar('{}/{}/{}'.format(prefix, model_name, name), value, epoch)
        self.writer.flush()

    def save_checkpoint(self):
        if not self.training:
            raise RuntimeError('Evaluation cannot save a training checkpoint')
        if not self.master:
            self.is_best = False
            return None
        from util.net import trans_state_dict
        state = dict(net=trans_state_dict(self.net.state_dict(), dist=False),
                     net_E=self.net_E.state_dict() if self.net_E is not None else None,
                     optimizer=self.optim.state_dict(), scheduler=self.scheduler.state_dict(),
                     scaler=self.loss_scaler.state_dict() if self.loss_scaler else None,
                     iter=self.iter, epoch=self.epoch, topk_recorder=self.topk_recorder,
                     total_time=time.perf_counter() - self.cfg.task_start_time,
                     nan_or_inf_cnt=self.nan_or_inf_cnt, pipeline_version=1, selection_split='val',
                     class_to_idx=self.bundle.class_to_idx, synthetic_smoke=self.cfg.synthetic_smoke,
                     legacy_topk_recorder=self.legacy_topk_recorder,
                     **self.best_metrics.checkpoint_fields())
        if self.fold_context is not None:
            state.update(validation_fold=self.fold_context.validation_fold,
                         fold_index=self.fold_context.fold_index,
                         plan_digest=self.fold_context.plan_digest,
                         config_digest=self.fold_context.config_digest)
        root = Path(self.cfg.logdir)
        path = root / 'latest_ckpt.pth'
        torch.save(state, path)
        if self.is_best:
            for old in root.glob('best_epoch*.pth'):
                old.unlink()
            best_path = root / 'best_epoch{:03d}.pth'.format(self.epoch)
            shutil.copyfile(path, best_path)
            self.best_checkpoint = str(best_path)
            self._log('Best checkpoint saved: {}'.format(best_path))
        if self.epoch % self.cfg.trainer.save_per_epoch == 0:
            shutil.copyfile(path, root / 'ckpt_{}.pth'.format(self.epoch))
        self.is_best = False
        return str(path)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.bundle.loaders.clear()
        if self.master:
            try:
                for key, recorder in self.recorders.items():
                    recorder.plot(recorder.checkpoint_dir, key + '.jpg', key, max_epoch=None)
                    recorder.export_txt(str(Path(recorder.checkpoint_dir) / (key + '.txt')), max_epoch=None)
            finally:
                self.writer.close()
