"""Sequential five-fold training orchestration and recovery."""
import copy
import dataclasses
import datetime
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .folds import (FoldContext, VALIDATION_ORDER, atomic_write_json,
                    FoldPlan,
                    build_config_fingerprint, load_or_create_fold_plan,
                    load_or_create_run_state, save_run_state)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TrainingResult:
    best_val_top1: float
    best_val_top5: float
    best_epoch: int
    best_checkpoint: str
    epochs_completed: int
    duration_seconds: float


@dataclass(frozen=True)
class FoldResult:
    fold_index: int
    validation_fold: str
    best_val_top1: float
    best_val_top5: float
    best_epoch: int
    best_checkpoint: str
    epochs_completed: int
    test_top1: float
    test_top5: float
    test_samples: int
    fold_duration_seconds: float
    completed_at: str


@dataclass(frozen=True)
class CrossValidationSummary:
    fold_results: tuple
    mean_top1: float
    mean_top5: float
    total_duration_seconds: float
    plan_digest: str
    config_digest: str
    completed_at: str
    run_dir: str

    def to_dict(self):
        value = dataclasses.asdict(self)
        value["fold_results"] = [dataclasses.asdict(item) for item in self.fold_results]
        return value


class CrossValidationRunner:
    def __init__(self, cfg, trainer_factory=None, config_fingerprint=None):
        self.cfg = cfg
        self.master = cfg.master
        self.trainer_factory = trainer_factory
        self.config_snapshot, self.config_digest = (config_fingerprint or build_config_fingerprint(cfg))
        resume = bool(cfg.trainer.resume_dir)
        if resume:
            candidate = Path(cfg.trainer.resume_dir)
            self.run_dir = candidate.resolve() if candidate.is_absolute() else (Path(cfg.trainer.checkpoint) / candidate).resolve()
            if not self.run_dir.is_dir():
                raise ValueError("Cross-validation resume directory does not exist: " + str(self.run_dir))
        else:
            self.run_dir = (Path(cfg.trainer.checkpoint) / ("cv_" + str(time.time_ns()))).resolve()
            if self.master:
                self.run_dir.mkdir(parents=True)
        import torch.distributed as dist
        if dist.is_initialized():
            values = [str(self.run_dir) if self.master else None]
            dist.broadcast_object_list(values, src=cfg.logger_rank)
            self.run_dir = Path(values[0])
        self.resume = resume

    def _log(self, message):
        logger = getattr(self.cfg, "logger", None)
        if self.master and logger is not None:
            logger.info(message)

    def _fold_config(self, context, resume_checkpoint):
        cfg = copy.deepcopy(self.cfg)
        cfg.fold_context = context
        cfg.fold_plan = self.plan
        cfg.logdir = context.fold_dir
        cfg.resume_checkpoint = str(resume_checkpoint) if resume_checkpoint else ""
        if hasattr(cfg, "model"):
            cfg.model.model_kwargs["checkpoint_path"] = cfg.resume_checkpoint
        cfg.task_start_time = time.perf_counter()
        if self.trainer_factory is not None:
            return cfg
        if not self.master:
            cfg.logger = cfg.writer = None
            return cfg
        from tensorboardX import SummaryWriter
        from util.util import get_logger
        cfg.logger = get_logger(cfg, datefmt="%Y-%m-%d %H:%M:%S")
        cfg.writer = SummaryWriter(log_dir=cfg.logdir, comment="")
        return cfg

    def _run_fold(self, index, validation_fold, plan, state):
        fold_dir = self.run_dir / "fold_{}_{}".format(index, validation_fold)
        fold_dir.mkdir(exist_ok=True)
        context = FoldContext(index, 5, validation_fold,
                              tuple(fold for fold in plan.folds if fold != validation_fold),
                              str(fold_dir), plan.plan_digest, self.config_digest)
        latest = fold_dir / "latest_ckpt.pth"
        resume_checkpoint = latest if latest.is_file() else None
        state["active_fold"] = validation_fold
        if self.master:
            save_run_state(self.run_dir / "state.json", state)
        cfg = self._fold_config(context, resume_checkpoint)
        factory = self.trainer_factory
        if factory is None:
            from .trainer import FastCLSTrainer
            factory = FastCLSTrainer
        trainer = factory(cfg, context, resume_checkpoint)
        started = time.perf_counter()
        try:
            training = trainer.run()
            evaluation = trainer.evaluate_best_on_test(training.best_checkpoint)
        finally:
            trainer.close()
        duration = time.perf_counter() - started
        return FoldResult(index, validation_fold, training.best_val_top1,
                          training.best_val_top5, training.best_epoch,
                          training.best_checkpoint, training.epochs_completed,
                          evaluation.top1, evaluation.top5, evaluation.samples,
                          duration, _utc_now())

    def run(self):
        plan_path = Path(self.cfg.trainer.fold_plan_path) if self.cfg.trainer.fold_plan_path else self.run_dir / "fold_plan.json"
        import torch.distributed as dist
        if self.master:
            plan = load_or_create_fold_plan(self.cfg.data.root_dir, plan_path, self.cfg.seed,
                                            tuple(self.cfg.trainer.fold_order))
            plan_value = plan.to_dict()
        else:
            plan_value = None
        if dist.is_initialized():
            values = [plan_value]
            dist.broadcast_object_list(values, src=self.cfg.logger_rank)
            plan_value = values[0]
        plan = FoldPlan.from_dict(plan_value)
        self.plan = plan
        state_path = self.run_dir / "state.json"
        if self.master:
            state = load_or_create_run_state(state_path, plan, self.config_snapshot,
                                             self.config_digest, self.resume)
        else:
            state = None
        if dist.is_initialized():
            values = [state]
            dist.broadcast_object_list(values, src=self.cfg.logger_rank)
            state = values[0]
        started = time.perf_counter()
        try:
            for index, validation_fold in enumerate(VALIDATION_ORDER, 1):
                if validation_fold in state["completed_folds"]:
                    self._log("[Fold {}/5] completed; skipping".format(index))
                    continue
                result = self._run_fold(index, validation_fold, plan, state)
                state["fold_results"][validation_fold] = dataclasses.asdict(result)
                state["completed_folds"].append(validation_fold)
                state["active_fold"] = None
                state["accumulated_duration_seconds"] += result.fold_duration_seconds
                if self.master:
                    save_run_state(state_path, state)
                elapsed = state["accumulated_duration_seconds"]
                eta = elapsed / len(state["completed_folds"]) * (5 - len(state["completed_folds"]))
                self._log("[Fold {}/5] seconds={:.3f}, fold ETA={:.3f}, total={:.3f}".format(
                    index, result.fold_duration_seconds, eta, elapsed))
            ordered = tuple(FoldResult(**state["fold_results"][fold]) for fold in VALIDATION_ORDER)
            if any(not Path(item.best_checkpoint).is_file() for item in ordered):
                raise RuntimeError("A fold best checkpoint is missing")
            total = state["accumulated_duration_seconds"]
            summary = CrossValidationSummary(
                ordered, sum(item.test_top1 for item in ordered) / 5,
                sum(item.test_top5 for item in ordered) / 5, total,
                plan.plan_digest, self.config_digest, _utc_now(), str(self.run_dir))
            if self.master:
                atomic_write_json(self.run_dir / "summary.json", summary.to_dict())
            state["status"] = "completed"
            state["active_fold"] = None
            if self.master:
                save_run_state(state_path, state)
            self._log("Five-fold training completed in {:.3f}s: test top1={:.3f}, top5={:.3f}".format(
                time.perf_counter() - started, summary.mean_top1, summary.mean_top5))
            return summary
        except BaseException as error:
            state["status"] = "failed"
            state["last_error"] = "{}: {}".format(type(error).__name__, error)
            if self.master:
                save_run_state(state_path, state)
            raise
