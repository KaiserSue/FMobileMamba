"""Public entry point for the independent classification pipeline."""
__all__ = ['run']


def run(args):
    import os
    import time
    from pathlib import Path
    import torch
    import torch.distributed as dist
    from .config import load_config
    cfg = load_config(args, Path(__file__).resolve().parent.parent)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: real FMobileMamba training/evaluation cannot run; synthetic smoke also requires CUDA')
    from util.net import init_training
    from util.util import run_pre, init_checkpoint
    trainer = None
    writer = None
    try:
        run_pre(cfg)
        init_training(cfg)
        cfg.nnodes = max(1, cfg.world_size // int(os.environ.get('LOCAL_WORLD_SIZE', cfg.world_size)))
        cfg.resume_checkpoint = ''
        cfg.logger = cfg.writer = cfg.logdir = None
        if cfg.mode in ('train', 'ft'):
            from .cross_validation import CrossValidationRunner
            CrossValidationRunner(cfg).run()
        else:
            from .trainer import FastCLSTrainer
            cfg.trainer.checkpoint = str(Path(cfg.trainer.checkpoint) / ('run_' + str(time.time_ns()))) if cfg.master else cfg.trainer.checkpoint
            if cfg.master:
                init_checkpoint(cfg, datefmt='%Y-%m-%d %H:%M:%S')
                writer = cfg.writer
            if dist.is_initialized():
                values = [cfg.logdir]
                dist.broadcast_object_list(values, src=cfg.logger_rank)
                cfg.logdir = values[0]
            trainer = FastCLSTrainer(cfg)
            trainer.run()
    finally:
        try:
            if trainer is not None:
                trainer.close()
            elif writer is not None:
                writer.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
