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
    from .trainer import FastCLSTrainer
    trainer = None
    writer = None
    try:
        run_pre(cfg)
        init_training(cfg)
        cfg.nnodes = max(1, cfg.world_size // int(os.environ.get('LOCAL_WORLD_SIZE', cfg.world_size)))
        # Read a resume source, but always write to a fresh run directory.
        cfg.resume_checkpoint = ''
        if cfg.trainer.resume_dir:
            source = Path(cfg.trainer.resume_dir)
            if not source.is_absolute():
                source = Path(cfg.trainer.checkpoint) / source
            cfg.resume_checkpoint = str(source / (Path(cfg.model.model_kwargs['checkpoint_path']).name or 'latest_ckpt.pth'))
            cfg.model.model_kwargs['checkpoint_path'] = cfg.resume_checkpoint
            cfg.trainer.resume_dir = ''
        # Give every run a unique root, including repeated eval invocations in one second.
        cfg.trainer.checkpoint = str(Path(cfg.trainer.checkpoint) / ('run_' + str(time.time_ns()))) if cfg.master else cfg.trainer.checkpoint
        if cfg.master:
            init_checkpoint(cfg, datefmt='%Y-%m-%d %H:%M:%S')
            writer = cfg.writer
        else:
            cfg.logdir = cfg.logger = cfg.writer = None
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
