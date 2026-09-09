"""Best validation metrics selected by Top-1 accuracy."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class BestValidationMetrics:
    top1: Optional[float] = None
    top5: Optional[float] = None
    epoch: Optional[int] = None

    def update(self, epoch, top1, top5):
        if self.top1 is not None and top1 <= self.top1:
            return False
        self.top1 = top1
        self.top5 = top5
        self.epoch = epoch
        return True

    @classmethod
    def from_checkpoint(cls, state):
        best_fields = ('best_top1', 'best_top5', 'best_epoch')
        if all(name in state for name in best_fields):
            values = tuple(state[name] for name in best_fields)
            if all(value is None for value in values):
                return cls()
            if any(value is None for value in values):
                raise ValueError('Checkpoint best validation metrics are incomplete')
            return cls(top1=values[0], top5=values[1], epoch=values[2])

        top1 = state.get('best_top1')
        if top1 is None:
            history = state['topk_recorder']['net_top1']
            if not history:
                return cls()
            top1 = max(history)

        top1_history = state['topk_recorder']['net_top1']
        if not top1_history:
            raise ValueError('Checkpoint best_top1 has no matching validation history')
        best_index = max(range(len(top1_history)), key=top1_history.__getitem__)
        if top1_history[best_index] != top1:
            raise ValueError('Checkpoint best_top1 does not match validation history')
        top5_history = state['topk_recorder']['net_top5']
        if len(top5_history) != len(top1_history):
            raise ValueError('Checkpoint Top-1 and Top-5 histories have different lengths')
        return cls(top1=top1, top5=top5_history[best_index], epoch=best_index + 1)

    def checkpoint_fields(self):
        return {
            'best_top1': self.top1,
            'best_top5': self.top5,
            'best_epoch': self.epoch,
        }

    def validation_log(self, current_top1, improved):
        if self.top1 is None or self.top5 is None or self.epoch is None:
            raise RuntimeError('Best validation metrics are unavailable')
        return (
            'Val best: current_top1={}, best_top1={}, top5_at_best={}, '
            'best_epoch={}, improved={}'
        ).format(current_top1, self.top1, self.top5, self.epoch, improved)

    def summary_log(self, best_checkpoint=None):
        if self.top1 is None:
            return 'Training summary: no validation result'
        if self.top5 is None or self.epoch is None:
            raise RuntimeError('Best validation metrics are incomplete')
        return (
            'Training summary: best_val_top1={}, top5_at_best={}, '
            'best_epoch={}, best_checkpoint={}'
        ).format(self.top1, self.top5, self.epoch,
                 best_checkpoint or 'unavailable')
