"""Native trainer with the first zero-score validation edge fixed."""
from pathlib import Path

from trainer.cls import CLSTrainer


class NativeCVTrainer(CLSTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._real_validation_count = 0

    def test(self):
        super().test()
        self._real_validation_count += 1
        if (self._real_validation_count == 1 and not self.is_best
                and not list(Path(self.cfg.logdir).glob('best_epoch*.pth'))):
            self.is_best = True
