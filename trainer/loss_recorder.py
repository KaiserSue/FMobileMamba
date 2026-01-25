import json
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


class LossRecorder:
    def __init__(self, checkpoint_dir: str, filename: str = 'train_loss.json', logger=None):
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.json_path = os.path.join(checkpoint_dir, filename)
        self.records: Dict[str, float] = self.load_history()

    def load_history(self) -> Dict[str, float]:
        if not os.path.exists(self.json_path):
            self.records = {}
            return self.records
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        self.records = {str(k): float(v) for k, v in data.items()}
        return self.records

    def record_train_loss(self, epoch: int, loss: float, cutoff_epoch: int) -> bool:
        if not math.isfinite(loss):
            raise ValueError(f'Invalid loss value: {loss}')
        if epoch >= cutoff_epoch:
            self.truncate_after(cutoff_epoch)
            return False
        self.records = self.load_history()
        epoch_key = str(int(epoch))
        self.records[epoch_key] = round(float(loss), 4)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        with open(self.json_path, 'w') as f:
            json.dump(self.records, f)
        return True

    def truncate_after(self, cutoff_epoch: int) -> Dict[str, float]:
        self.records = self.load_history()
        filtered = {k: v for k, v in self.records.items() if int(k) < cutoff_epoch}
        if filtered != self.records:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            with open(self.json_path, 'w') as f:
                json.dump(filtered, f)
        self.records = filtered
        return self.records

    def get_sorted_items(self, max_epoch: Optional[int]) -> List[Tuple[int, float]]:
        self.records = self.load_history()
        items = self.records.items()
        if max_epoch is not None:
            items = filter(lambda kv: int(kv[0]) < max_epoch, items)
        return sorted(((int(k), v) for k, v in items), key=lambda kv: kv[0])

    def plot(self, loss_dir: str, loss_filename: str, title: str, max_epoch: Optional[int] = None) -> Optional[str]:
        sorted_items = self.get_sorted_items(max_epoch)
        if len(sorted_items) == 0:
            return None
        epochs = [epoch for epoch, _ in sorted_items]
        losses = [loss for _, loss in sorted_items]
        os.makedirs(loss_dir, exist_ok=True)
        plt.figure()
        plt.plot(epochs, losses, color='blue')
        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        output_path = os.path.join(loss_dir, loss_filename)
        plt.savefig(output_path, format='jpg')
        plt.close()
        return output_path

    def export_txt(self, txt_path: str, max_epoch: Optional[int]) -> None:
        sorted_items = self.get_sorted_items(max_epoch)
        dir_path = os.path.dirname(txt_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(txt_path, 'w') as f:
            for epoch, loss in sorted_items:
                f.write(f'{epoch}, {loss}\n')

    def append(self, epoch: int, loss: float) -> None:
        self.record_train_loss(epoch, loss, cutoff_epoch=math.inf)
