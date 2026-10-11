"""Fold views over the physical training split for the native trainer."""
import json
from pathlib import Path

from torch.utils.data import Dataset

from native_cv.folds import FoldPlan, fold_sample_ids
from .CLS_dataset import DefaultCLS
from . import DATA


@DATA.register_module
class NativeCVCLS(Dataset):
    def __init__(self, cfg, train=True, transform=None, subset=None):
        cv = cfg.native_cv
        plan = FoldPlan.from_dict(json.loads(Path(cv.plan_path).read_text(encoding='utf-8')))
        if plan.plan_digest != cv.plan_digest:
            raise ValueError('Fold plan digest mismatch')
        if plan.class_to_idx != cv.expected_class_to_idx:
            raise ValueError('Fold class mapping mismatch')
        root = Path(cfg.data.root_dir).resolve()
        dataset = DefaultCLS(cfg, train=True, transform=transform,
                             subset=cfg.data.train_subdir)
        if dataset.class_to_idx != plan.class_to_idx or len(dataset.class_to_idx) != cfg.data.nb_classes:
            raise ValueError('Physical train class mapping mismatch')
        train_ids, val_ids = fold_sample_ids(plan, cv.validation_fold)
        selected = set(train_ids if train else val_ids)
        by_id = {}
        for index, (path, _) in enumerate(dataset.samples):
            sample_id = Path(path).resolve().relative_to(root).as_posix()
            if sample_id in by_id:
                raise ValueError('Duplicate physical training sample: ' + sample_id)
            by_id[sample_id] = index
        if set(by_id) != {item.sample_id for item in plan.assignments}:
            raise ValueError('Physical training samples differ from fold plan')
        if not selected or selected - set(by_id):
            raise ValueError('Empty or unknown fold sample IDs')
        self.dataset = dataset
        self.sample_ids = tuple(sample_id for sample_id in by_id if sample_id in selected)
        self.indices = tuple(by_id[sample_id] for sample_id in self.sample_ids)
        self.samples = tuple(dataset.samples[index] for index in self.indices)
        self.class_to_idx = dict(dataset.class_to_idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
