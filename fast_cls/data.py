"""Requested split construction; no implicit production-data fallback."""
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


@dataclass
class LoaderBundle:
    loaders: dict
    lengths: dict
    class_to_idx: dict
    sample_ids: dict


class FoldSubsetDataset(Dataset):
    """An exact, ordered sample view over an ImageFolder-style dataset."""
    def __init__(self, dataset, sample_ids, dataset_root):
        requested = set(sample_ids)
        if not requested:
            raise ValueError('Fold sample IDs must be nonempty')
        root = Path(dataset_root).resolve()
        index_by_id = {}
        for index, (path, _) in enumerate(dataset.samples):
            sample_id = Path(path).resolve().relative_to(root).as_posix()
            if sample_id in index_by_id:
                raise ValueError('Duplicate dataset sample ID: ' + sample_id)
            index_by_id[sample_id] = index
        unknown = requested - set(index_by_id)
        if unknown:
            raise ValueError('Unknown fold sample IDs: ' + ', '.join(sorted(unknown)[:5]))
        self.dataset = dataset
        self.indices = tuple(index for sample_id, index in index_by_id.items() if sample_id in requested)
        self.sample_ids = tuple(Path(dataset.samples[index][0]).resolve().relative_to(root).as_posix()
                                for index in self.indices)
        self.samples = tuple(dataset.samples[index] for index in self.indices)
        self.class_to_idx = dict(dataset.class_to_idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


class SyntheticDataset(Dataset):
    def __init__(self, split, length, size, num_classes, seed):
        if split not in ('train', 'val', 'test') or min(length, size, num_classes) <= 0:
            raise ValueError('Invalid synthetic dataset specification')
        self.length, self.size, self.num_classes = length, size, num_classes
        self.seed = seed + {'train': 0, 'val': 1000000, 'test': 2000000}[split]
        self.class_to_idx = {str(i): i for i in range(num_classes)}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        return {'img': torch.randn(3, self.size, self.size, generator=generator),
                'target': torch.randint(self.num_classes, (), generator=generator)}


class MaskedEvalDataset(Dataset):
    def __init__(self, dataset, world_size):
        if len(dataset) == 0 or world_size <= 0:
            raise ValueError('Evaluation dataset and world size must be positive')
        self.dataset = dataset
        self.length = ((len(dataset) + world_size - 1) // world_size) * world_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        valid = index < len(self.dataset)
        return dict(self.dataset[index if valid else 0], valid=valid)


def build_loaders(cfg, requested_splits, expected_class_to_idx=None,
                  fold_plan=None, validation_fold=None):
    if not requested_splits or len(set(requested_splits)) != len(requested_splits):
        raise ValueError('Request distinct, nonempty splits')
    loaders, lengths, sample_ids = {}, {}, {}
    mapping = expected_class_to_idx
    td = cfg.trainer.data
    for split in requested_splits:
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown split: ' + split)
        train = split == 'train'
        if cfg.synthetic_smoke:
            dataset = SyntheticDataset(split, 8 if train else 5, cfg.size, cfg.data.nb_classes, cfg.seed)
        else:
            if cfg.data.type != 'DefaultCLS':
                raise ValueError('Fast pipeline supports DefaultCLS only')
            from data.CLS_dataset import DefaultCLS
            from data.utils import get_transforms
            transform = get_transforms(cfg, train, cfg.data.train_transforms if train else cfg.data.test_transforms)
            physical_split = 'train' if fold_plan is not None and split in ('train', 'val') else split
            dataset = DefaultCLS(cfg, train=physical_split == 'train', transform=transform,
                                 subset=vars(cfg.data)[physical_split + '_subdir'])
            if split in ('train', 'val') and fold_plan is not None:
                if validation_fold is None:
                    raise ValueError('validation_fold is required with a fold plan')
                from pathlib import Path
                from .folds import fold_sample_ids
                train_ids, val_ids = fold_sample_ids(fold_plan, validation_fold)
                selected = train_ids if split == 'train' else val_ids
                dataset = FoldSubsetDataset(dataset, selected, Path(cfg.data.root_dir))
        if not len(dataset):
            raise ValueError('Empty split: ' + split)
        if len(dataset.class_to_idx) != cfg.data.nb_classes or sorted(dataset.class_to_idx.values()) != list(range(cfg.data.nb_classes)):
            raise ValueError('Class count/mapping mismatch in ' + split)
        if mapping is not None and mapping != dataset.class_to_idx:
            raise ValueError('Class mapping mismatch in ' + split)
        mapping = dict(dataset.class_to_idx)
        lengths[split] = len(dataset)
        sample_ids[split] = tuple(getattr(dataset, 'sample_ids', ()))
        if not train:
            dataset = MaskedEvalDataset(dataset, cfg.world_size)
        sampler = None
        if cfg.dist or cfg.world_size > 1:
            if train and cfg.data.sampler == 'ra':
                from timm.data.distributed_sampler import RepeatAugSampler
                sampler = RepeatAugSampler(dataset, num_replicas=cfg.world_size, rank=cfg.rank, shuffle=True)
            elif not train or cfg.data.sampler == 'naive':
                sampler = DistributedSampler(dataset, num_replicas=cfg.world_size, rank=cfg.rank, shuffle=train, drop_last=False)
            else:
                raise ValueError('Unsupported training sampler: ' + cfg.data.sampler)
        workers = td.num_workers_per_gpu if train else td.num_workers_per_gpu_eval
        kwargs = dict(num_workers=workers, pin_memory=td.pin_memory,
                      persistent_workers=bool(workers and td.persistent_workers))
        if workers:
            kwargs['prefetch_factor'] = td.prefetch_factor if train else td.prefetch_factor_eval
        loader = DataLoader(dataset, batch_size=td.batch_size_per_gpu if train else td.batch_size_per_gpu_test,
                            shuffle=train and sampler is None, sampler=sampler,
                            drop_last=td.drop_last if train else False, **kwargs)
        if len(loader) == 0:
            raise ValueError('Zero batches for split ' + split + '; reduce batch size or disable drop_last')
        loaders[split] = loader
    return LoaderBundle(loaders, lengths, mapping, sample_ids)
