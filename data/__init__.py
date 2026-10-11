import glob
import importlib
import inspect
import os
from data.split_selection import require_val_subdir, select_test_subdir
import torch
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torchvision import datasets
from torchvision import transforms as tr
from util.registry import Registry
from timm.data.distributed_sampler import RepeatAugSampler
#from ipmix import IPMixDataset
TRANSFORMS = Registry('Transforms')
DATA = Registry('Data')
files = glob.glob('data/[!_]*.py')
for file in files:
	model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))

from data.utils import get_transforms


def _get_split_dirs(cfg):
	root_dir = getattr(cfg.data, 'root_dir', cfg.data.root)
	train_subdir = getattr(cfg.data, 'train_subdir', 'train')
	val_subdir = getattr(cfg.data, 'val_subdir', 'val')
	test_subdir = getattr(cfg.data, 'test_subdir', 'test')
	return root_dir, train_subdir, val_subdir, test_subdir


def _build_dataset(cfg, train, transform, subset):
	dataset_builder = DATA.get_module(cfg.data.type)
	if inspect.isclass(dataset_builder):
		params = inspect.signature(dataset_builder.__init__).parameters
	else:
		params = inspect.signature(dataset_builder).parameters
	call_kwargs = {'cfg': cfg, 'train': train}
	if 'subset' in params:
		call_kwargs['subset'] = subset
	if 'split' in params and 'subset' not in params:
		call_kwargs['split'] = subset
	if 'transform' in params:
		call_kwargs['transform'] = transform
	elif 'transforms' in params:
		call_kwargs['transforms'] = transform
	if 'target_transform' in params:
		call_kwargs['target_transform'] = None
	return dataset_builder(**call_kwargs)


def get_dataset(cfg):
	train_transforms = get_transforms(cfg, train=True, cfg_transforms=cfg.data.train_transforms)
	eval_transforms = get_transforms(cfg, train=False, cfg_transforms=cfg.data.test_transforms)
	root_dir, train_subdir, val_subdir, test_subdir = _get_split_dirs(cfg)
	if cfg.data.type == 'DefaultCLS':
		cfg.data.root_dir = root_dir
		cfg.data.train_subdir, cfg.data.val_subdir, cfg.data.test_subdir = train_subdir, val_subdir, test_subdir
	train_set = _build_dataset(cfg, train=True, transform=train_transforms, subset=train_subdir)
	if cfg.data.type == 'DefaultCLS' and cfg.mode in ('train', 'ft'):
		require_val_subdir(root_dir, val_subdir)
		test_set = _build_labeled_val_dataset(cfg, train_set, eval_transforms, val_subdir)
	elif cfg.data.type == 'DefaultCLS':
		selected = select_test_subdir(root_dir, test_subdir, val_subdir)
		test_set = _build_dataset(cfg, train=False, transform=eval_transforms, subset=selected)
	else:
		test_set = _build_dataset(cfg, train=False, transform=eval_transforms, subset=test_subdir)
	return train_set, test_set


def _build_labeled_val_dataset(cfg, train_set, transform, val_subdir):
	from data.CLS_dataset import AnnotatedValCLS
	root_dir = cfg.data.root_dir
	val_root = os.path.join(root_dir, val_subdir)
	annotation_file = os.path.join(val_root, 'val_annotations.txt')
	if os.path.exists(annotation_file):
		val_set = AnnotatedValCLS(val_root, train_set.class_to_idx, cfg.data.loader_type, transform)
	else:
		val_set = _build_dataset(cfg, train=False, transform=transform, subset=val_subdir)
	if len(train_set.class_to_idx) != cfg.data.nb_classes:
		raise ValueError(f'Training class count {len(train_set.class_to_idx)} does not match configured {cfg.data.nb_classes}')
	if val_set.class_to_idx != train_set.class_to_idx:
		raise ValueError(f'Validation class mapping differs from training: {val_root}')
	if len(val_set) == 0:
		raise ValueError(f'Validation set has no samples: {val_root}')
	return val_set


def get_loader(cfg):
	train_set, test_set = get_dataset(cfg)
	if cfg.dist:
		if cfg.data.sampler == 'naive':
			sampler = DistributedSampler
			train_sampler = sampler(train_set, shuffle=True)
		elif cfg.data.sampler == 'ra':
			train_sampler = RepeatAugSampler(train_set, shuffle=True)
		else:
			raise NotImplementedError("sampler '{}' is not implemented".format(cfg.data.sampler))
		test_sampler = DistributedSampler(test_set, shuffle=False)
	else:
		train_sampler, test_sampler = None, None
	train_loader = torch.utils.data.DataLoader(dataset=train_set,
											   batch_size=cfg.trainer.data.batch_size_per_gpu,
											   shuffle=(train_sampler is None),
											   sampler=train_sampler,
											   num_workers=cfg.trainer.data.num_workers_per_gpu,
											   pin_memory=cfg.trainer.data.pin_memory,
											   drop_last=cfg.trainer.data.drop_last,
											   persistent_workers=cfg.trainer.data.persistent_workers)
	test_loader = torch.utils.data.DataLoader(dataset=test_set,
											  batch_size=cfg.trainer.data.batch_size_per_gpu_test,
											  shuffle=False,
											  sampler=test_sampler,
											  num_workers=cfg.trainer.data.num_workers_per_gpu,
											  pin_memory=cfg.trainer.data.pin_memory,
											  drop_last=False,
											  persistent_workers=cfg.trainer.data.persistent_workers)
	return train_loader, test_loader
