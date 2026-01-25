import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pickle
import copy
import random
import numpy as np

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS
# from util.data import get_img_loader
from data.utils import get_transforms, get_scales
from . import DATA
import torch
import json


@DATA.register_module
class ImageFolderLMDB(data.Dataset):
	def __init__(self, cfg, train=True, transform=None, target_transform=None, subset=None):
		self.cfg = cfg
		self.train = train
		self.transform = transform
		self.target_transform = target_transform
		# scale_kwargs = cfg.trainer.scale_kwargs
		# if scale_kwargs is not None and scale_kwargs['n_scale'] > 0:
		# 	scale_kwargs = {k: v for k, v in scale_kwargs.items()}
		# 	self.scales = get_scales(**scale_kwargs)
		# else:
		# 	self.scales = [(cfg.size, cfg.size)]
		# self.num = 0
		# self.batch_size_per_gpu = cfg.trainer.data.batch_size_per_gpu
		
		self.loader = pickle.loads
		root_dir = getattr(cfg.data, 'root_dir', cfg.data.root)
		train_subdir = getattr(cfg.data, 'train_subdir', 'train')
		val_subdir = getattr(cfg.data, 'val_subdir', 'val')
		split = subset if subset is not None else (train_subdir if train else val_subdir)
		db_path = '{}/{}.lmdb'.format(root_dir, split)
		self.env = lmdb.open(db_path, subdir=osp.isdir(db_path), readonly=True, lock=False, readahead=False, meminit=False)
		self.txn = self.env.begin(write=False)
		self.length = pickle.loads(self.txn.get(b'__len__'))
		self.keys = pickle.loads(self.txn.get(b'__keys__'))
	
	# def reset_scale_transform(self):
	# 	scale_rand = random.choices(self.scales, k=1)[0]
	# 	scale_rand = scale_rand[0]
	# 	self.cfg.size = scale_rand
	# 	self.cfg.data.train_transforms[0]['input_size'] = scale_rand
	# 	self.transform = get_transforms(self.cfg, train=True, cfg_transforms=self.cfg.data.train_transforms)
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, index):
		# if len(self.scales) > 1 and self.num % self.batch_size_per_gpu == 0:
		# 	self.reset_scale_transform()
		# self.num += 1
		byteflow = self.txn.get(self.keys[index])
		imgbuf, target = self.loader(byteflow)
		buf = six.BytesIO()
		buf.write(imgbuf)
		buf.seek(0)
		img = Image.open(buf).convert('RGB')
		img = self.transform(img) if self.transform is not None else img
		target = self.target_transform(target) if self.target_transform is not None else target
		return {'img': img, 'target':target}


@DATA.register_module
class CustomImageDataset(data.Dataset):
	def __init__(self, cfg, train=True, transform=None, target_transform=None):
		"""
        Args:
            root_dir (string): Directory with all the images and json files.
            train (bool): If True, load train.json, else load val.json.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_transform (callable, optional): Optional transform to be applied
                on the target.
        """
		self.root_dir = cfg.data.root
		self.train = train
		self.transform = transform
		self.target_transform = target_transform
		# Determine which json file to load
		json_file = 'train.json' if train else 'val.json'
		json_path = os.path.join(self.root_dir, json_file)
		# Load the json file
		with open(json_path, 'r') as f:
			self.annotations = json.load(f)
		# Create a list of image file names and their corresponding labels
		self.img_labels = list(self.annotations.items())
		self.length = len(self.img_labels)

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name, label = self.img_labels[idx]
		img_path = os.path.join(self.root_dir, 'images', img_name)
		image = Image.open(img_path).convert('RGB')
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		sample = {'img': image, 'target': label}
		return sample

def folder2lmdb(root, name="train", write_frequency=1000):
	# https://github.com/xunge/pytorch_lmdb_imagenet/blob/master/folder2lmdb.py
	def raw_reader(path):
		with open(path, 'rb') as f:
			bin_data = f.read()
		return bin_data
	
	img_dir = f'{root}/{name}'
	dataset = ImageFolder(root=img_dir, loader=raw_reader)
	data_loader = DataLoader(dataset, num_workers=32, collate_fn=lambda x: x)
	
	lmdb_path = osp.join(root, f'{name}.lmdb')
	db = lmdb.open(lmdb_path, subdir=True, map_size=1099511627776 * 2, readonly=False, meminit=False, map_async=True)
	txn = db.begin(write=True)
	for idx, data in enumerate(data_loader):
		image, label = data[0]
		txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((image, label)))
		if (idx + 1) % write_frequency == 0:
			print(f'{name} {idx + 1}/{len(data_loader)}')
			txn.commit()
			txn = db.begin(write=True)
	txn.commit()
	keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
	txn = db.begin(write=True)
	txn.put(b'__keys__', pickle.dumps(keys))
	txn.put(b'__len__', pickle.dumps(len(keys)))
	txn.commit()
	db.sync()
	db.close()


		
