import importlib.util
import os
import tempfile
import unittest
from argparse import Namespace

from PIL import Image

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _build_cfg(root, mode):
	data = Namespace()
	data.type = 'DefaultCLS'
	data.root = root
	data.loader_type = 'pil'
	data.sampler = 'naive'
	data.nb_classes = 2
	data.train_transforms = [
		dict(type='RandomHorizontalFlip', p=0.0),
		dict(type='ToTensor'),
	]
	data.test_transforms = [
		dict(type='ToTensor'),
	]
	trainer_data = Namespace()
	trainer_data.batch_size_per_gpu = 1
	trainer_data.batch_size_per_gpu_test = 1
	trainer_data.num_workers_per_gpu = 0
	trainer_data.pin_memory = False
	trainer_data.drop_last = False
	trainer_data.persistent_workers = False
	trainer = Namespace()
	trainer.data = trainer_data
	cfg = Namespace()
	cfg.data = data
	cfg.trainer = trainer
	cfg.dist = False
	cfg.mode = mode
	return cfg


def _write_empty_file(path):
	with open(path, 'wb') as f:
		f.write(b'{}')


def _create_defaultcls_structure(root):
	for split in ['train', 'val', 'test']:
		class_dir = os.path.join(root, split, 'class0')
		os.makedirs(class_dir, exist_ok=True)
		img_path = os.path.join(class_dir, 'img.png')
		Image.new('RGB', (2, 2), color=(255, 0, 0)).save(img_path)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestAssertClsSplits(unittest.TestCase):
	def test_missing_splits(self):
		from data import _assert_cls_splits
		test_cases = [
			('DefaultCLS', ['train', 'val'], 'test'),
			('ImageFolderLMDB', ['train.lmdb', 'val.lmdb'], 'test.lmdb'),
			('CustomImageDataset', ['train.json', 'val.json'], 'test.json'),
		]
		for data_type, present, missing in test_cases:
			with self.subTest(data_type=data_type, missing=missing):
				with tempfile.TemporaryDirectory() as root:
					for name in present:
						path = os.path.join(root, name)
						if data_type == 'DefaultCLS':
							os.makedirs(path, exist_ok=True)
						else:
							_write_empty_file(path)
					cfg = Namespace()
					cfg.data = Namespace()
					cfg.data.type = data_type
					cfg.data.root = root
					with self.assertRaises(SystemExit):
						_assert_cls_splits(cfg)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestGetLoader(unittest.TestCase):
	def test_get_loader_train_and_test(self):
		from data import get_loader
		with tempfile.TemporaryDirectory() as root:
			_create_defaultcls_structure(root)
			cfg_train = _build_cfg(root, mode='train')
			train_loader, val_loader, test_loader = get_loader(cfg_train)
			self.assertIsNotNone(train_loader)
			self.assertIsNotNone(val_loader)
			self.assertIsNone(test_loader)
			self.assertTrue(train_loader.dataset.root.endswith('train'))
			self.assertTrue(val_loader.dataset.root.endswith('val'))
			self.assertEqual(len(train_loader.dataset.transform.transforms), 2)
			self.assertEqual(len(val_loader.dataset.transform.transforms), 1)

			cfg_test = _build_cfg(root, mode='test')
			train_loader, val_loader, test_loader = get_loader(cfg_test)
			self.assertIsNone(train_loader)
			self.assertIsNone(val_loader)
			self.assertIsNotNone(test_loader)
			self.assertTrue(test_loader.dataset.root.endswith('test'))
