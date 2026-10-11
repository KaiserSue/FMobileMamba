import os


def require_val_subdir(root_dir, val_subdir):
	"""Require a validation directory for training-time evaluation."""
	path = os.path.join(root_dir, val_subdir)
	if os.path.isdir(path):
		return val_subdir
	if os.path.exists(path):
		raise NotADirectoryError(f'Validation split path is not a directory: {path}')
	raise FileNotFoundError(f'Validation split directory does not exist: {path}')


def select_test_subdir(root_dir, test_subdir, val_subdir):
	"""Select the explicit test directory, preserving the test-first rule."""
	for subdir in (test_subdir, val_subdir):
		path = os.path.join(root_dir, subdir)
		if os.path.isdir(path):
			return subdir
		if os.path.exists(path):
			raise NotADirectoryError(f'Evaluation split path is not a directory: {path}')
	raise FileNotFoundError(
		f'No evaluation split directory found; checked '
		f'{os.path.join(root_dir, test_subdir)} and {os.path.join(root_dir, val_subdir)}'
	)
