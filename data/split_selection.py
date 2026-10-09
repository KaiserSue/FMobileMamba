import os


def select_eval_subdir(root_dir, test_subdir, val_subdir):
	"""Select the evaluation directory, preferring an existing test split."""
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
