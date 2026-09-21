import ast
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


@dataclass(frozen=True)
class ImageMeta:
	path: Path
	width: int
	height: int
	aspect_ratio: float
	area: int


@dataclass(frozen=True)
class ReferenceProfile:
	target_aspect_ratio: float
	reference_area: float
	max_crop_ratio: float
	max_pad_ratio: float


@dataclass(frozen=True)
class SelectionPolicy:
	seed: int
	max_crop_ratio: float
	max_pad_ratio: float
	pad_color: Tuple[int, int, int]
	target_aspect_ratio: Optional[float] = None


@dataclass(frozen=True)
class CropPlan:
	crop_box: Tuple[int, int, int, int]
	crop_ratio_per_edge: Tuple[float, float, float, float]
	pad: Tuple[int, int, int, int]
	pad_ratio_per_edge: Tuple[float, float, float, float]


@dataclass(frozen=True)
class SelectedItem:
	meta: ImageMeta
	plan: CropPlan


@dataclass(frozen=True)
class TileSpec:
	width: int
	height: int
	pad_color: Tuple[int, int, int]


def read_data_config(config_path):
	path = Path(config_path)
	if not path.is_file():
		raise FileNotFoundError(f'Config file not found: {path}')
	source = path.read_text(encoding='utf-8')
	tree = ast.parse(source, filename=str(path))

	values = {}
	for node in tree.body:
		if not isinstance(node, ast.Assign):
			continue
		for target in node.targets:
			key = _extract_data_key(target)
			if key is None:
				continue
			value = _resolve_value(node.value, values)
			if value is not None:
				values[key] = value

	root = values.get('root') or values.get('root_dir')
	if root is None:
		raise ValueError(f'Config missing data.root or data.root_dir: {path}')
	values['root'] = root
	return values


def get_split_root(values, split):
	if split not in {'train', 'val', 'test'}:
		raise ValueError(f'Unsupported split: {split}')
	root = Path(values['root'])
	subdir = values.get(f'{split}_subdir')
	return root / subdir if subdir else root


def list_class_dirs(root_path):
	if not root_path.exists():
		raise FileNotFoundError(f'Dataset root not found: {root_path}')
	if not root_path.is_dir():
		raise NotADirectoryError(f'Dataset root is not a directory: {root_path}')
	dirs = [p for p in root_path.iterdir() if p.is_dir()]
	return sorted(dirs, key=lambda p: p.name)


def collect_image_meta(class_dir):
	images = _collect_images(class_dir)
	if not images:
		raise ValueError(f'No images found in class dir: {class_dir}')
	metas = []
	for img_path in images:
		with Image.open(img_path) as img:
			width, height = img.size
		if width <= 0 or height <= 0:
			raise ValueError(f'Invalid image size for {img_path}')
		metas.append(
			ImageMeta(
				path=img_path,
				width=width,
				height=height,
				aspect_ratio=width / height,
				area=width * height,
			)
		)
	return metas


def compute_reference_profile(all_class_metas, policy):
	if not all_class_metas:
		raise ValueError('all_class_metas is empty')
	per_class_ar = []
	per_class_area = []
	for metas in all_class_metas:
		if not metas:
			raise ValueError('class metas is empty')
		per_class_ar.append(statistics.median([m.aspect_ratio for m in metas]))
		per_class_area.append(statistics.median([m.area for m in metas]))

	target_ar = policy.target_aspect_ratio if policy.target_aspect_ratio is not None else statistics.median(per_class_ar)
	if target_ar <= 0:
		raise ValueError('target_aspect_ratio must be positive')
	reference_area = statistics.median(per_class_area)
	if reference_area <= 0:
		raise ValueError('reference_area must be positive')
	return ReferenceProfile(
		target_aspect_ratio=target_ar,
		reference_area=reference_area,
		max_crop_ratio=policy.max_crop_ratio,
		max_pad_ratio=policy.max_pad_ratio,
	)


def select_images_per_class(class_dirs, policy, return_profile=False):
	if not class_dirs:
		raise ValueError('class_dirs is empty')
	all_class_metas = [collect_image_meta(class_dir) for class_dir in class_dirs]
	profile = compute_reference_profile(all_class_metas, policy)
	rng = random.Random(policy.seed)
	selected = []
	for class_dir, metas in zip(class_dirs, all_class_metas):
		candidates = []
		for meta in metas:
			plan = _try_plan_crop_and_pad(
				meta,
				target_aspect_ratio=profile.target_aspect_ratio,
				max_crop_ratio=profile.max_crop_ratio,
				max_pad_ratio=profile.max_pad_ratio,
			)
			if plan is None:
				continue
			size_diff = abs(math.log(meta.area / profile.reference_area))
			pad_total_ratio = sum(plan.pad_ratio_per_edge)
			crop_total_ratio = sum(plan.crop_ratio_per_edge)
			candidates.append((size_diff, pad_total_ratio, crop_total_ratio, meta.path.name, SelectedItem(meta, plan)))
		if not candidates:
			raise ValueError(f'No valid images found for class: {class_dir}')
		candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
		best_score = candidates[0][:3]
		best = [c for c in candidates if c[:3] == best_score]
		selected.append(rng.choice(best)[4])
	return (selected, profile) if return_profile else selected


def plan_crop_and_pad(meta, target_aspect_ratio, max_crop_ratio, max_pad_ratio):
	plan = _try_plan_crop_and_pad(meta, target_aspect_ratio, max_crop_ratio, max_pad_ratio)
	if plan is None:
		raise ValueError(f'No valid crop/pad plan for {meta.path}')
	return plan


def render_tile(selected_item, tile_spec):
	meta = selected_item.meta
	plan = selected_item.plan
	with Image.open(meta.path) as img:
		image = img.convert('RGB')
	if plan.crop_box != (0, 0, meta.width, meta.height):
		image = image.crop(plan.crop_box)
	pad_left = int(math.floor(tile_spec.width * plan.pad_ratio_per_edge[0]))
	pad_top = int(math.floor(tile_spec.height * plan.pad_ratio_per_edge[1]))
	pad_right = int(math.floor(tile_spec.width * plan.pad_ratio_per_edge[2]))
	pad_bottom = int(math.floor(tile_spec.height * plan.pad_ratio_per_edge[3]))
	content_width = tile_spec.width - pad_left - pad_right
	content_height = tile_spec.height - pad_top - pad_bottom
	if content_width <= 0 or content_height <= 0:
		raise ValueError('Padding exceeds tile dimensions')
	image = image.resize((content_width, content_height), resample=Image.BICUBIC)
	canvas = Image.new('RGB', (tile_spec.width, tile_spec.height), color=tile_spec.pad_color)
	canvas.paste(image, (pad_left, pad_top))
	return canvas


def build_image_grid(selected_items, rows, cols, tile_spec):
	if rows <= 0 or cols <= 0:
		raise ValueError('rows and cols must be positive')
	if tile_spec.width <= 0 or tile_spec.height <= 0:
		raise ValueError('tile size must be positive')
	expected = rows * cols
	if len(selected_items) != expected:
		raise ValueError(f'Expected {expected} images, got {len(selected_items)}')

	grid = Image.new('RGB', (cols * tile_spec.width, rows * tile_spec.height), color=tile_spec.pad_color)
	for idx, item in enumerate(selected_items):
		tile = render_tile(item, tile_spec)
		row = idx // cols
		col = idx % cols
		grid.paste(tile, (col * tile_spec.width, row * tile_spec.height))
	return grid


def make_tile_spec(tile_size, target_aspect_ratio, pad_color):
	if tile_size <= 0:
		raise ValueError('tile_size must be positive')
	if target_aspect_ratio <= 0:
		raise ValueError('target_aspect_ratio must be positive')
	if target_aspect_ratio >= 1:
		height = tile_size
		width = max(1, int(round(tile_size * target_aspect_ratio)))
	else:
		width = tile_size
		height = max(1, int(round(tile_size / target_aspect_ratio)))
	return TileSpec(width=width, height=height, pad_color=pad_color)


def _extract_data_key(target):
	if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
		if target.value.id == 'data':
			return target.attr
	return None


def _resolve_value(node, values):
	if isinstance(node, ast.Constant) and isinstance(node.value, str):
		return node.value
	if isinstance(node, ast.Str):
		return node.s
	if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
		if node.value.id == 'data':
			return values.get(node.attr)
	if isinstance(node, ast.Name):
		return values.get(node.id)
	return None


def _collect_images(class_dir):
	paths = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
	return sorted(paths, key=lambda p: p.name)


def _try_plan_crop_and_pad(meta, target_aspect_ratio, max_crop_ratio, max_pad_ratio):
	if meta.width <= 0 or meta.height <= 0:
		raise ValueError('Image dimensions must be positive')
	if target_aspect_ratio <= 0:
		raise ValueError('target_aspect_ratio must be positive')
	if max_crop_ratio < 0 or max_pad_ratio < 0:
		raise ValueError('max_crop_ratio and max_pad_ratio must be non-negative')

	aspect = meta.aspect_ratio
	full_crop = (0, 0, meta.width, meta.height)
	zeros = (0.0, 0.0, 0.0, 0.0)
	zero_pad = (0, 0, 0, 0)
	if abs(aspect - target_aspect_ratio) <= 1e-6:
		return CropPlan(crop_box=full_crop, crop_ratio_per_edge=zeros, pad=zero_pad, pad_ratio_per_edge=zeros)

	if aspect > target_aspect_ratio:
		return _plan_for_wide(meta, target_aspect_ratio, max_crop_ratio, max_pad_ratio)
	return _plan_for_tall(meta, target_aspect_ratio, max_crop_ratio, max_pad_ratio)


def _plan_for_wide(meta, target_aspect_ratio, max_crop_ratio, max_pad_ratio):
	width = meta.width
	height = meta.height
	max_crop_px = int(math.floor(max_crop_ratio * width))
	for crop_px in range(max_crop_px, -1, -1):
		new_width = width - 2 * crop_px
		if new_width <= 0:
			continue
		required_height = new_width / target_aspect_ratio
		if required_height < height:
			continue
		pad_total = required_height - height
		pad_total_px = int(math.floor(pad_total + 1e-6))
		final_height = height + pad_total_px
		if final_height <= 0:
			continue
		top_pad, bottom_pad = _split_padding(pad_total_px)
		if max(top_pad, bottom_pad) / final_height > max_pad_ratio:
			continue
		left_crop = crop_px
		right_crop = crop_px
		crop_box = (left_crop, 0, width - right_crop, height)
		plan = CropPlan(
			crop_box=crop_box,
			crop_ratio_per_edge=(left_crop / width, 0.0, right_crop / width, 0.0),
			pad=(0, top_pad, 0, bottom_pad),
			pad_ratio_per_edge=(0.0, top_pad / final_height, 0.0, bottom_pad / final_height),
		)
		return plan
	return None


def _plan_for_tall(meta, target_aspect_ratio, max_crop_ratio, max_pad_ratio):
	width = meta.width
	height = meta.height
	max_crop_px = int(math.floor(max_crop_ratio * height))
	for crop_px in range(max_crop_px, -1, -1):
		new_height = height - 2 * crop_px
		if new_height <= 0:
			continue
		required_width = new_height * target_aspect_ratio
		if required_width < width:
			continue
		pad_total = required_width - width
		pad_total_px = int(math.floor(pad_total + 1e-6))
		final_width = width + pad_total_px
		if final_width <= 0:
			continue
		left_pad, right_pad = _split_padding(pad_total_px)
		if max(left_pad, right_pad) / final_width > max_pad_ratio:
			continue
		top_crop = crop_px
		bottom_crop = crop_px
		crop_box = (0, top_crop, width, height - bottom_crop)
		plan = CropPlan(
			crop_box=crop_box,
			crop_ratio_per_edge=(0.0, top_crop / height, 0.0, bottom_crop / height),
			pad=(left_pad, 0, right_pad, 0),
			pad_ratio_per_edge=(left_pad / final_width, 0.0, right_pad / final_width, 0.0),
		)
		return plan
	return None


def _split_padding(total):
	left_or_top = total // 2
	right_or_bottom = total - left_or_top
	return left_or_top, right_or_bottom
