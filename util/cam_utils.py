import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from configs import get_cfg
from data.utils import get_transforms
from model import get_model


@dataclass(frozen=True)
class CamRunConfig:
	cfg_path: str
	weights_path: str
	export_root: str = 'outputs/cam'
	alpha: float = 0.65
	target_layer: Optional[str] = None
	device: Optional[str] = None
	clip_low: float = 5.0
	clip_high: float = 99.0
	mask_threshold: float = 0.2
	fixed_seed: int = 42
	num_images: int = 10


@dataclass(frozen=True)
class ResolvedCamRunConfig:
	cfg_path: Path
	weights_path: Path
	export_root: Path
	alpha: float
	target_layer: Optional[str]
	device: Optional[str]
	clip_low: float
	clip_high: float
	mask_threshold: float
	fixed_seed: int
	num_images: int


# Edit this block when running util/cam_utils.py directly from an IDE.
IDE_RUN_CONFIG = CamRunConfig(
	cfg_path='configs/mobilemamba/mobilemamba_t2.py',
	weights_path='/home/kaiser/dl_project/FMobileMamba/runs/mobilemamba_lo/500_epochs/only_layer_operator/best_epoch431.pth',
	export_root='outputs/cam/fmobilemamba_mfconv_only',
	alpha=0.65,
	target_layer=None,
	device=None,
	clip_low=5.0,
	clip_high=99.0,
	mask_threshold=0.2,
	fixed_seed=42,
	num_images=10,
)


def select_random_image(root_dir: str, seed: int) -> str:
	"""Select one image path deterministically by seed.

	Args:
		root_dir: directory containing candidate images (recursively).
		seed: random seed used to choose the image.

	Returns:
		Absolute file path of the selected image.

	Raises:
		ValueError: when no images are found.
	"""
	rng = random.Random(seed)
	img_paths = sorted([str(p) for p in Path(root_dir).rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
	if not img_paths:
		raise ValueError(f'No images found under {root_dir}')
	return rng.choice(img_paths)


def select_random_images(root_dir: str, seed: int, num_images: int) -> list:
	"""Select multiple image paths deterministically by seed.

	Args:
		root_dir: directory containing candidate images (recursively).
		seed: random seed used to choose images.
		num_images: number of images to select.

	Returns:
		Absolute file paths of the selected images.

	Raises:
		ValueError: when no images are found or not enough images exist.
	"""
	if num_images <= 0:
		raise ValueError('num_images must be positive')
	rng = random.Random(seed)
	img_paths = sorted([str(p) for p in Path(root_dir).rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
	if len(img_paths) < num_images:
		raise ValueError(f'Not enough images under {root_dir} to sample {num_images}, found {len(img_paths)}')
	return rng.sample(img_paths, k=num_images)


def preprocess(img_path: str, transform) -> Tuple[torch.Tensor, Image.Image]:
	"""Load an image and apply transforms.

	Returns model input tensor (C,H,W) and original PIL image copy.
	"""
	img = Image.open(img_path).convert('RGB')
	orig = img.copy()
	if transform is not None:
		img = transform(img)
	return img, orig


def _get_default_target_layer(model: torch.nn.Module) -> torch.nn.Module:
	# heuristically pick the last convolutional layer
	convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
	if not convs:
		raise ValueError('No Conv2d layer found for CAM computation')
	return convs[-1]


def compute_grad_cam(model: torch.nn.Module, input_tensor: torch.Tensor, target_layer: Optional[str] = None) -> np.ndarray:
	"""Compute Grad-CAM heatmap normalized to [0,1].

	Args:
		model: network in eval mode.
		input_tensor: shape (1,C,H,W)
		target_layer: module name to hook; if None use last Conv2d.
	"""
	model.eval()
	input_tensor = input_tensor.requires_grad_(True)
	# pick layer
	layer = dict(model.named_modules()).get(target_layer) if target_layer else _get_default_target_layer(model)
	if layer is None:
		raise ValueError(f'Target layer {target_layer} not found')

	activations = []
	gradients = []

	def fwd_hook(_, __, output):
		activations.append(output.detach())

	def bwd_hook(_, grad_input, grad_output):
		# grad_output[0] corresponds to dL/dA
		gradients.append(grad_output[0].detach())

	handle_fwd = layer.register_forward_hook(fwd_hook)
	handle_bwd = layer.register_full_backward_hook(bwd_hook)

	try:
		out = model(input_tensor)
		if isinstance(out, tuple):
			out = out[0]
			# if distillation head exists, average predictions
			if isinstance(out, tuple):
				out = sum(out) / len(out)
		pred_class = out.argmax(dim=1)
		target = out[0, pred_class]
		target.backward()
		act = activations[0]
		grad = gradients[0]
		weights = grad.mean(dim=(2, 3), keepdim=True)
		cam = F.relu((weights * act).sum(dim=1, keepdim=True))
		cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
		cam = cam.squeeze().detach().cpu().numpy()
		cam -= cam.min()
		cam /= (cam.max() + 1e-8)
		return cam
	finally:
		handle_fwd.remove()
		handle_bwd.remove()


def resize_to_orig(cam_map: np.ndarray, orig_size: Tuple[int, int]) -> np.ndarray:
	cam_tensor = torch.from_numpy(cam_map).unsqueeze(0).unsqueeze(0)
	resized = F.interpolate(cam_tensor, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)
	return resized.squeeze().cpu().numpy()


def _clip_and_mask(cam_map: np.ndarray, clip_low: float, clip_high: float, mask_threshold: float) -> np.ndarray:
	low_v, high_v = np.percentile(cam_map, [clip_low, clip_high])
	cam = np.clip((cam_map - low_v) / (high_v - low_v + 1e-8), 0, 1)
	if mask_threshold > 0:
		cam[cam < mask_threshold] = 0
	return cam


def to_colormap(cam_map: np.ndarray, clip_low: float = 5.0, clip_high: float = 99.0, mask_threshold: float = 0.2) -> Image.Image:
	cam_norm = _clip_and_mask(cam_map, clip_low, clip_high, mask_threshold)
	colormap = cm.get_cmap('turbo')
	rgba = (colormap(cam_norm) * 255).astype(np.uint8)  # (H, W, 4)
	return Image.fromarray(rgba, mode='RGBA')


def overlay(orig: Image.Image, cam_color: Image.Image, alpha: float) -> Image.Image:
	orig = orig.convert('RGBA')
	cam_resized = cam_color.resize(orig.size, resample=Image.BILINEAR)
	# 将 cam 的自身 alpha 叠加到整体透明度，突出高激活区域
	r, g, b, a = cam_resized.split()
	a = a.point(lambda v: int(v * alpha))
	cam_resized = Image.merge('RGBA', (r, g, b, a))
	return Image.alpha_composite(orig, cam_resized)


def save_outputs(cam_color: Image.Image, overlay_img: Image.Image, export_root: str, base_name: str, seed: int) -> Tuple[str, str]:
	os.makedirs(export_root, exist_ok=True)
	cam_path = os.path.join(export_root, f"{base_name}__seed{seed}__cam.png")
	overlay_path = os.path.join(export_root, f"{base_name}__seed{seed}__overlay.png")
	cam_color.save(cam_path)
	overlay_img.save(overlay_path)
	return cam_path, overlay_path


def save_overlay_only(overlay_img: Image.Image, export_root: str, base_name: str, seed: int, index: int) -> str:
	os.makedirs(export_root, exist_ok=True)
	overlay_path = os.path.join(export_root, f"{base_name}__seed{seed}__idx{index}__overlay.png")
	overlay_img.save(overlay_path)
	return overlay_path


def build_vertical_comparison(orig_image: Image.Image, overlay_img: Image.Image) -> Image.Image:
	orig_rgb = orig_image.convert('RGB')
	overlay_rgb = overlay_img.convert('RGB')
	if orig_rgb.size != overlay_rgb.size:
		raise ValueError(f'orig_image size {orig_rgb.size} does not match overlay_img size {overlay_rgb.size}')

	width, height = orig_rgb.size
	comparison_img = Image.new('RGB', (width, height * 2))
	comparison_img.paste(orig_rgb, (0, 0))
	comparison_img.paste(overlay_rgb, (0, height))
	return comparison_img


def save_comparison_only(comparison_img: Image.Image, export_root: str, base_name: str, seed: int, index: int) -> str:
	os.makedirs(export_root, exist_ok=True)
	comparison_path = os.path.join(export_root, f"{base_name}__seed{seed}__idx{index}__con.png")
	comparison_img.save(comparison_path)
	return comparison_path


def _validate_cam_run_config(run_config: CamRunConfig) -> None:
	if not run_config.cfg_path:
		raise ValueError('cfg_path is required')
	if not run_config.weights_path:
		raise ValueError('weights_path is required')
	if not run_config.export_root:
		raise ValueError('export_root is required')
	if run_config.num_images <= 0:
		raise ValueError('num_images must be positive')
	if not 0 <= run_config.alpha <= 1:
		raise ValueError('alpha must be between 0 and 1')
	if run_config.clip_low >= run_config.clip_high:
		raise ValueError('clip_low must be smaller than clip_high')
	if not 0 <= run_config.mask_threshold <= 1:
		raise ValueError('mask_threshold must be between 0 and 1')


def _resolve_existing_path(path_value: str, project_root: Path, field_name: str) -> Path:
	path = Path(path_value)
	if not path.is_absolute():
		path = project_root / path
	path = path.resolve()
	if not path.exists():
		raise ValueError(f'{field_name} does not exist: {path}')
	return path


def _resolve_export_root(export_root: str, project_root: Path) -> Path:
	path = Path(export_root)
	if not path.is_absolute():
		path = project_root / path
	path = path.resolve()
	project_root = project_root.resolve()
	if path != project_root and project_root not in path.parents:
		raise ValueError(f'export_root must be inside project_root: {project_root}')
	return path


def resolve_cam_run_config(run_config: CamRunConfig, project_root: Path = PROJECT_ROOT) -> ResolvedCamRunConfig:
	_validate_cam_run_config(run_config)
	project_root = project_root.resolve()
	return ResolvedCamRunConfig(
		cfg_path=_resolve_existing_path(run_config.cfg_path, project_root, 'cfg_path'),
		weights_path=_resolve_existing_path(run_config.weights_path, project_root, 'weights_path'),
		export_root=_resolve_export_root(run_config.export_root, project_root),
		alpha=run_config.alpha,
		target_layer=run_config.target_layer,
		device=run_config.device,
		clip_low=run_config.clip_low,
		clip_high=run_config.clip_high,
		mask_threshold=run_config.mask_threshold,
		fixed_seed=run_config.fixed_seed,
		num_images=run_config.num_images,
	)


def _normalize_cfg_path_for_get_cfg(cfg_path: str, project_root: Path = PROJECT_ROOT) -> str:
	path = Path(cfg_path)
	if not path.is_absolute():
		return path.as_posix()
	path = path.resolve()
	project_root = project_root.resolve()
	if path != project_root and project_root not in path.parents:
		raise ValueError(f'cfg_path must be inside project_root: {project_root}')
	return path.relative_to(project_root).as_posix()


def _build_cfg_args(cfg_path: str) -> argparse.Namespace:
	return argparse.Namespace(cfg_path=cfg_path, mode='test', sleep=-1, memory=-1, dist_url='env://', logger_rank=0, opts=[])


def load_cfg(cfg_path: str):
	normalized_cfg_path = _normalize_cfg_path_for_get_cfg(cfg_path, project_root=PROJECT_ROOT)
	return get_cfg(_build_cfg_args(normalized_cfg_path))


def load_transforms(cfg_path: str):
	cfg = load_cfg(cfg_path)
	return get_transforms(cfg, train=False, cfg_transforms=cfg.data.test_transforms)


def load_model(weights_path: str, cfg_path: str, device: torch.device):
	cfg = load_cfg(cfg_path)
	cfg.model.model_kwargs['checkpoint_path'] = weights_path
	model = get_model(cfg.model)
	model.to(device)
	model.eval()
	return model


def _resolve_device(device_name: Optional[str]) -> torch.device:
	device_str = device_name if device_name is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
	return torch.device(device_str)


def _infer_target_layer_name(model: torch.nn.Module, target_layer: Optional[str]) -> Optional[str]:
	if target_layer is not None:
		return target_layer
	convs = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
	if not convs:
		return None
	return convs[-1]


def generate_cam_outputs(run_config: CamRunConfig, project_root: Path = PROJECT_ROOT) -> dict:
	resolved_config = resolve_cam_run_config(run_config, project_root=project_root)
	device = _resolve_device(resolved_config.device)
	cfg = load_cfg(str(resolved_config.cfg_path))
	test_transforms = get_transforms(cfg, train=False, cfg_transforms=cfg.data.test_transforms)
	cfg.model.model_kwargs['checkpoint_path'] = str(resolved_config.weights_path)
	model = get_model(cfg.model).to(device)
	model.eval()

	root_dir = Path(cfg.data.root_dir) / getattr(cfg.data, 'test_subdir', 'test')
	img_paths = select_random_images(str(root_dir), seed=resolved_config.fixed_seed, num_images=resolved_config.num_images)
	layer_name = _infer_target_layer_name(model, resolved_config.target_layer)

	overlay_paths = []
	comparison_paths = []
	for idx, img_path in enumerate(img_paths, start=1):
		input_tensor, orig_image = preprocess(img_path, test_transforms)
		input_tensor = input_tensor.unsqueeze(0).to(device)
		cam_map = compute_grad_cam(model, input_tensor, target_layer=resolved_config.target_layer)
		cam_resized = resize_to_orig(cam_map, orig_image.size)
		cam_color = to_colormap(
			cam_resized,
			clip_low=resolved_config.clip_low,
			clip_high=resolved_config.clip_high,
			mask_threshold=resolved_config.mask_threshold,
		)
		overlay_img = overlay(orig_image, cam_color, alpha=resolved_config.alpha)
		base_name = Path(img_path).stem
		overlay_path = save_overlay_only(
			overlay_img,
			str(resolved_config.export_root),
			base_name=base_name,
			seed=resolved_config.fixed_seed,
			index=idx,
		)
		comparison_img = build_vertical_comparison(orig_image, overlay_img)
		comparison_path = save_comparison_only(
			comparison_img,
			str(resolved_config.export_root),
			base_name=base_name,
			seed=resolved_config.fixed_seed,
			index=idx,
		)
		overlay_paths.append((img_path, overlay_path))
		comparison_paths.append((img_path, comparison_path))

	return {
		'config': str(resolved_config.cfg_path),
		'weights': str(resolved_config.weights_path),
		'device': str(device),
		'fixed_seed': resolved_config.fixed_seed,
		'num_images': resolved_config.num_images,
		'target_layer': layer_name,
		'selected_imgs': img_paths,
		'export_overlays': overlay_paths,
		'export_cons': comparison_paths,
		'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
	}


def print_cam_summary(summary: dict) -> None:
	print('=== CAM Generation Summary ===')
	print(f"config      : {summary['config']}")
	print(f"weights     : {summary['weights']}")
	print(f"device      : {summary['device']}")
	print(f"fixed_seed  : {summary['fixed_seed']}")
	print(f"num_images  : {summary['num_images']}")
	print(f"target_layer: {summary['target_layer']}")
	print('selected_imgs:')
	for idx, img_path in enumerate(summary['selected_imgs'], start=1):
		print(f'  [{idx}] {img_path}')
	print('export_overlays:')
	for idx, (_, overlay_path) in enumerate(summary['export_overlays'], start=1):
		print(f'  [{idx}] {overlay_path}')
	print('export_cons:')
	for idx, (_, comparison_path) in enumerate(summary['export_cons'], start=1):
		print(f'  [{idx}] {comparison_path}')
	print(f"timestamp   : {summary['timestamp']}")


def main(run_config: Optional[CamRunConfig] = None) -> dict:
	config = IDE_RUN_CONFIG if run_config is None else run_config
	summary = generate_cam_outputs(config, project_root=PROJECT_ROOT)
	print_cam_summary(summary)
	return summary


if __name__ == '__main__':
	main()
