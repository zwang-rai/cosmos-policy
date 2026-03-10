#!/usr/bin/env python3

"""Inspect one random sample from AnyTask (VPLDataset) and print key metadata.

Example:
	/home/zwang/workspace/cosmos-policy/my_venv/bin/python utils/anytask_dataset_inspect.py
"""

from __future__ import annotations

import argparse
import glob
import math
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from cosmos_policy.datasets.vpl_dataset import VPLDataset


def _shape_of(value: Any) -> str:
	"""Return a compact shape/dtype description for array-like values."""
	if isinstance(value, torch.Tensor):
		return f"shape={tuple(value.shape)}, dtype={value.dtype}"
	if isinstance(value, np.ndarray):
		return f"shape={tuple(value.shape)}, dtype={value.dtype}"
	return f"type={type(value).__name__}"


def _scalar(value: Any) -> Any:
	"""Convert tensor/numpy scalar to a Python scalar for readable printing."""
	if isinstance(value, torch.Tensor):
		if value.numel() == 1:
			return value.detach().cpu().item()
		return value.detach().cpu().tolist()
	if isinstance(value, np.ndarray):
		if value.size == 1:
			return value.item()
		return value.tolist()
	return value


def _to_hwc_uint8(video: np.ndarray) -> np.ndarray:
	"""Convert video from (C, T, H, W) to (T, H, W, C) uint8 for visualization."""
	if video.ndim != 4:
		raise ValueError(f"Expected 4D video tensor, got shape={video.shape}")
	frames = np.transpose(video, (1, 2, 3, 0))
	if frames.dtype != np.uint8:
		if np.issubdtype(frames.dtype, np.floating):
			# Handles [-1, 1] normalized inputs.
			frames = np.clip((frames + 1.0) * 127.5, 0, 255).astype(np.uint8)
		else:
			frames = np.clip(frames, 0, 255).astype(np.uint8)
	return frames


def _build_frame_gallery(frames: np.ndarray, max_frames: int = 36) -> Image.Image:
	"""Create a labeled frame gallery with temporal indices."""
	frames = frames[:max_frames]
	t, h, w, _ = frames.shape
	cols = min(6, t)
	rows = math.ceil(t / cols)
	margin = 8
	label_h = 18

	canvas = Image.new(
		"RGB",
		(cols * w + (cols + 1) * margin, rows * (h + label_h) + (rows + 1) * margin),
		color=(20, 20, 20),
	)
	draw = ImageDraw.Draw(canvas)

	for i in range(t):
		r, c = divmod(i, cols)
		x = margin + c * (w + margin)
		y = margin + r * (h + label_h + margin)
		canvas.paste(Image.fromarray(frames[i]), (x, y + label_h))
		draw.text((x, y), f"idx={i}", fill=(255, 255, 255))
	return canvas


def main() -> None:
	parser = argparse.ArgumentParser(description="Randomly inspect one AnyTask dataset instance")
	parser.add_argument("--data-dir", type=str, default="data/skillgen_1000_replay")
	parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible sampling")
	parser.add_argument("--index", type=int, default=None, help="Explicit sample index. If omitted, sample randomly")
	parser.add_argument("--chunk-size", type=int, default=16)
	parser.add_argument("--image-size", type=int, default=224)
	parser.add_argument("--max-frames", type=int, default=36)
	parser.add_argument("--output-dir", type=str, default="utils/inspect")
	parser.add_argument(
		"--fast-single-episode",
		action="store_true",
		help="Create a temp index file with 1 random episode for much faster loading (default: enabled)",
	)
	parser.add_argument(
		"--no-fast-single-episode",
		action="store_true",
		help="Disable one-episode fast mode and load from the full dataset",
	)
	args = parser.parse_args()
	use_fast_single_episode = True
	if args.no_fast_single_episode:
		use_fast_single_episode = False
	if args.fast_single_episode:
		use_fast_single_episode = True

	if args.seed is not None:
		random.seed(args.seed)
		np.random.seed(args.seed)

	index_file = ""
	selected_episode_file = None
	selected_episode_idx = None
	if use_fast_single_episode:
		hdf5_files = sorted(glob.glob(str(Path(args.data_dir) / "episode_*" / "*.h5")))
		if not hdf5_files:
			raise RuntimeError(f"No episode h5 files found under {args.data_dir}")
		selected_episode_idx = random.randrange(len(hdf5_files))
		selected_episode_file = hdf5_files[selected_episode_idx]
		with tempfile.NamedTemporaryFile(mode="w", suffix="_one_episode_index.txt", delete=False) as tf:
			# VPLDataset index parser accepts comma/newline/whitespace separated indices.
			tf.write(f"{selected_episode_idx}\n")
			index_file = tf.name

	dataset = VPLDataset(
		data_dir=args.data_dir,
		is_train=True,
		index_file=index_file,
		chunk_size=args.chunk_size,
		final_image_size=args.image_size,
		normalize_actions=True,
		normalize_proprio=True,
		normalize_images=False,
		use_image_aug=False,
		use_stronger_image_aug=False,
		return_value_function_returns=True,
		use_wrist_images=True,
		use_third_person_images=True,
	)

	# Some AnyTask episodes may have empty command strings while embeddings do not contain "".
	# For inspection, remap missing commands to a valid embedding key so __getitem__ can run.
	embedding_keys = list(getattr(dataset, "t5_text_embeddings", {}).keys())
	if embedding_keys:
		default_command = embedding_keys[0]
		patched = 0
		for _, episode_data in dataset.data.items():
			cmd = episode_data.get("command", "")
			if cmd not in dataset.t5_text_embeddings:
				episode_data["command"] = default_command
				patched += 1
		if patched > 0:
			print(
				f"Patched {patched} episode(s) with missing command embeddings. "
				f"Fallback command: {default_command}"
			)
	else:
		raise RuntimeError("No T5 embeddings found in dataset; cannot build sample input.")

	if len(dataset) == 0:
		raise RuntimeError("Dataset is empty after loading.")

	if args.index is not None:
		if args.index < 0 or args.index >= len(dataset):
			raise ValueError(f"index {args.index} is out of range [0, {len(dataset) - 1}]")
		idx = args.index
	else:
		idx = random.randrange(len(dataset))

	sample = dataset[idx]

	video = sample["video"]
	video_np = video.detach().cpu().numpy() if isinstance(video, torch.Tensor) else np.asarray(video)
	frames = _to_hwc_uint8(video_np)

	out_dir = Path(args.output_dir) / f"anytask_inspect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	out_dir.mkdir(parents=True, exist_ok=True)
	gallery = _build_frame_gallery(frames, max_frames=args.max_frames)
	gallery_path = out_dir / "input_gallery_with_indices.png"
	gallery.save(gallery_path)

	print("=== AnyTask Dataset Random Instance ===")
	print(f"data_dir: {args.data_dir}")
	print(f"dataset_len: {len(dataset)}")
	print(f"sample_idx: {idx}")
	print(f"num_episodes: {dataset.num_episodes}")
	print(f"num_steps: {dataset.num_steps}")
	if use_fast_single_episode:
		print(f"temp_index_file: {index_file}")
		print(f"selected_episode_global_idx: {selected_episode_idx}")
		print(f"selected_episode_file: {selected_episode_file}")
	print(f"saved_gallery: {gallery_path}")

	# If this maps to a demo step, show episode provenance.
	episode_idx = None
	relative_step_idx = None
	if hasattr(dataset, "_step_to_episode_map") and idx in dataset._step_to_episode_map:
		episode_idx, relative_step_idx = dataset._step_to_episode_map[idx]
		episode_data = dataset.data.get(episode_idx, {})
		print(f"episode_idx: {episode_idx}")
		print(f"relative_step_idx: {relative_step_idx}")
		if "file_path" in episode_data:
			print(f"episode_file: {episode_data['file_path']}")
		if "command" in episode_data:
			print(f"command: {episode_data['command']}")

	print("\nSample keys and tensor info:")
	for key in sorted(sample.keys()):
		value = sample[key]
		if isinstance(value, (torch.Tensor, np.ndarray)):
			print(f"  {key}: {_shape_of(value)}")
		else:
			print(f"  {key}: {_scalar(value)}")

	latent_index_keys = [
		"current_proprio_latent_idx",
		"current_wrist_image_latent_idx",
		"current_image_latent_idx",
		"action_latent_idx",
		"future_proprio_latent_idx",
		"future_wrist_image_latent_idx",
		"future_image_latent_idx",
		"value_latent_idx",
	]
	print("\nLatent index layout:")
	for key in latent_index_keys:
		if key in sample:
			print(f"  {key}: {_scalar(sample[key])}")


if __name__ == "__main__":
	main()
