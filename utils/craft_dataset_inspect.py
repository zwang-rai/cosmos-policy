#!/usr/bin/env python3

"""Inspect one random sample from CraftDataset and visualize model inputs.

Example:
	/home/zwang/workspace/cosmos-policy/my_venv/bin/python utils/craft_dataset_inspect.py \
		--data-dir data/stack_banana_on_can
"""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cosmos_policy.datasets.craft_dataset import CraftDataset


def _to_hwc_uint8(video: np.ndarray) -> np.ndarray:
	"""Convert video from (C, T, H, W) to (T, H, W, C) uint8."""
	if video.ndim != 4:
		raise ValueError(f"Expected 4D video tensor, got shape={video.shape}")
	chw = np.transpose(video, (1, 2, 3, 0))
	if chw.dtype != np.uint8:
		if np.issubdtype(chw.dtype, np.floating):
			chw = np.clip((chw + 1.0) * 127.5, 0, 255).astype(np.uint8)
		else:
			chw = np.clip(chw, 0, 255).astype(np.uint8)
	return chw


def _build_frame_grid(frames: np.ndarray, max_frames: int = 20) -> Image.Image:
	"""Make a labeled mosaic of temporal frames."""
	frames = frames[:max_frames]
	t, h, w, _ = frames.shape
	cols = min(5, t)
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
		img = Image.fromarray(frames[i])
		canvas.paste(img, (x, y + label_h))
		draw.text((x, y), f"t={i}", fill=(255, 255, 255))
	return canvas


def _segment_summary(sample: dict) -> list[tuple[str, int]]:
	fields = [
		("current_proprio", "current_proprio_latent_idx"),
		("current_wrist", "current_wrist_image_latent_idx"),
		("current_primary", "current_image_latent_idx"),
		("action", "action_latent_idx"),
		("future_proprio", "future_proprio_latent_idx"),
		("future_wrist", "future_wrist_image_latent_idx"),
		("future_primary", "future_image_latent_idx"),
		("value", "value_latent_idx"),
	]
	out = []
	for name, key in fields:
		val = int(sample[key])
		if val >= 0:
			out.append((name, val))
	return out


def main() -> None:
	parser = argparse.ArgumentParser(description="Inspect one CraftDataset sample")
	parser.add_argument("--data-dir", type=str, required=True, help="Path to craft-format data directory")
	parser.add_argument("--index", type=int, default=None, help="Sample index; if omitted, use random")
	parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible random index")
	parser.add_argument("--chunk-size", type=int, default=16)
	parser.add_argument("--image-size", type=int, default=224)
	parser.add_argument("--max-frames", type=int, default=20, help="How many temporal frames to render")
	parser.add_argument(
		"--output-dir",
		type=str,
		default="/tmp/cosmos_policy_inspect",
		help="Directory where visualization artifacts are written",
	)
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)

	ds = CraftDataset(
		data_dir=args.data_dir,
		chunk_size=args.chunk_size,
		final_image_size=args.image_size,
		normalize_actions=False,
		normalize_proprio=False,
		normalize_images=False,
		use_image_aug=False,
		return_value_function_returns=True,
	)

	if len(ds) == 0:
		raise RuntimeError("Dataset is empty after loading.")

	idx = args.index if args.index is not None else random.randrange(len(ds))
	sample = ds[idx]

	video = sample["video"]
	if hasattr(video, "detach"):
		video_np = video.detach().cpu().numpy()
	else:
		video_np = np.asarray(video)
	frames = _to_hwc_uint8(video_np)

	out_dir = Path(args.output_dir) / f"craft_inspect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	out_dir.mkdir(parents=True, exist_ok=True)
	grid = _build_frame_grid(frames, max_frames=args.max_frames)
	grid_path = out_dir / "model_input_video_grid.png"
	grid.save(grid_path)

	seg = _segment_summary(sample)

	print("=== CraftDataset Sample Inspection ===")
	print(f"data_dir: {args.data_dir}")
	print(f"dataset_len: {len(ds)}")
	print(f"sample_idx: {idx}")
	print(f"video_shape (C,T,H,W): {tuple(video_np.shape)}")
	print(f"actions_shape: {tuple(np.asarray(sample['actions']).shape)}")
	print(f"next_actions_shape: {tuple(np.asarray(sample['next_action_chunk']).shape)}")
	print(f"proprio_shape: {tuple(np.asarray(sample['proprio']).shape)}")
	print(f"future_proprio_shape: {tuple(np.asarray(sample['future_proprio']).shape)}")
	print(f"t5_text_embeddings_shape: {tuple(np.asarray(sample['t5_text_embeddings']).shape)}")
	print(f"value_function_return: {float(sample['value_function_return']):.6f}")
	print(f"next_value_function_return: {float(sample['next_value_function_return']):.6f}")
	print("segment_latent_indices:")
	for name, sidx in seg:
		print(f"  - {name}: {sidx}")
	print(f"saved_grid: {grid_path}")


if __name__ == "__main__":
	main()
