# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-arm robot dataset loader.

Reads HDF5 files produced by VPL/GELLO data collection pipelines.
Each HDF5 contains JPEG-compressed camera views in the `color` dataset,
joint states in `arm_joint_positions` / `arm_gripper_width`, and actions.

Run this command to print a few samples:
    python -m cosmos_policy.datasets.craft_dataset
"""

import os
import pickle

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import (
    build_demo_step_index_mapping,
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    get_action_chunk_with_padding,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    get_hdf5_files,
    preprocess_image,
    rescale_data,
)
from cosmos_policy.utils.utils import duplicate_array

# Set floating point precision to 3 decimal places and disable line wrapping
np.set_printoptions(precision=3, linewidth=np.inf)


def decode_jpeg_from_color(jpeg_bytes) -> np.ndarray:
    """Decode a single JPEG frame from the `color` dataset (object dtype).

    Args:
        jpeg_bytes: Raw JPEG bytes stored as an HDF5 object element.

    Returns:
        np.ndarray: Decoded RGB image of shape (H, W, 3), uint8.
    """
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class CraftDataset(Dataset):
    """Dataset for real-world single-arm anytask saved in craft format.

    Expected HDF5 layout per episode file::

        action                  (T, action_dim)   float32
        arm_joint_positions     (T, 7)            float64
        arm_gripper_width       (T, 1)            float64
        color                   (T, num_cams)     object   # JPEG bytes
        [optional] task_description attribute

    Camera convention (from ``color`` dataset, axis-1 index):
        0 = primary / high camera
        1 = wrist camera

    The ``__getitem__`` output matches the format used by ``LIBERODataset``
    so it is compatible with the existing Cosmos-Policy training loop
    (``state_t=9`` config: blank, proprio, wrist, primary, action,
    future proprio, future wrist, future primary, value).
    """

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 16,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 1.0,
        success_rollout_sampling_prob: float = 0.0,
        treat_success_rollouts_as_demos: bool = False,
        return_value_function_returns: bool = True,
        gamma: float = 0.998,
        # Franka-specific
        task_description: str = "",
        primary_cam_idx: int = 0,
        wrist_cam_idx: int = 1,
    ):
        """
        Args:
            data_dir: Path to directory containing ``episode_*/episode_*.h5``.
            chunk_size: Action chunk size.
            final_image_size: Target square image size.
            t5_text_embeddings_path: Path to pre-computed T5 embeddings pkl.
            normalize_images: Whether to normalize images to float32.
            normalize_actions: Whether to min-max normalize actions to [-1,1].
            normalize_proprio: Whether to min-max normalize proprio to [-1,1].
            use_image_aug: Whether to apply image augmentations.
            use_stronger_image_aug: Whether to apply stronger augmentations.
            use_wrist_images: Whether to include wrist camera.
            use_third_person_images: Whether to include primary camera.
            use_proprio: Whether to include proprioceptive state.
            num_duplicates_per_image: Temporal duplication factor per image (WAN tokenizer).
            rollout_data_dir: Path to rollout data directory (kept for API compat).
            demonstration_sampling_prob: Probability of sampling demos.
            success_rollout_sampling_prob: Probability of sampling success rollouts.
            return_value_function_returns: Whether to compute/return value function returns.
            gamma: Discount factor for Monte-Carlo returns.
            task_description: Fallback language instruction if not in HDF5 attrs.
            primary_cam_idx: Index into ``color`` for the primary camera.
            wrist_cam_idx: Index into ``color`` for the wrist camera.
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.task_description = task_description
        self.primary_cam_idx = primary_cam_idx
        self.wrist_cam_idx = wrist_cam_idx
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob

        assert self.use_wrist_images or self.use_third_person_images, (
            "Must use at least one of wrist images or third-person images!"
        )

        # Discover HDF5 files
        hdf5_files = get_hdf5_files(data_dir)

        # ----------------------------------------------------------------
        # Load episode metadata + non-image data into RAM
        # Images are decoded lazily in __getitem__ to save memory.
        # ----------------------------------------------------------------
        self.data = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.unique_commands = set()

        for file_path in tqdm(hdf5_files, desc="Loading episodes"):
            try:
                with h5py.File(file_path, "r") as f:
                    # --- Actions ---
                    actions = f["action"][:].astype(np.float32)  # (T, action_dim)
                    num_steps = actions.shape[0]

                    # --- Proprioception ---
                    if "arm_joint_positions" in f and "arm_gripper_width" in f:
                        proprio = np.concatenate(
                            [
                                f["arm_joint_positions"][:].astype(np.float32),
                                f["arm_gripper_width"][:].reshape(-1, 1).astype(np.float32),
                            ],
                            axis=-1,
                        )  # (T, 8)
                    elif "joint_positions" in f and "gripper_width" in f:
                        # Fallback for alternative key naming
                        proprio = np.concatenate(
                            [
                                f["joint_positions"][:].astype(np.float32),
                                f["gripper_width"][:].reshape(-1, 1).astype(np.float32),
                            ],
                            axis=-1,
                        )
                    else:
                        print(f"WARNING: No joint state keys found in {file_path}, using zeros.")
                        proprio = np.zeros((num_steps, actions.shape[1]), dtype=np.float32)

                    # --- Language instruction ---
                    command = f.attrs.get("task_description", self.task_description)
                    if isinstance(command, bytes):
                        command = command.decode("utf-8")
                    self.unique_commands.add(command)

                    # --- Value function returns ---
                    returns = None
                    if self.return_value_function_returns:
                        returns = compute_monte_carlo_returns(
                            num_steps, terminal_reward=1.0, gamma=self.gamma
                        )

                    self.data[self.num_episodes] = dict(
                        file_path=file_path,
                        proprio=proprio,
                        actions=actions,
                        command=command,
                        num_steps=num_steps,
                        returns=returns.copy() if returns is not None else None,
                        success=True,
                    )
                    self.num_episodes += 1
                    self.num_steps += num_steps

            except Exception as e:
                print(f"ERROR loading {file_path}: {e}")

        print(f"Loaded {self.num_episodes} episodes, {self.num_steps} total steps.")
        print(f"Unique commands: {self.unique_commands}")

        # Build mapping from global step index → (episode_idx, relative_step_idx)
        self._build_step_index_mapping()

        # --- T5 text embeddings ---
        self.t5_text_embeddings = None
        if t5_text_embeddings_path and os.path.exists(t5_text_embeddings_path):
            with open(t5_text_embeddings_path, "rb") as fobj:
                self.t5_text_embeddings = pickle.load(fobj)
            print(f"Loaded T5 text embeddings from: {t5_text_embeddings_path}")

        # --- Dataset statistics (for normalization) ---
        self.dataset_stats = load_or_compute_dataset_statistics(
            data_dir=self.data_dir,
            data=self.data,
            calculate_dataset_statistics_func=calculate_dataset_statistics,
        )

        # Normalize actions and/or proprio
        if self.normalize_actions or self.normalize_proprio:
            if self.normalize_actions:
                self.data = rescale_data(self.data, self.dataset_stats, "actions")
            if self.normalize_proprio:
                self.data = rescale_data(self.data, self.dataset_stats, "proprio")

            self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )

        # Calculate epoch structure (demo-only, no rollouts)
        self._calculate_epoch_structure()

    # ------------------------------------------------------------------
    # Index mappings
    # ------------------------------------------------------------------

    def _build_step_index_mapping(self):
        result = build_demo_step_index_mapping(self.data)
        self._step_to_episode_map = result["_step_to_episode_map"]
        self._total_steps = result["_total_steps"]

    def _calculate_epoch_structure(self):
        result = calculate_epoch_structure(
            num_steps=self.num_steps,
            rollout_success_total_steps=0,
            rollout_failure_total_steps=0,
            demonstration_sampling_prob=self.demonstration_sampling_prob,
            success_rollout_sampling_prob=self.success_rollout_sampling_prob,
        )
        self.adjusted_demo_count = result["adjusted_demo_count"]
        self.adjusted_success_rollout_count = result["adjusted_success_rollout_count"]
        self.adjusted_failure_rollout_count = result["adjusted_failure_rollout_count"]
        self.epoch_length = result["epoch_length"]

    def __len__(self):
        return self.epoch_length

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """Return a training sample matching the LIBERO / Cosmos-Policy format.

        The image sequence layout (state_t=9):
            0: blank (1 frame)
            1: current proprio placeholder (num_duplicates_per_image frames)
            2: current wrist image (num_duplicates_per_image frames)
            3: current primary image (num_duplicates_per_image frames)
            4: action placeholder (num_duplicates_per_image frames)
            5: future proprio placeholder (num_duplicates_per_image frames)
            6: future wrist image (num_duplicates_per_image frames)
            7: future primary image (num_duplicates_per_image frames)
            8: value placeholder (num_duplicates_per_image frames)
        """
        # Map global index to episode/step
        global_step_idx = idx % self.num_steps
        episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
        episode_data = self.data[episode_idx]

        # Future frame index (clamped to episode end)
        future_frame_idx = min(
            relative_step_idx + self.chunk_size,
            episode_data["num_steps"] - 1,
        )

        # ----- Decode images from HDF5 (lazy) -----
        with h5py.File(episode_data["file_path"], "r") as f:
            color_ds = f["color"]
            current_row = color_ds[relative_step_idx]   # (num_cams,) object
            future_row = color_ds[future_frame_idx]     # (num_cams,) object

            primary_current = decode_jpeg_from_color(current_row[self.primary_cam_idx])
            primary_future = decode_jpeg_from_color(future_row[self.primary_cam_idx])

            wrist_current = decode_jpeg_from_color(current_row[self.wrist_cam_idx])
            wrist_future = decode_jpeg_from_color(future_row[self.wrist_cam_idx])

        # ----- Build image sequence -----
        frames = []      # list of (H, W, 3) np.ndarray
        repeats = []     # number of temporal duplicates per frame
        segment_idx = 0  # logical segment counter for latent indices

        # Pre-init latent indices
        action_latent_idx = -1
        value_latent_idx = -1
        current_proprio_latent_idx = -1
        current_wrist_image_latent_idx = -1
        current_image_latent_idx = -1
        future_proprio_latent_idx = -1
        future_wrist_image_latent_idx = -1
        future_image_latent_idx = -1

        # 0. Blank first frame
        blank = np.zeros_like(primary_current)
        frames.append(blank)
        repeats.append(1)
        segment_idx += 1

        # 1. Current proprio placeholder
        if self.use_proprio:
            current_proprio_latent_idx = segment_idx
            frames.append(np.zeros_like(primary_current))
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # 2. Current wrist image
        if self.use_wrist_images:
            current_wrist_image_latent_idx = segment_idx
            frames.append(wrist_current)
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # 3. Current primary image
        if self.use_third_person_images:
            current_image_latent_idx = segment_idx
            frames.append(primary_current)
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # 4. Action placeholder
        action_latent_idx = segment_idx
        frames.append(np.zeros_like(primary_current))
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # 5. Future proprio placeholder
        if self.use_proprio:
            future_proprio_latent_idx = segment_idx
            frames.append(np.zeros_like(primary_current))
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # 6. Future wrist image
        if self.use_wrist_images:
            future_wrist_image_latent_idx = segment_idx
            frames.append(wrist_future)
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # 7. Future primary image
        if self.use_third_person_images:
            future_image_latent_idx = segment_idx
            frames.append(primary_future)
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # 8. Value placeholder
        if self.return_value_function_returns:
            value_latent_idx = segment_idx
            frames.append(np.zeros_like(primary_current))
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # ----- Preprocess images -----
        all_unique_images = np.stack(frames, axis=0)  # (num_segments, H, W, 3)
        all_unique_images = preprocess_image(
            all_unique_images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        # Expand by repeat counts
        lengths = torch.as_tensor(repeats, dtype=torch.long, device=all_unique_images.device)
        all_images = torch.repeat_interleave(all_unique_images, lengths, dim=1)

        # ----- Action chunks -----
        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        next_relative_step_idx = min(
            relative_step_idx + self.chunk_size,
            episode_data["num_steps"] - 1,
        )
        next_action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=next_relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # ----- Value function return -----
        if self.return_value_function_returns and episode_data["returns"] is not None:
            value_function_return = float(episode_data["returns"][future_frame_idx])
        else:
            value_function_return = -100.0

        next_future_frame_idx = min(
            next_relative_step_idx + self.chunk_size,
            episode_data["num_steps"] - 1,
        )
        if self.return_value_function_returns and episode_data["returns"] is not None:
            next_value_function_return = float(episode_data["returns"][next_future_frame_idx])
        else:
            next_value_function_return = -100.0

        # ----- T5 text embeddings -----
        if self.t5_text_embeddings is not None and episode_data["command"] in self.t5_text_embeddings:
            t5_emb = self.t5_text_embeddings[episode_data["command"]]
            if isinstance(t5_emb, torch.Tensor):
                t5_emb = t5_emb.squeeze() # (1, 512, 1024) -> (512, 1024)
        else:
            t5_emb = torch.zeros(512, 1024)

        # ----- Proprio -----
        proprio = (
            episode_data["proprio"][relative_step_idx]
            if self.use_proprio
            else np.zeros_like(episode_data["proprio"][relative_step_idx])
        )
        future_proprio = (
            episode_data["proprio"][future_frame_idx]
            if self.use_proprio
            else np.zeros_like(episode_data["proprio"][future_frame_idx])
        )

        return {
            "video": all_images,
            "actions": action_chunk,
            "t5_text_embeddings": t5_emb,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio,
            "future_proprio": future_proprio,
            "__key__": idx,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx,
            "current_proprio_latent_idx": current_proprio_latent_idx if self.use_proprio else -1,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx if self.use_wrist_images else -1,
            "current_wrist_image2_latent_idx": -1,  # Single arm — no second wrist
            "current_image_latent_idx": current_image_latent_idx if self.use_third_person_images else -1,
            "future_proprio_latent_idx": future_proprio_latent_idx if self.use_proprio else -1,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx if self.use_wrist_images else -1,
            "future_wrist_image2_latent_idx": -1,  # Single arm — no second wrist
            "future_image_latent_idx": future_image_latent_idx if self.use_third_person_images else -1,
            "value_function_return": value_function_return,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": next_value_function_return,
        }


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/stack_banana_on_can"
    ds = CraftDataset(
        data_dir=data_dir,
        chunk_size=16,
        final_image_size=224,
        normalize_actions=False,
        normalize_proprio=False,
        use_image_aug=False,
    )
    print(f"\nDataset length: {len(ds)}")
    sample = ds[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"  video shape:           {sample['video'].shape}")
    print(f"  actions shape:         {sample['actions'].shape}")
    print(f"  proprio shape:         {np.array(sample['proprio']).shape}")
    print(f"  t5_text_embeddings:    {sample['t5_text_embeddings'].shape}")
    print(f"  action_latent_idx:     {sample['action_latent_idx']}")
    print(f"  value_latent_idx:      {sample['value_latent_idx']}")
