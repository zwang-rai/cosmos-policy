import os
import glob
import h5py
import numpy as np
import cv2
import torch
import random
from tqdm import tqdm
from cosmos_policy.datasets.aloha_dataset import ALOHADataset, get_video_num_frames, load_video_as_images
from cosmos_policy.datasets.dataset_common import determine_sample_type, get_action_chunk_with_padding
from cosmos_policy.utils.utils import duplicate_array
from cosmos_policy.datasets.dataset_utils import preprocess_image
import pickle
from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings, save_embeddings
from cosmos_policy.utils.transform_utils import quat2euler

class AnyTaskDataset(ALOHADataset):
    def __init__(
        self,
        data_dir: str,
        is_train: bool = True,
        chunk_size: int = 16,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images=False,
        normalize_actions=True,
        normalize_proprio=True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        debug: bool = False,
        debug2: bool = False,
        use_proprio: bool = True,
        num_history_indices: int = 8,
        history_spacing_factor: int = 12,
        num_duplicates_per_image: int = 4,
        return_value_function_returns: bool = True,
        gamma: float = 0.998,
        lazy_video_decompression: bool = False,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        treat_demos_as_success_rollouts: bool = False,
        treat_success_rollouts_as_demos: bool = False,
        use_jpeg_for_rollouts: bool = False,
        load_all_rollouts_into_ram: bool = False,
        use_third_person_images: bool = True,
        use_wrist_images: bool = True,
    ):
        # Initialize basic attributes (copying from ALOHADataset)
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.debug = debug
        self.debug2 = debug2
        self.use_proprio = use_proprio
        self.num_history_indices = num_history_indices
        self.history_spacing_factor = history_spacing_factor
        self.num_duplicates_per_image = num_duplicates_per_image
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.lazy_video_decompression = lazy_video_decompression
        self.rollout_data_dir = rollout_data_dir
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob
        self.treat_demos_as_success_rollouts = treat_demos_as_success_rollouts
        self.treat_success_rollouts_as_demos = treat_success_rollouts_as_demos
        self.use_jpeg_for_rollouts = use_jpeg_for_rollouts
        self.load_all_rollouts_into_ram = load_all_rollouts_into_ram
        self._jpeg_rollout_hint_emitted = False
        self.use_third_person_images = use_third_person_images
        self.use_wrist_images = use_wrist_images

        # Anytask (VPL) dataset file discovery
        hdf5_files = glob.glob(os.path.join(data_dir, "episode_*", "*.h5"))
        print("hdf5 files found:", len(hdf5_files))
        hdf5_files.sort() # Ensure deterministic order

        if is_train:
            # Simple split for demo: 90% train, 10% val
            split_idx = int(0.05 * len(hdf5_files))
            hdf5_files = hdf5_files[:split_idx]
            print(f"Training split: {len(hdf5_files)} episodes")
        else:
            # Validation set
            split_idx = int(0.05 * len(hdf5_files))
            hdf5_files = hdf5_files[split_idx:]
            print(f"Validation split: {len(hdf5_files)} episodes")
        
        if debug:
            hdf5_files = hdf5_files[:1]

        self.data = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.unique_commands = set()

        print(f"Found {len(hdf5_files)} HDF5 files in {data_dir}")

        for file in tqdm(hdf5_files):
            try:
                with h5py.File(file, "r") as f:
                    # Load End-Effector Actions & Proprio
                    # User requested 7-dim action space (Pos + Euler + Gripper) to match LIBERO
                    ee_pos = f["ee_position"][:] # (T, 3)
                    ee_rot = f["ee_rotation"][:] # (T, 4) Quaternion (assume x,y,z,w or w,x,y,z - quat2euler handles shape)
                    gripper = f["gripper_width"][:] # (T, 1)
                    
                    # Convert quaternion to euler
                    ee_euler = quat2euler(ee_rot) # (T, 3)
                    
                    # Form 7-dim action vector
                    # [x, y, z, roll, pitch, yaw, gripper]
                    actions = np.concatenate([ee_pos, ee_euler, gripper], axis=-1).astype(np.float32)
                    
                    # User requested 9-dim proprio (joint positions)
                    proprio = f["joint_pos"][:] # (T, 9)

                    # Determine Sequence Length
                    episode_num_steps = actions.shape[0]

                    # Construct Video Paths
                    # episode_folder/videos/camera0/episode_X.mp4
                    episode_folder = os.path.dirname(file)
                    # Extract episode name from folder or filename
                    episode_name = os.path.splitext(os.path.basename(file))[0] # episode_0
                    
                    # Construct Video Paths
                    video_paths_candidates = {
                        "cam_high": os.path.join(episode_folder, "videos", "camera1", f"{episode_name}.mp4"),
                        "cam_wrist": os.path.join(episode_folder, "videos", "wrist_camera", f"{episode_name}.mp4"),
                    }
                    
                    # Verify they exist
                    video_paths = {}
                    for k, v in video_paths_candidates.items():
                        if os.path.exists(v):
                            video_paths[k] = v
                    
                    if "cam_high" not in video_paths:
                        print(f"Warning: Primary video (cam_high/camera1) not found for {file}, skipping.")
                        continue
                        
                    # Load Images (Lazy or Eager)
                    if self.lazy_video_decompression:
                        # In lazy mode, we rely on video_paths. 
                        images = None
                        wrist_images = None
                    else:
                        # Helper to load safely
                        def load_safe(path):
                            if path and os.path.exists(path):
                                return load_video_as_images(path, resize_size=self.final_image_size)
                            return None
                        
                        images = load_safe(video_paths.get("cam_high"))
                        wrist_images = load_safe(video_paths.get("cam_wrist"))
                        
                    # Task Description / Command
                    # Search for task description in attributes, fallback to reasonable default
                    command = f.attrs.get("task_description", "Stack the banana on top of the meat can")
                    if isinstance(command, bytes):
                        command = command.decode("utf-8")
                    
                    self.unique_commands.add(command)
 
                    # Store Entry
                    # Must match keys expected by __getitem__
                    self.data[self.num_episodes] = {
                        "file_path": file,
                        "proprio": proprio,
                        "actions": actions,
                        "command": command,
                        "num_steps": episode_num_steps,
                        "video_paths": video_paths, # For lazy loading
                        "images": images,
                        "wrist_images": wrist_images,
                        "is_lazy_video": self.lazy_video_decompression,
                        "success": True, # Assume all are success demos
                    }
                    
                    self.num_episodes += 1
                    self.num_steps += episode_num_steps

            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

        # Post-loading setup (stats, mapping, etc) -> Copy from ALOHA
        from cosmos_policy.datasets.dataset_common import (
            calculate_epoch_structure, 
            load_or_compute_dataset_statistics, 
            load_or_compute_post_normalization_statistics
        )
        from cosmos_policy.datasets.dataset_utils import calculate_dataset_statistics, rescale_data

        # Build mapping
        self._build_step_index_mapping()

        # Stats
        self.dataset_stats = load_or_compute_dataset_statistics(
            data_dir=self.data_dir,
            data=self.data,
            calculate_dataset_statistics_func=calculate_dataset_statistics,
        )

        # Normalize
        if self.normalize_actions:
            self.data = rescale_data(self.data, self.dataset_stats, "actions")
        if self.normalize_proprio:
            self.data = rescale_data(self.data, self.dataset_stats, "proprio")
            self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )
            
        # T5 Embeddings Loading / Generation
        # TODO: This is dataset wise unified text description embedding, should be changed in the future
        t5_embeddings_path = os.path.join(self.data_dir, "t5_embeddings.pkl")
        if not os.path.exists(t5_embeddings_path):
            print(f"T5 embeddings not found at {t5_embeddings_path}. Generating...")
            # Convert set to list for stable ordering/iteration
            unique_commands_list = list(self.unique_commands)
            embeddings = generate_t5_embeddings(unique_commands_list)
            save_embeddings(embeddings, self.data_dir)
        
        # Load embeddings
        with open(t5_embeddings_path, "rb") as f:
            self.t5_text_embeddings = pickle.load(f)

        # Rollout mappings
        self.rollout_data = {}
        self.rollout_episode_metadata = {}
        self._rollout_success_step_to_episode_map = {}
        self._rollout_failure_step_to_episode_map = {}
        self._rollout_success_total_steps = 0
        self._rollout_failure_total_steps = 0
        
        self._calculate_epoch_structure()

    def __getitem__(self, idx):
        """
        Fetches images and action chunk sample by index.
        Returns action chunk rather than just single-step action.
        AnyTask / LIBERO style: 2 cameras (Wrist + Primary).
        Sequence: Blank, Proprio, Wrist, Primary, Action, FutProp, FutWrist, FutPrim, Value
        Total length: 9 frames.
        """
        # Determine which dataset to sample from based on index ranges
        sample_type = determine_sample_type(idx, self.adjusted_demo_count, self.adjusted_success_rollout_count)

        if sample_type == "demo":
            global_step_idx = idx % self.num_steps
            # Using global step index, get episode index and relative step index within that episode
            episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
            episode_metadata = None
            episode_data = self.data[episode_idx]
            global_rollout_idx = -1
        elif sample_type == "success_rollout":
            success_idx = idx - self.adjusted_demo_count
            global_rollout_idx = success_idx % max(1, getattr(self, "_rollout_success_total_steps", 1))
            episode_idx, relative_step_idx = self._rollout_success_step_to_episode_map[global_rollout_idx]
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            episode_data = self._load_rollout_episode_data(episode_metadata)
        else:
            sample_type = "failure_rollout"
            failure_idx = idx - self.adjusted_demo_count - self.adjusted_success_rollout_count
            global_rollout_idx = failure_idx % max(1, getattr(self, "_rollout_failure_total_steps", 1))
            episode_idx, relative_step_idx = self._rollout_failure_step_to_episode_map[global_rollout_idx]
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            episode_data = self._load_rollout_episode_data(episode_metadata)

        # Value function sampling logic
        is_world_model_sample = False
        is_value_function_sample = False
        if sample_type != "demo":
            if self.return_value_function_returns:
                p_world_model = 0.5
                if random.random() < p_world_model:
                    is_world_model_sample = True
                    is_value_function_sample = False
                else:
                    is_world_model_sample = False
                    is_value_function_sample = True
            else:
                is_world_model_sample = True
                is_value_function_sample = False

        # Lazy-load videos if needed
        if episode_data.get("is_lazy_video", False) and (
            ("images" not in episode_data) or (episode_data["images"] is None)
        ):
            video_paths = episode_data["video_paths"]
            images = load_video_as_images(video_paths["cam_high"], resize_size=self.final_image_size)  # uint8
            wrist_images = load_video_as_images(
                video_paths["cam_wrist"], resize_size=self.final_image_size
            )  # uint8
            episode_data["images"] = images
            episode_data["wrist_images"] = wrist_images
            episode_data["is_lazy_video"] = False

        # Calculate future frame index
        future_frame_idx = relative_step_idx + self.chunk_size
        max_possible_idx = episode_data["num_steps"] - 1
        if future_frame_idx > max_possible_idx:
            future_frame_idx = max_possible_idx

        # Retrieve images (Eager or Lazy loaded above)
        # Note: AnyTask uses MP4s, not JPEGs inside HDF5 usually, so logic is simpler than ALOHA/LIBERO mix
        current_image = episode_data["images"][relative_step_idx]
        current_wrist_image = episode_data["wrist_images"][relative_step_idx]
        
        future_image = episode_data["images"][future_frame_idx]
        future_wrist_image = episode_data["wrist_images"][future_frame_idx]

        # Build mapping of segment indices for different modalities
        # Logic: we track which index corresponds to which frame in the temporal sequence
        segment_idx = 0
        action_latent_idx = -1
        value_latent_idx = -1
        current_proprio_latent_idx = -1
        current_wrist_image_latent_idx = -1
        current_image_latent_idx = -1
        future_proprio_latent_idx = -1
        future_wrist_image_latent_idx = -1
        future_image_latent_idx = -1

        # Initialize list to store all images for the sequence
        image_list = []
        
        # 1. Blank first input frame (for tokenizer)
        first_input_image = np.expand_dims(np.zeros_like(current_image), axis=0)
        image_list.append(first_input_image)
        segment_idx += 1

        # 2. Current Proprio (blank image)
        if self.use_proprio:
            current_proprio_latent_idx = segment_idx
            # We add dummy image for proprio, values injected later
            blank_image = np.zeros_like(current_image)
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            segment_idx += 1

        # 3. Current Wrist Image
        if self.use_wrist_images:
            current_wrist_image_latent_idx = segment_idx
            wrist_img_dup = duplicate_array(current_wrist_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(wrist_img_dup)
            segment_idx += 1

        # 4. Current Primary Image
        if self.use_third_person_images:
            current_image_latent_idx = segment_idx
            curr_img_dup = duplicate_array(current_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(curr_img_dup)
            segment_idx += 1

        # 5. Action (blank image)
        action_latent_idx = segment_idx
        blank_image = np.zeros_like(current_image)
        blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
        image_list.append(blank_image)
        segment_idx += 1

        # 6. Future Proprio (blank image)
        if self.use_proprio:
            future_proprio_latent_idx = segment_idx
            blank_image = np.zeros_like(current_image)
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            segment_idx += 1
        
        # 7. Future Wrist Image
        if self.use_wrist_images:
            future_wrist_image_latent_idx = segment_idx
            fut_wrist_dup = duplicate_array(future_wrist_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(fut_wrist_dup)
            segment_idx += 1

        # 8. Future Primary Image
        if self.use_third_person_images:
            future_image_latent_idx = segment_idx
            fut_img_dup = duplicate_array(future_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(fut_img_dup)
            segment_idx += 1

        # 9. Value (blank image)
        if self.return_value_function_returns:
            value_latent_idx = segment_idx
            blank_image = np.zeros_like(current_image)
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            segment_idx += 1

        # Stack images and preprocess
        images = np.concatenate(image_list, axis=0)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        # Calculate action chunk
        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # Get return (value)
        if self.return_value_function_returns:
            # Simple retrieval for now
            # If "returns" is in metadata or data
            if episode_metadata is not None and "returns" in episode_metadata:
                value_function_return = episode_metadata["returns"][future_frame_idx]
            elif "returns" in episode_data:
                value_function_return = episode_data["returns"][future_frame_idx]
            else:
                value_function_return = float("-100") # Placeholder
        else:
            value_function_return = float("-100")

        # T5 Embeddings (Dummy)
        # We need to ensure we return a valid tensor.
        # self.t5_text_embeddings is a dummy dict we created in __init__
        t5_text_embeddings = self.t5_text_embeddings[episode_data["command"]]
        t5_text_mask = torch.ones(t5_text_embeddings.shape[0], dtype=torch.bool) # All valid

        # Create sample dict
        sample = {
            "video": images,
            "actions": action_chunk,
            "t5_text_embeddings": t5_text_embeddings,
            "t5_text_mask": t5_text_mask,
            "fps": torch.tensor(16, dtype=torch.int64), 
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "num_frames": torch.tensor(episode_data["num_steps"], dtype=torch.int64),
            "image_size": self.final_image_size * torch.ones(4),
            "rollout_data_mask": torch.tensor(1 if sample_type != "demo" else 0, dtype=torch.bool),
            "rollout_data_success_mask": torch.tensor(1 if sample_type == "success_rollout" else 0, dtype=torch.bool),
            "global_rollout_idx": torch.tensor(global_rollout_idx, dtype=torch.int64),
            "value_function_return": torch.tensor(value_function_return, dtype=torch.float32),
            "world_model_sample_mask": torch.tensor(1 if is_world_model_sample else 0, dtype=torch.int64),
            "value_function_sample_mask": torch.tensor(1 if is_value_function_sample else 0, dtype=torch.int64),
            "action_latent_idx": torch.tensor(action_latent_idx, dtype=torch.int64),
            "value_latent_idx": torch.tensor(value_latent_idx, dtype=torch.int64),
            "current_proprio_latent_idx": torch.tensor(current_proprio_latent_idx, dtype=torch.int64),
            "current_wrist_image_latent_idx": torch.tensor(current_wrist_image_latent_idx, dtype=torch.int64),
            "current_image_latent_idx": torch.tensor(current_image_latent_idx, dtype=torch.int64),
            "future_proprio_latent_idx": torch.tensor(future_proprio_latent_idx, dtype=torch.int64),
            "future_wrist_image_latent_idx": torch.tensor(future_wrist_image_latent_idx, dtype=torch.int64),
            "future_image_latent_idx": torch.tensor(future_image_latent_idx, dtype=torch.int64),
        }
        
        if self.use_proprio:
             sample["proprio"] = torch.tensor(episode_data["proprio"][relative_step_idx], dtype=torch.float32)
             sample["future_proprio"] = torch.tensor(episode_data["proprio"][future_frame_idx], dtype=torch.float32)

        return sample
