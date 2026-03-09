import os
import time
import datetime
import argparse
import numpy as np

import sys
# Hack: cosmos_policy/constants.py parses sys.argv on import for "aloha" to set ACTION_DIM=14
sys.argv.append("aloha")

from cosmos_policy.experiments.robot.aloha.deploy import DeployConfig, validate_config
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_model,
    load_dataset_stats,
    init_t5_text_embeddings_cache,
    get_action,
)
from cosmos_policy.utils.utils import set_seed_everywhere

import cv2
import h5py

sys.argv.remove("aloha")

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a pretrained ALOHA checkpoint on a Franka sample.")
    parser.add_argument("--ckpt_path", type=str, default="nvidia/Cosmos-Policy-ALOHA-Predict2-2B")
    parser.add_argument("--dataset_path", type=str, default="data/small_test_set/vpl_processed/episode_0/episode_0.h5")
    parser.add_argument("--stats_path", type=str, default="data/small_test_set/vpl_processed/dataset_statistics.json")
    parser.add_argument("--t5_path", type=str, default="data/small_test_set/vpl_processed/t5_embeddings.pkl")
    
    args = parser.parse_args()
    args.dataset_path = os.path.abspath(args.dataset_path)
    args.stats_path = os.path.abspath(args.stats_path)
    args.t5_path = os.path.abspath(args.t5_path)
    return args


def decode_single_jpeg_frame(jpeg_bytes):
    # jpeg_bytes is a uint8 array from hdf5
    frame = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    # BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def main():
    args = parse_args()
    
    # Standard inference config for ALOHA
    cfg = DeployConfig(
        suite="aloha",    # Pretend it is aloha to match the model architecture
        config="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only",
        ckpt_path=args.ckpt_path,
        use_third_person_image=True,
        use_wrist_image=True,
        num_wrist_images=2,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        dataset_stats_path=args.stats_path,
        t5_text_embeddings_path=args.t5_path,
        trained_with_image_aug=True,
        chunk_size=50,
        num_open_loop_steps=50,
        seed=195,
        deterministic=True,
    )

    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    
    print("Loading T5 Embeddings Cache...")
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    
    print("Loading Dataset Statistics...")
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    
    print("Loading Model from Checkpoint...")
    model, cosmos_config = get_model(cfg)
    
    print(f"Loading first frame from dataset: {args.dataset_path}")
    with h5py.File(args.dataset_path, "r") as f:
        # Load the instruction
        task_description = f.attrs.get("task_description", "lift the green cubes up")
        
        # Load exactly the first frame
        curr_frames = f["color"][0] # Array of bytes for cams
        primary_jpeg = curr_frames[0]
        left_wrist_jpeg = curr_frames[1]
        right_wrist_jpeg = curr_frames[2]
        
        primary_image = decode_single_jpeg_frame(primary_jpeg)
        left_wrist_image = decode_single_jpeg_frame(left_wrist_jpeg)
        right_wrist_image = decode_single_jpeg_frame(right_wrist_jpeg)
        
        # Proprio (first step)
        left_joint_pos = f["left_joint_positions"][0]
        left_gripper_width = f["left_gripper_width"][0]
        right_joint_pos = f["right_joint_positions"][0]
        right_gripper_width = f["right_gripper_width"][0]
        proprio = np.concatenate([left_joint_pos, [left_gripper_width], right_joint_pos, [right_gripper_width]])
        
        # ALOHA policy expects a 14-dim proprio.
        # If the Franka's proprio is 16-dim (7 joint + 1 gripper per arm), we print a warning and truncate it to 14
        # just to run inference on the pretrained ALOHA checkpoint. 
        if proprio.shape[0] == 16:
            print("WARNING: Franka proprio is 16D, but ALOHA model expects 14D. Truncating to 14D for inference.")
            proprio = proprio[:14]
            
        # Do the same for original dataset stats to avoid dimension mismatch shape errors during normalization
        if dataset_stats["proprio_mean"].shape[0] == 16:
            for k in ["proprio_mean", "proprio_std", "proprio_min", "proprio_max"]:
                dataset_stats[k] = dataset_stats[k][:14]
            for k in ["actions_mean", "actions_std", "actions_min", "actions_max"]:
                dataset_stats[k] = dataset_stats[k][:14]

    observation = {
        "primary_image": np.array(primary_image, dtype=np.uint8),
        "left_wrist_image": np.array(left_wrist_image, dtype=np.uint8),
        "right_wrist_image": np.array(right_wrist_image, dtype=np.uint8),
        "proprio": np.array(proprio, dtype=np.float32),
        "task_description": task_description,
    }

    print("\nRunning Model Inference...")
    start_time = time.time()
    return_dict = get_action(
        cfg=cfg,
        model=model,
        dataset_stats=dataset_stats,
        obs=observation,
        task_label_or_embedding=task_description,
        seed=cfg.seed,
    )
    query_time = time.time() - start_time
    
    print(f"\nModel Query Completed in {query_time:.3f} seconds.")
    print("Predicted Actions Shape:", return_dict["actions"][0].shape)
    print("Predicted Actions (first 3 steps):")
    print(return_dict["actions"][0][:3])
    
    if "future_image_predictions" in return_dict:
        predictions = return_dict["future_image_predictions"]
        if isinstance(predictions, list):
            predictions = predictions[0] # batch dim
        
        # Create output directory with date string
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join("demo_output", date_str)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nSaving predicted images to: {out_dir}")
        
        # Generate Gallery Image
        # Top Row: Observations
        # Bottom Row: Predictions
        
        # Original observations (RGB)
        obs_primary = observation["primary_image"]
        obs_left = observation["left_wrist_image"]
        obs_right = observation["right_wrist_image"]
        
        # Predictions (RGB)
        pred_primary = predictions.get("future_image")
        pred_left = predictions.get("future_wrist_image")
        pred_right = predictions.get("future_wrist_image2")
        
        if pred_primary is not None and pred_left is not None and pred_right is not None:
            # Resize primary to match wrist images height if needed, 
            # or just resize all to 256x256 for a clean grid
            def resize_to_square(img, size=224):
                return cv2.resize(img, (size, size))
            
            top_row = np.concatenate([
                resize_to_square(obs_left), 
                resize_to_square(obs_right), 
                resize_to_square(obs_primary)
            ], axis=1)
            
            bottom_row = np.concatenate([
                resize_to_square(pred_left), 
                resize_to_square(pred_right), 
                resize_to_square(pred_primary)
            ], axis=1)
            
            # Combine into 2-row gallery
            gallery_rgb = np.concatenate([top_row, bottom_row], axis=0)
            
            # Convert RGB to BGR for cv2 saving
            gallery_bgr = cv2.cvtColor(gallery_rgb, cv2.COLOR_RGB2BGR)
            
            save_path = os.path.join(out_dir, "gallery.jpg")
            cv2.imwrite(save_path, gallery_bgr)
            print(f"  Saved {save_path}")
        else:
            print("  Could not generate gallery: missing future image predictions.")

        # Save prompt and predicted actions to a text file
        info_path = os.path.join(out_dir, "prompt_and_actions.txt")
        with open(info_path, "w") as f:
            f.write(f"Prompt: {task_description}\n\n")
            f.write(f"Predicted Actions (Shape: {return_dict['actions'][0].shape}):\n")
            f.write(f"{return_dict['actions'][0]}\n")
        print(f"  Saved {info_path}")

if __name__ == "__main__":
    main()
