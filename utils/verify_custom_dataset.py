
import os
import torch
import numpy as np
from tqdm import tqdm
from cosmos_policy.datasets.vpl_dataset import VPLDataset

def verify_dataset():
    data_dir = "data/skillgen_1000_replay"
    
    print(f"Initializing VPLDataset from {data_dir}...")
    dataset = VPLDataset(
        data_dir=data_dir,
        is_train=True,
        chunk_size=50,
        use_image_aug=False,
        debug=True, # Load only first episode initially
    )
    
    print(f"Dataset initialized.")
    print(f"Num episodes: {dataset.num_episodes}")
    print(f"Num steps: {dataset.num_steps}")
    print(f"Unique commands: {dataset.unique_commands}")
    
    # Check a few samples
    # Fetch a sample
    sample = dataset[0]
    print("\nSample keys:", sample.keys())
    
    # Verify video shape with new defaults (Matching config)
    # Expected: 33 frames (1 Blank + 8 items * 4 duplicates)
    # 8 items = Proprio, Wrist, Primary, Action, FutProp, FutWrist, FutPrim, Value
    if "video" in sample:
        video = sample["video"]
        print(f"Video shape: {video.shape}")
        if video.shape[1] == 33: # C, T, H, W or T, C, H, W? User output was [3, 41..] suggesting dim 1 is time if dim 0 is 3 (channels)
             print("SUCCESS: Video has 33 frames as expected for 2-camera config.")
        elif video.shape[0] == 33:
             print("SUCCESS: Video has 33 frames as expected for 2-camera config.")
        else:
            print(f"WARNING: Video has {video.shape[1] if video.shape[0]==3 else video.shape[0]} frames (expected 33).")
            
    for k, v in sample.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            # print(f"  {k}: {v.shape} ({v.dtype})")
            if k == "wrist_images" and v is not None:
                 print(f"    Min: {v.min()}, Max: {v.max()}")
        if k == "actions":
            print(f"  Actions shape: {v.shape}")
            if v.shape[-1] == 7:
                 print("SUCCESS: Actions are 7-dim (Pos + Euler + Gripper).")
            else:
                 print(f"WARNING: Actions are {v.shape[-1]}-dim (expected 7).")
        if k == "proprio":
            print(f"  Proprio shape: {v.shape}")
            if v.shape[-1] == 9:
                 print("SUCCESS: Proprio is 9-dim (Joint Pos).")
            else:
                 print(f"WARNING: Proprio is {v.shape[-1]}-dim (expected 9).")

    print("\nVerification complete!")

if __name__ == "__main__":
    verify_dataset()
