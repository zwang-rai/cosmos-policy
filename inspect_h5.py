import h5py
import numpy as np

file_path = "data/skillgen_1000_replay/episode_0/episode_0.h5"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: {obj.shape} ({obj.dtype})")
        if name == "action":
            print(f"  Sample action[0]: {obj[0]}")
        if name == "joint_pos":
            print(f"  Sample joint_pos[0]: {obj[0]}")
    else:
        print(f"{name}/")

try:
    with h5py.File(file_path, "r") as f:
        print(f"Inspecting File: {file_path}")
        f.visititems(print_structure)
except Exception as e:
    print(f"Error: {e}")
