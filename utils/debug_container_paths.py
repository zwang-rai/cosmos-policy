import os
import glob
import h5py

def debug():
    print(f"CWD: {os.getcwd()}")
    data_dir = "data/small_test_set/vpl_processed"
    print(f"Checking data_dir: {data_dir}")
    print(f"Exists: {os.path.exists(data_dir)}")
    
    glob_pattern = os.path.join(data_dir, "episode_*", "*.h5")
    print(f"Glob pattern: {glob_pattern}")
    files = glob.glob(glob_pattern)
    print(f"Found {len(files)} files via glob")
    for f in files[:5]:
        print(f"  {f}")

    # Check for stats file
    stats_file = os.path.join(data_dir, "dataset_statistics.json")
    print(f"Stats file {stats_file} exists: {os.path.exists(stats_file)}")

if __name__ == "__main__":
    debug()
