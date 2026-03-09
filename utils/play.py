import random
import numpy as np
from cosmos_policy.datasets.vpl_dataset import VPLDataset
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

import torch
from PIL import Image
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    load_dataset_stats,
    init_t5_text_embeddings_cache,
    get_t5_embedding_from_cache,
)

# Instantiate config (see PolicyEvalConfig in cosmos_policy/experiments/robot/libero/run_libero_eval.py for definitions)
cfg = PolicyEvalConfig(
    config="cosmos_predict2_2b_480p_libero__inference_only",
    ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    config_file="cosmos_policy/config/config.py",
    dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
    t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
    use_wrist_image=True,
    use_proprio=True,
    normalize_proprio=True,
    unnormalize_actions=True,
    chunk_size=16,
    num_open_loop_steps=16,
    trained_with_image_aug=True,
    use_jpeg_compression=True,
    flip_images=False,  # Only for LIBERO; images render upside-down
    num_denoising_steps_action=5,
    num_denoising_steps_future_state=1,
    num_denoising_steps_value=1,
)
# Load dataset stats for action/proprio scaling
dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
# Initialize T5 text embeddings cache
init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
# Load model
model, cosmos_config = get_model(cfg)

# Initialize VPLDataset
print("Initializing VPLDataset...")
dataset = VPLDataset(
    data_dir="data/skillgen_1000_replay",
    is_train=True,
    normalize_images=False, # We want raw uint8 images for get_action
    use_proprio=True,
    use_wrist_images=True,
    use_third_person_images=True,
    debug=True, # Load all data
)

# Randomly select and load an episode
episode_idx = random.randint(0, dataset.num_episodes - 1)
episode_data = dataset.data[episode_idx]
num_steps = episode_data["num_steps"]
# Ensure step_idx + 1 (future) is valid? No, get_action doesn't need future GT.
step_idx = random.randint(0, num_steps - 1)

print(f"Selected Episode: {episode_idx} (File: {episode_data['file_path']})")
print(f"Selected Step: {step_idx}/{num_steps-1}")

primary_image = episode_data["images"][step_idx]
wrist_image = episode_data["wrist_images"][step_idx]
proprio = episode_data["proprio"][step_idx]
task_description = episode_data["command"]

observation = {
    "primary_image": primary_image,
    "wrist_image": wrist_image,
    "proprio": proprio,
}

print(f"Task: {task_description}")

# Generate robot actions, future state (proprio + images), and value
if True:
    action_return_dict = get_action(
        cfg,
        model,
        dataset_stats,
        observation,
        task_description,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=True,
    )
    # Print actions
    print(action_return_dict.keys())
    print(f"Generated action chunk shape: {len(action_return_dict['actions'])}, {action_return_dict['actions'][0].shape}")
    # print(f"Generated action chunk: {action_return_dict['actions']}")
    
    # Save future image predictions (third-person image and wrist image)
    # Create a figure gallery
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Upper row: Observations
    axes[0, 0].imshow(primary_image)
    axes[0, 0].set_title("Observation: Primary")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(wrist_image)
    axes[0, 1].set_title("Observation: Wrist")
    axes[0, 1].axis('off')
    
    # Lower row: Predictions
    if 'future_image' in action_return_dict['future_image_predictions']:
        axes[1, 0].imshow(action_return_dict['future_image_predictions']['future_image'])
        axes[1, 0].set_title("Prediction: Future Primary")
    else:
        axes[1, 0].text(0.5, 0.5, "No Prediction", ha='center')
    axes[1, 0].axis('off')

    if 'future_wrist_image' in action_return_dict['future_image_predictions']:
        axes[1, 1].imshow(action_return_dict['future_image_predictions']['future_wrist_image'])
        axes[1, 1].set_title("Prediction: Future Wrist")
    else:
        axes[1, 1].text(0.5, 0.5, "No Prediction", ha='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("inference_gallery.png")
    print("Saved inference gallery to: inference_gallery.png")

    # Print value
    print(f"Generated value: {action_return_dict['value_prediction']}")

    # Print value
    print(f"Generated value: {action_return_dict['value_prediction']}")