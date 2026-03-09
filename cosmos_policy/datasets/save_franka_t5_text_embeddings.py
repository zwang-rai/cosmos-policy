# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Precomputes T5 text embeddings for Franka task descriptions and saves them to disk.
Alternatively, allows generating an embedding for a specific task description string.

Usage (Bulk Generation from Dataset):
    uv run -m cosmos_policy.datasets.save_franka_t5_text_embeddings [--data_dir DATA_DIR] [--is_dual_arm]

Usage (Single String Generation):
    uv run -m cosmos_policy.datasets.save_franka_t5_text_embeddings --data_dir DATA_DIR --task_description "pick up the red block"
"""

import argparse

from cosmos_policy.datasets.craft_dataset import CraftDataset
from cosmos_policy.datasets.t5_embedding_utils import (
    generate_t5_embeddings,
    save_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute T5 text embeddings for Franka task descriptions")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/small_test_set/vpl_processed",
        help="Directory containing Franka dataset or where embeddings should be saved",
    )
    parser.add_argument(
        "--is_dual_arm",
        action="store_true",
        help="Flag to parse the dataset using the dual arm configuration.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default=None,
        help="Specific task description to encode (overrides dataset scanning).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir

    if args.task_description is not None:
        print(f"Generating T5 embeddings for single description: '{args.task_description}'")
        unique_commands = [args.task_description]
    else:
        print("Scanning dataset for unique commands...")
        dataset = CraftDataset(
            data_dir=data_dir,
            is_dual_arm=args.is_dual_arm,
        )
        unique_commands = list(dataset.unique_commands)
        if len(unique_commands) == 0:
            print("No commands found in the dataset. Exiting.")
            return

    t5_text_embeddings = generate_t5_embeddings(unique_commands)
    save_embeddings(t5_text_embeddings, data_dir)


if __name__ == "__main__":
    main()
