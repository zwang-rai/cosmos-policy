## Quick Start

### Build the Docker Image

From the project root, build the image:

```bash
docker build -t cosmos-policy docker
```

### Launch the Docker Container

Start an interactive session:

```bash
docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/cosmos/.cache \
  -v $(pwd):/workspace \
  --gpus all \
  --ipc=host \
  -it \
  --rm \
  -w /workspace \
  --entrypoint bash \
  cosmos-policy
```

**Optional arguments:**
* `-v $HOME/.cache:/home/cosmos/.cache`: Reuses host cache for `uv`, `huggingface`, etc.
* `--ipc=host`: Shares host's inter-process communication namespace. Required for PyTorch's parallel data loading to avoid out-of-memory errors (containers get only 64MB shared memory by default). If your security policy doesn't allow this, try using `--shm-size 32g` instead to allocate sufficient isolated shared memory.

### Fine-tuning the Policy Model

To start training, run the following command from within the docker container:

```bash
uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.scripts.train \
  --config cosmos_policy/config/config_v2.py \
  -- \
  experiment=cosmos_predict2_2b_480p_anytask
```

**Note:** The `--` separator is required before specifying the `experiment` override.

To do a dry run to verify your configuration without starting training:

```bash
uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.scripts.train \
  --config cosmos_policy/config/config_v2.py \
  --dryrun \
  -- \
  experiment=cosmos_predict2_2b_480p_anytask
```
