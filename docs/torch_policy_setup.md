# Required Torch Policy Setup From JAX Checkpoints

The guided release path requires the PyTorch pi0 policy implementation. Collision and semantic guidance are implemented in `pi0_pytorch.py`, not the JAX policy path.

The policy server loads PyTorch weights from `checkpoint_dir`, which defaults to `torchckpts/pi05droid`. If `torchckpts/pi05droid/model.safetensors` is not already present, convert the upstream JAX checkpoint once and keep the converted checkpoint in that directory.

## Install

From the repository root:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

## Install the Transformers Patch

The PyTorch pi0 implementation requires the local `transformers_replace` patch before conversion and serving:

```bash
uv pip install transformers==4.53.2
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

## Download the JAX Checkpoint

Use OpenPI's checkpoint downloader:

```bash
uv run python - <<'PY'
from openpi.shared import download

download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
PY
```

This writes the checkpoint to:

```text
~/.cache/openpi/openpi-assets/checkpoints/pi05_droid
```

## Convert to PyTorch

```bash
mkdir -p torchckpts/pi05droid

PYTHONPATH="$PWD:${PYTHONPATH:-}" uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
  --config_name pi05_droid_jointpos \
  --output_path torchckpts/pi05droid
```

Conversion can use substantial RAM. On a cluster, request enough memory before running it, for example:

```bash
srun --gpus=1 --nodes=1 --mem=128G --cpus-per-task=8 --time=12:00:00 --pty bash
```

## Copy Norm Stats

The policy also needs DROID normalization stats next to the converted checkpoint:

```bash
mkdir -p torchckpts/pi05droid/assets/droid
cp ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json \
  torchckpts/pi05droid/assets/droid/norm_stats.json
```

## Validate

```bash
test -f torchckpts/pi05droid/model.safetensors
test -f torchckpts/pi05droid/assets/droid/norm_stats.json
```

## Serve the Converted Policy

For simulation:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run server.py real_robot=false checkpoint_dir=torchckpts/pi05droid guidance=collision_sim
```

For the real robot, use `real_robot=true` so the server uses the real-robot action mode:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run server.py real_robot=true checkpoint_dir=torchckpts/pi05droid
```
