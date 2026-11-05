#!/usr/bin/env bash
# train.sh - runs only after bootstrap.sh succeeded (see run.sh).
# PLACEHOLDER: the Python training scripts do not exist yet. Replace the commands below.
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venv}"
REPO_DIR="${REPO_DIR:-$HOME/post-train}"

# bootstrap.sh ran in a separate process, so activate the env here.
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
[ -f .env ] && { set -a; source .env; set +a; }

# Make sure checkpoints survive a crash: write them somewhere you will copy off the VM.
mkdir -p outputs

echo "=== SFT ==="
python train/sft.py --output_dir outputs/sft        # TODO: write this script

echo "=== RL (GRPO) ==="
python train/grpo.py --init outputs/sft --output_dir outputs/rl   # TODO: write this script

echo "train.sh finished OK. Copy outputs/ off the VM, then DELETE the VM."
