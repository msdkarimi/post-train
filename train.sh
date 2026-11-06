#!/usr/bin/env bash
# train.sh - build the SFT data (if missing) and run SFT. Runs after bootstrap.sh succeeded (see run.sh).
#
#   bash train.sh                              # LoRA SFT with train/configs/sft_lora.yaml
#   bash train.sh --set max_steps=20           # extra args go to train.sft (quick throughput check first!)
#   SFT_CONFIG=train/configs/sft_full.yaml bash train.sh     # full fine-tuning (A100-80GB)
#
# RL (GRPO) is not implemented yet.
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venv}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SFT_CONFIG="${SFT_CONFIG:-train/configs/sft_lora.yaml}"
MIX_CONFIG="${MIX_CONFIG:-etl/configs/sft_mix.yaml}"
DATA_DIR="${DATA_DIR:-data/processed/sft_v1}"      # must match `name:` in the mix config unless you pass --out

# bootstrap.sh ran in a separate process, so activate the env here.
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
[ -f .env ] && { set -a; source .env; set +a; }

if [ ! -d "$DATA_DIR/tokenized" ]; then
  echo "=== ETL: building $DATA_DIR from $MIX_CONFIG ==="
  python -m etl.build_sft --config "$MIX_CONFIG" --out "$DATA_DIR"
else
  echo "=== ETL: reusing existing $DATA_DIR (delete it to rebuild) ==="
fi

echo "=== SFT: $SFT_CONFIG ==="
python -m train.sft --config "$SFT_CONFIG" --set data_dir="$DATA_DIR" "$@"

echo "train.sh finished OK. Copy outputs/ off the VM (or upload to the Hub), then DELETE the VM."
