#!/usr/bin/env bash

set -euo pipefail

WITH_VLLM="${WITH_VLLM:1}"
WITH_TRAIN="${WITH_TRAIN:1}"
WITH_CAUSAL_CONV1D="${WITH_CAUSAL_CONV1D:1}"
VENV_DIR="${VENV_DIR:-$HOME/venv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"  # must match driver CUDA (12.8)
REPO_URL="${REPO_URL:-https://github.com/msdkarimi/post-train.git}"   # public repo: no credentials needed
[ -n "${REPO_DIR:-}" ] && REPO_DIR_SET=1   # user passed REPO_DIR explicitly: don't auto-detect
REPO_DIR="${REPO_DIR:-$HOME/post-train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# --- 1. GPU check -------------------------------------------------------------
log "GPU check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# --- 2. System packages + Docker ---------------------------------------------
log "System packages + Docker"
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y docker.io git tmux htop build-essential curl
else
  sudo apt-get install -y git tmux htop build-essential curl >/dev/null
fi
sudo systemctl enable --now docker >/dev/null 2>&1 || true
# Takes effect on next login; the script itself uses sudo for docker.
sudo usermod -aG docker "$USER" || true

# --- 3. uv --------------------------------------------------------------------
log "uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# --- 4. Python env ------------------------------------------------------------
log "Python env at $VENV_DIR"
[ -d "$VENV_DIR" ] || uv venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# torch FIRST, from the cu128 index, so nothing later pulls a newer CUDA build.
uv pip install torch --index-url "$TORCH_INDEX"
TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
printf 'torch==%s*\n' "$TORCH_VER" > /tmp/torch-constraint.txt

uv pip install -c /tmp/torch-constraint.txt \
  transformers accelerate flash-linear-attention

# causal-conv1d: optional fast kernel for Qwen3.5's Gated DeltaNet layers.
# Its setup first tries to download a prebuilt wheel; if none matches your torch/CUDA it
# compiles from source (needs nvcc, can take 10+ min of paid time). Failure is NOT fatal:
# transformers falls back to the slower reference code. Disable with WITH_CAUSAL_CONV1D=0.
if [ "$WITH_CAUSAL_CONV1D" = "1" ]; then
  log "causal-conv1d (optional)"
  uv pip install -c /tmp/torch-constraint.txt packaging ninja wheel setuptools
  MAX_JOBS="${MAX_JOBS:-8}" uv pip install -c /tmp/torch-constraint.txt \
    --no-build-isolation causal-conv1d \
    || echo "causal-conv1d install failed - continuing with the slower fallback kernel"
fi

if [ "$WITH_TRAIN" = "1" ]; then
  uv pip install -c /tmp/torch-constraint.txt trl peft datasets wandb huggingface_hub
fi

if [ "$WITH_VLLM" = "1" ]; then
  log "vLLM (may replace torch; --torch-backend keeps it on cu128)"
  uv pip install --torch-backend=cu128 vllm || echo "vLLM install failed; continue without it"
fi

# --- 5a. Clone / update the (public) repo ---------------------------------------
# If this script already lives inside a git checkout (e.g. ~/post-train/bootstrap.sh),
# use that checkout as the repo root instead of cloning again.
if [ -z "${REPO_DIR_SET:-}" ] && TOPLEVEL="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  REPO_DIR="$TOPLEVEL"
fi
log "Repo: $REPO_DIR"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only || echo "git pull failed; continuing with existing checkout"
elif [ -d "$REPO_DIR" ] && [ -n "$(ls -A "$REPO_DIR" 2>/dev/null)" ]; then
  # Folder exists but is not a checkout - typically because .env was scp'd into it first.
  # `git clone` refuses a non-empty folder, so attach the repo's .git to it in place.
  # Only tracked files are touched; .env (untracked, gitignored) is left alone.
  echo "$REPO_DIR exists and is not a git checkout - cloning into it in place"
  TMP_CLONE="$(mktemp -d)"
  git clone --quiet --no-checkout "$REPO_URL" "$TMP_CLONE/r"
  mv "$TMP_CLONE/r/.git" "$REPO_DIR/.git"
  rm -rf "$TMP_CLONE"
  git -C "$REPO_DIR" reset --hard --quiet HEAD
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

# --- 5b. Load tokens from .env (if present) -------------------------------------
# .env lives at the repo root. Search order: repo root, script folder, its parent,
# current folder, $HOME.
ENV_FILE=""
for f in "$REPO_DIR/.env" "$SCRIPT_DIR/.env" "$SCRIPT_DIR/../.env" "$PWD/.env" "$HOME/.env"; do
  if [ -f "$f" ]; then ENV_FILE="$(cd "$(dirname "$f")" && pwd)/.env"; break; fi
done

# Safe loader: handles CRLF line endings, comments, blank lines, `export KEY=...` and
# quoted values, and never executes the file as shell code.
load_env_file() {
  local line key val
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in ''|'#'*|[[:space:]]*'#'*) continue ;; esac
    line="${line#export }"
    key="${line%%=*}"; val="${line#*=}"
    if ! [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "  skipping malformed .env line (key not shown)"; continue
    fi
    val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
    export "$key=$val"
  done < "$1"
}

if [ -n "$ENV_FILE" ]; then
  log "Loading $ENV_FILE"
  load_env_file "$ENV_FILE"
  for k in HF_TOKEN WANDB_API_KEY; do
    if [ -n "${!k:-}" ]; then echo "  $k: set"; else echo "  $k: MISSING in $ENV_FILE"; fi
  done
  [ -n "${HF_TOKEN:-}" ] && python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])" 2>/dev/null || echo "login to HF failed!"
else
  echo "No .env found (looked in $REPO_DIR, $SCRIPT_DIR, $PWD, $HOME)."
  echo "Copy one with: scp .env <vm>:$REPO_DIR/.env   (gated models/datasets and W&B need it)"
fi

# --- 6. Persist activation for new shells --------------------------------------
if ! grep -qF "$VENV_DIR/bin/activate" "$HOME/.bashrc"; then
  echo "source $VENV_DIR/bin/activate" >> "$HOME/.bashrc"
fi
if [ -n "$ENV_FILE" ] && ! grep -qF "$ENV_FILE" "$HOME/.bashrc"; then
  printf '[ -f %q ] && { set -a; . %q; set +a; }\n' "$ENV_FILE" "$ENV_FILE" >> "$HOME/.bashrc"
fi

# --- 7. Verification -------------------------------------------------------------
log "Verify: torch sees the GPU"
python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available: torch build does not match driver"
print("device:", torch.cuda.get_device_name(0))
EOF

log "Verify: Docker sandbox has no network"
sudo docker pull -q python:3.12-slim >/dev/null
if sudo docker run --rm --network none python:3.12-slim \
     python -c "import urllib.request; urllib.request.urlopen('http://example.com', timeout=3)" >/dev/null 2>&1; then
  echo "WARNING: container reached the internet - isolation is NOT working"; exit 1
else
  echo "OK: network blocked inside the container"
fi

log "Verify: inference (load model, generate a few tokens)"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B-Base}" python - <<'EOF'
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
name = os.environ["MODEL_NAME"]
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda()
ids = tok("The capital of France is", return_tensors="pt").to("cuda")
out = model.generate(**ids, max_new_tokens=20, do_sample=False)
text = tok.decode(out[0], skip_special_tokens=True)
print(text)
assert "Paris" in text, "inference sanity check failed: expected 'Paris' in output"
print("peak GPU mem GB:", round(torch.cuda.max_memory_allocated() / 1e9, 2))
EOF

log "BOOTSTRAP OK (exit 0). Docker group + venv are active in NEW shells (source ~/.bashrc)."
echo "REMINDER: delete the VM in the console when finished - billing runs until it is deleted."
