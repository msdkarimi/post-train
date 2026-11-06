"""Run configuration for SFT: a dataclass filled from a YAML file, with `--set key=value` overrides."""
from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SFTRunConfig:
    # --- model / data / output -------------------------------------------------------------
    model_name: str = "Qwen/Qwen3.5-2B-Base"
    data_dir: str = "data/processed/sft_v1"      # output of `python -m etl.build_sft`
    output_dir: str = "outputs/sft"
    run_name: str = "sft"

    # --- method ---------------------------------------------------------------------------
    method: str = "lora"                          # "lora" (cheap, fits 24GB) | "full" (A100-80GB)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target: str = "all-linear"              # every linear layer except the LM head (robust to the hybrid arch)
    merge_lora: bool = True                      # also write merged bf16 weights (needed for eval / RL)

    # --- optimizer & schedule -------------------------------------------------------------
    optim: str = "adamw_torch_fused"             # or "adamw_bnb_8bit" (needs bitsandbytes) to save memory
    learning_rate: float = 1e-4                  # LoRA ~1e-4 .. 2e-4; full fine-tune ~1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine_with_min_lr"
    min_lr_ratio: float = 0.1                    # final LR = min_lr_ratio * peak LR
    warmup_ratio: float = 0.03
    num_epochs: float = 1.0
    max_steps: int = -1                          # >0 overrides num_epochs

    # --- batching -------------------------------------------------------------------------
    per_device_batch_size: int = 4
    grad_accum_steps: int = 8                    # effective batch = per_device * accum * n_gpus sequences
    group_by_length: bool = True                 # batch similar lengths together -> less padding
    num_workers: int = 2

    # --- precision / memory ---------------------------------------------------------------
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    attn_implementation: str = ""                # "" = library default; or "sdpa" / "flash_attention_2"

    # --- logging / eval / saving ----------------------------------------------------------
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    report_to: str = "wandb"                     # "wandb" (structured W&B logger) | "none"; none if WANDB_API_KEY is unset
    seed: int = 42
    max_train_minutes: float = 0                 # >0: stop (and save) after this many minutes - protects paid GPU time
    resume_from_checkpoint: str = ""             # path, or "auto" to resume the latest checkpoint in output_dir
    sample_after_train: bool = True              # generate one sample from the val set at the end (eyeball check)

    # --- generation logging (what does the model write?) ------------------------------------------
    gen_log_steps: int = 100                     # every N steps (+ once before training, + at the end); 0 = off
    gen_prompts_per_source: int = 3              # fixed val prompts per data source (hermes, smoltalk, ...)
    gen_max_new_tokens: int = 160                # more = slower rounds; tool calls need ~60-150


def _coerce(field_type, value):
    """Force a value to the dataclass field's declared type.

    Needed because PyYAML (YAML 1.1) reads `1e-5` / `3e-3` (no decimal point) as a STRING, which would
    otherwise reach the optimizer as learning_rate="1e-5" and crash.
    """
    t = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", str(field_type))
    if t == "float":
        return float(value)
    if t == "int":
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"expected an integer, got {value!r}")
        return int(value)
    if t == "bool":
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no"):
                return False
            raise ValueError(f"expected a boolean, got {value!r}")
        return bool(value)
    if t == "str":
        return "" if value is None else str(value)
    return value


def _cast(field_type, raw: str):
    return _coerce(field_type, yaml.safe_load(raw))   # "true" -> True, "8" -> 8, "3e-3" -> 0.003


def load_config(path: Optional[str], overrides: list[str]) -> SFTRunConfig:
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    data = data or {}
    valid = {f.name: f for f in fields(SFTRunConfig)}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if k not in valid:
            raise ValueError(f"unknown config key {k!r}; valid keys: {sorted(valid)}")
        data[k] = _cast(valid[k].type, v)
    unknown = set(data) - set(valid)
    if unknown:
        raise ValueError(f"unknown keys in config: {sorted(unknown)}; valid keys: {sorted(valid)}")
    data = {k: _coerce(valid[k].type, v) for k, v in data.items()}
    cfg = SFTRunConfig(**data)
    if cfg.method not in ("lora", "full"):
        raise ValueError("method must be 'lora' or 'full'")
    return cfg
