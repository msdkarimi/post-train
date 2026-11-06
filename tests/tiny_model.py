"""Build a tiny RANDOM-WEIGHT model with the Qwen3.5 architecture (hybrid linear/full attention) plus the real
tokenizer, so the training code path can be exercised in seconds on a laptop CPU.

    python -m tests.tiny_model --out data/tiny_model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.5-2B-Base"


def build_tiny_model(out_dir: str) -> str:
    cfg = AutoConfig.from_pretrained(BASE)
    tc = cfg.text_config
    tc.hidden_size = 64
    tc.intermediate_size = 128
    tc.num_hidden_layers = 4
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.head_dim = 16
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 2
    tc.linear_key_head_dim = 16
    tc.linear_value_head_dim = 16
    tc.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    tc.dtype = "float32"
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(tc)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
    n = sum(p.numel() for p in model.parameters())
    print(f"tiny model: {type(model).__name__}, {n / 1e6:.1f}M params -> {out}")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tiny_model")
    build_tiny_model(ap.parse_args().out)
