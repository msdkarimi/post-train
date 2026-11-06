"""Trainer subclass that logs extra metrics (they show up in the console and in W&B automatically).

Extra metrics, on top of the Trainer defaults (loss, grad_norm, learning_rate, epoch, eval_*):
  train:  tokens_seen, trained_tokens_seen   cumulative tokens through the model / tokens that receive loss
          tokens_per_sec                     throughput since the previous log line
          gpu_mem_alloc_gb, gpu_mem_reserved_gb   peak CUDA memory so far
  eval:   eval_<source>_loss / _ppl          per data source (e.g. hermes vs smoltalk), perplexity = exp(loss)
          eval_loss / eval_ppl               token-weighted overall (exactly the loss over the whole val set)
Token counts are per process (multiply by the number of GPUs for the global figure).
"""
from __future__ import annotations

import math
import time

import torch
from transformers import Trainer


class SFTTrainer(Trainer):
    def __init__(self, *args, eval_weights: dict[str, float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_weights = eval_weights or {}      # trained-token count per eval group, for the overall eval loss
        self._tok_seen = 0
        self._trained_tok_seen = 0
        self._last_t = None
        self._last_tok = 0
        self._eval_buf: dict[str, float] = {}

    def training_step(self, model, inputs, *args, **kwargs):
        # inputs are still CPU tensors here (the base class moves them to the GPU), so counting costs no GPU sync
        if self._last_t is None:
            self._last_t = time.time()
        self._tok_seen += int(inputs["attention_mask"].sum())
        self._trained_tok_seen += int((inputs["labels"] != -100).sum())
        return super().training_step(model, inputs, *args, **kwargs)

    def log(self, logs: dict, *args, **kwargs):
        is_eval = any(k.startswith("eval_") for k in logs)
        if not is_eval and "loss" in logs:
            now = time.time()
            logs["tokens_seen"] = self._tok_seen
            logs["trained_tokens_seen"] = self._trained_tok_seen
            if self._last_t is not None and now > self._last_t:
                logs["tokens_per_sec"] = round((self._tok_seen - self._last_tok) / (now - self._last_t), 1)
            self._last_t, self._last_tok = now, self._tok_seen
            if torch.cuda.is_available():
                logs["gpu_mem_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                logs["gpu_mem_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)

        if is_eval:
            # The Trainer calls log() once per eval set, so collect per-source losses across calls and attach the
            # overall (token-weighted) figure to the call that completes the round.
            for n in self.eval_weights:
                if f"eval_{n}_loss" in logs:
                    self._eval_buf[n] = logs[f"eval_{n}_loss"]
            if self.eval_weights and set(self._eval_buf) == set(self.eval_weights):
                total = sum(self.eval_weights.values())
                logs["eval_loss"] = sum(self.eval_weights[n] * self._eval_buf[n] for n in self._eval_buf) / total
                self._eval_buf = {}
            for k in [k for k in logs if k.startswith("eval_") and k.endswith("_loss")]:
                logs[k[: -len("_loss")] + "_ppl"] = math.exp(min(logs[k], 20.0))
        return super().log(logs, *args, **kwargs)
