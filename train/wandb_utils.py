"""Structured Weights & Biases logging.

W&B groups panels into collapsible sections by the part of the metric name before the first "/". The Hugging Face
W&B callback puts everything that is not an eval metric under `train/`; this module replaces it so that every
metric lands in a section that says what it is:

    train/        loss, ppl, epoch
    optim/        learning_rate, grad_norm
    eval/         loss, ppl                       (overall, token-weighted)
                  by_source/<source>/loss, ppl    (e.g. hermes = tool use, smoltalk = general chat)
    completion/   length_mean, stopped_rate, tool/{format_rate,name_match,args_match}, chat/...   (generation_callback)
    perf/         tokens_per_sec, gpu_mem_alloc_gb, gpu_mem_reserved_gb, train_runtime_s
    data/         tokens_seen, trained_tokens_seen, loss_token_fraction
    generations/  latest, history, final_merged   (tables with the model's actual outputs)
    system/       GPU / CPU / disk (collected by W&B itself)

Every point carries a `step` (optimizer step) and all metrics use it as their x axis, so train, eval and completion
curves line up. Console logging is unaffected.
"""
from __future__ import annotations

import dataclasses
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback

_EVAL_DROP = {"eval_runtime", "eval_samples_per_second", "eval_steps_per_second", "eval_model_preparation_time", "epoch"}
_TRAIN_MAP = {
    "loss": "train/loss", "epoch": "train/epoch",
    "learning_rate": "optim/learning_rate", "grad_norm": "optim/grad_norm",
    "tokens_per_sec": "perf/tokens_per_sec", "gpu_mem_alloc_gb": "perf/gpu_mem_alloc_gb",
    "gpu_mem_reserved_gb": "perf/gpu_mem_reserved_gb",
    "tokens_seen": "data/tokens_seen", "trained_tokens_seen": "data/trained_tokens_seen",
    # the summary line the Trainer logs once at the end of training
    "train_runtime": "perf/train_runtime_s", "train_loss": "train/loss_run_average",
}
_END_DROP = {"train_samples_per_second", "train_steps_per_second", "total_flos"}


def structure_logs(logs: dict) -> dict:
    """Map a Trainer log dict to hierarchical W&B names (pure function, unit-tested)."""
    out: dict[str, float] = {}
    for k, v in logs.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if k.startswith("eval_"):
            if k in _EVAL_DROP:
                continue
            name = k[len("eval_"):]
            if name in ("loss", "ppl"):
                out[f"eval/{name}"] = v
            else:
                m = re.fullmatch(r"(.+)_(loss|ppl)", name)
                out[f"eval/by_source/{m.group(1)}/{m.group(2)}" if m else f"eval/misc/{name}"] = v
        elif k in _END_DROP:
            continue
        elif k in _TRAIN_MAP:
            if k == "epoch" and any(x.startswith("eval_") for x in logs):
                continue
            out[_TRAIN_MAP[k]] = v
        else:
            out[f"misc/{k}"] = v
    if "train/loss" in out:
        out["train/ppl"] = math.exp(min(out["train/loss"], 20.0))
    seen, trained = out.get("data/tokens_seen"), out.get("data/trained_tokens_seen")
    if seen and trained is not None:
        out["data/loss_token_fraction"] = trained / seen
    return out


def dataset_stats(data_dir: str) -> dict:
    """Summary numbers from etl/build_sft.py (stats.json) to store in the run config."""
    import json
    p = Path(data_dir) / "stats.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text())
    tr = s.get("tokenized_train", {})
    return {"data": {"name": s.get("name"), "train_examples": tr.get("examples"), "train_tokens": tr.get("tokens_total"),
                     "train_trained_tokens": tr.get("tokens_trained"), "len_p95": tr.get("len_p95"),
                     "by_source": tr.get("by_source")}}


def git_info() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip()
        return (sha + ("+uncommitted" if dirty else "")) if sha else "unknown"
    except Exception:
        return "unknown"


# ---- thin wrappers (no-ops when W&B is off) -------------------------------------------------------------------------
def run():
    try:
        import wandb
        return wandb.run
    except Exception:
        return None


def log(step: int, payload: dict) -> None:
    r = run()
    if r is not None:
        r.log({"step": step, **payload})


def log_sample(step: int, prompt: str, completion: str, target: str) -> None:
    r = run()
    if r is not None:
        import wandb
        log(step, {"generations/final_merged": wandb.Table(columns=["prompt_tail", "completion", "target"],
                                                           data=[[prompt, completion, target]])})


def finish() -> None:
    r = run()
    if r is not None:
        r.finish()


class StructuredWandbCallback(TrainerCallback):
    """Starts the W&B run, defines the shared `step` axis and summaries, and logs every Trainer log line under the
    hierarchical names from structure_logs(). Must come BEFORE GenerationLogCallback in the callback list."""

    # (metric, how W&B summarises it in the runs table)
    SUMMARIES = [("train/loss", "min"), ("eval/loss", "min"), ("completion/tool/format_rate", "max"),
                 ("completion/tool/name_match", "max"), ("completion/tool/args_match", "max")]

    def __init__(self, cfg, extra: Optional[dict] = None):
        self.cfg, self.extra = cfg, extra or {}

    def on_train_begin(self, args, state, control, **kw):
        if not state.is_world_process_zero:
            return
        import wandb
        if wandb.run is None:
            wandb.init(project=os.environ.get("WANDB_PROJECT", "post-train"), name=self.cfg.run_name, job_type="sft",
                       tags=[self.cfg.method], config={"sft": dataclasses.asdict(self.cfg), "model": self.cfg.model_name,
                                                       "code": {"git": git_info()}, **self.extra})
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")
        for name, mode in self.SUMMARIES:
            wandb.define_metric(name, step_metric="step", summary=mode)

    def on_log(self, args, state, control, logs=None, **kw):
        if state.is_world_process_zero and logs:
            payload = structure_logs(logs)
            if payload:
                log(state.global_step, payload)
