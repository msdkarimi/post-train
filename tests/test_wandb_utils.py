"""Metric-name hierarchy tests (pure function; no W&B account or network needed)."""
import math

from train.wandb_utils import structure_logs


def test_train_log_is_split_into_sections():
    out = structure_logs({"loss": 1.5, "grad_norm": 0.3, "learning_rate": 1e-4, "epoch": 0.1,
                          "tokens_seen": 1000, "trained_tokens_seen": 250, "tokens_per_sec": 5000.0,
                          "gpu_mem_alloc_gb": 30.5, "gpu_mem_reserved_gb": 35.0})
    assert out["train/loss"] == 1.5 and out["train/epoch"] == 0.1
    assert math.isclose(out["train/ppl"], math.exp(1.5))
    assert out["optim/learning_rate"] == 1e-4 and out["optim/grad_norm"] == 0.3
    assert out["perf/tokens_per_sec"] == 5000.0 and out["perf/gpu_mem_alloc_gb"] == 30.5
    assert out["data/tokens_seen"] == 1000 and out["data/loss_token_fraction"] == 0.25
    assert not any(k.startswith("train/") and k.split("/")[1] in ("grad_norm", "tokens_per_sec") for k in out)


def test_eval_log_overall_and_by_source():
    out = structure_logs({"eval_loss": 1.2, "eval_ppl": 3.3, "eval_hermes_loss": 1.0, "eval_hermes_ppl": 2.7,
                          "eval_sandbox_trajectories_loss": 0.9,           # source names may contain underscores
                          "eval_runtime": 1.0, "eval_samples_per_second": 9.0, "epoch": 0.5})
    assert out == {"eval/loss": 1.2, "eval/ppl": 3.3, "eval/by_source/hermes/loss": 1.0,
                   "eval/by_source/hermes/ppl": 2.7, "eval/by_source/sandbox_trajectories/loss": 0.9}


def test_end_of_training_summary_and_unknown_keys():
    out = structure_logs({"train_runtime": 120.0, "train_loss": 0.8, "train_samples_per_second": 3.0,
                          "train_steps_per_second": 0.1, "total_flos": 1e15, "epoch": 1.0, "weird": 7})
    assert out["perf/train_runtime_s"] == 120.0 and out["train/loss_run_average"] == 0.8
    assert out["train/epoch"] == 1.0 and out["misc/weird"] == 7
    assert "train/samples_per_second" not in out and not any("flos" in k for k in out)


def test_non_numeric_values_are_ignored():
    assert structure_logs({"loss": 1.0, "note": "text", "flag": True}) == {"train/loss": 1.0, "train/ppl": math.exp(1.0)}
