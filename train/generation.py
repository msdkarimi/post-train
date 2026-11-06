"""Periodic generation logging: see what the model actually writes while it trains.

Every `gen_log_steps` optimizer steps (and once BEFORE training = baseline, and once at the end) the callback
greedy-decodes a FIXED set of validation prompts, so rows are comparable across steps, and logs:

  W&B tables      generations/latest   this round only
                  generations/history  every round so far (filter by `prompt_id`, slide `step`)
  W&B scalars     completion/length_mean, completion/stopped_rate           (all prompts)
                  completion/tool/format_rate, name_match, args_match       (prompts whose target is a tool call)
                  completion/chat/length_mean, completion/chat/stopped_rate (plain chat prompts)
                  completion/gen_seconds                                    (what the logging costs you)
  local file      <output_dir>/generations.jsonl   (works without W&B)

Tool-call metrics parse the model's output with the Qwen3.5 call format and compare against the target:
  format_rate  every <tool_call> block is well formed          name_match  same function names as the target
  args_match   same function names AND same argument values
Generation is sequential (one prompt at a time): left-padded batching is not guaranteed correct with the
Gated-DeltaNet layers, and a few prompts keep the cost small. Single-process (rank 0) only.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

from train import wandb_utils

_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


# ---- tool-call parsing / metrics ---------------------------------------------------------------------------------
def parse_tool_calls(text: str) -> tuple[list[tuple[str, dict]], int]:
    """Return ([(function_name, {param: value_str})...], number_of_<tool_call>_openers_seen)."""
    calls = [(name.strip(), {k.strip(): v.strip() for k, v in _PARAM_RE.findall(body)})
             for name, body in _CALL_RE.findall(text)]
    return calls, text.count("<tool_call>")


def tool_metrics(output: str, target: str) -> dict[str, bool]:
    out_calls, n_open = parse_tool_calls(output)
    tgt_calls, _ = parse_tool_calls(target)

    def key(c):
        return c[0], json.dumps(c[1], sort_keys=True)

    return {
        "format_ok": n_open > 0 and n_open == len(out_calls),
        "name_match": sorted(c[0] for c in out_calls) == sorted(c[0] for c in tgt_calls),
        "args_match": sorted(map(key, out_calls)) == sorted(map(key, tgt_calls)),
    }


# ---- prompt selection --------------------------------------------------------------------------------------------
def select_prompts(val_ds, tok, per_source: int) -> list[dict]:
    """First `per_source` validation examples of each data source (the split is hash-deterministic, so the set is
    stable across runs). Prompt = everything before the first trained token; target = the first trained span."""
    picked: dict[str, list[int]] = {}
    for i, src in enumerate(val_ds["source"]):
        name = src.split("/")[0]
        if len(picked.setdefault(name, [])) < per_source:
            picked[name].append(i)
    prompts = []
    for name, idxs in picked.items():
        for i in idxs:
            ex = val_ds[i]
            labels = ex["labels"]
            first = next((j for j, l in enumerate(labels) if l != -100), None)
            if first is None:
                continue
            end = first
            while end < len(labels) and labels[end] != -100:
                end += 1
            target = tok.decode(labels[first:end])
            prompts.append({"prompt_id": ex["id"], "source": name, "prompt_ids": ex["input_ids"][:first],
                            "target": target, "kind": "tool" if "<tool_call>" in target else "chat"})
    return prompts


# ---- the callback ------------------------------------------------------------------------------------------------
class GenerationLogCallback(TrainerCallback):
    COLUMNS = ["step", "prompt_id", "source", "kind", "prompt_tail", "completion", "target",
               "stopped", "n_tokens", "tool_format_ok", "tool_name_match", "tool_args_match"]

    def __init__(self, tok, val_ds, cfg, output_dir):
        self.tok, self.cfg = tok, cfg
        self.every = int(cfg.gen_log_steps)
        self.max_new = int(cfg.gen_max_new_tokens)
        self.prompts = select_prompts(val_ds, tok, int(cfg.gen_prompts_per_source))
        self.eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
        self.out_path = Path(output_dir) / "generations.jsonl"
        self.rows: list[list] = []
        self._last_step = -1
        print(f"[generation logging] {len(self.prompts)} fixed prompts "
              f"({sum(p['kind'] == 'tool' for p in self.prompts)} tool-call, "
              f"{sum(p['kind'] == 'chat' for p in self.prompts)} chat), every {self.every} steps")

    # -- hooks
    def on_train_begin(self, args, state, control, model=None, **kw):
        if state.is_world_process_zero and model is not None:
            self._round(model, step=state.global_step)          # baseline before any update

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.is_world_process_zero and model is not None and state.global_step % self.every == 0:
            self._round(model, step=state.global_step)

    def on_train_end(self, args, state, control, model=None, **kw):
        if state.is_world_process_zero and model is not None and state.global_step != self._last_step:
            self._round(model, step=state.global_step)

    # -- one logging round
    @torch.no_grad()
    def _round(self, model, step: int) -> None:
        if not self.prompts:
            return
        t0 = time.time()
        net = getattr(model, "module", model)                    # unwrap DDP
        was_training = net.training
        net.eval()
        device = next(net.parameters()).device
        use_amp = bool(self.cfg.bf16) and device.type == "cuda"
        latest, recs = [], []
        try:
            for p in self.prompts:
                ids = torch.tensor([p["prompt_ids"]], device=device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = net.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.max_new,
                                       do_sample=False, use_cache=True, eos_token_id=self.eos_ids,
                                       pad_token_id=self.tok.pad_token_id)
                gen = out[0, ids.shape[1]:]
                text = self.tok.decode(gen)                      # keep special tokens so the stop token is visible
                stopped = bool(len(gen)) and int(gen[-1]) in self.eos_ids
                m = tool_metrics(text, p["target"]) if p["kind"] == "tool" else {}
                rec = {"step": step, "prompt_id": p["prompt_id"], "source": p["source"], "kind": p["kind"],
                       "prompt_tail": self.tok.decode(p["prompt_ids"][-400:]), "completion": text,
                       "target": p["target"], "stopped": stopped, "n_tokens": int(len(gen)),
                       "tool_format_ok": m.get("format_ok"), "tool_name_match": m.get("name_match"),
                       "tool_args_match": m.get("args_match")}
                recs.append(rec)
                latest.append([rec[c] for c in self.COLUMNS])
        finally:
            net.train(was_training)
        self.rows.extend(latest)
        self._last_step = step
        metrics = self._aggregate(recs)
        metrics["completion/gen_seconds"] = round(time.time() - t0, 1)

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = {k.removeprefix("completion/"): (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}
        print(f"\n[generation @ step {step}] {summary}")
        self._log_wandb(step, metrics, latest)

    @staticmethod
    def _aggregate(recs: list[dict]) -> dict:
        def mean(xs):
            return sum(xs) / len(xs) if xs else None

        out = {"completion/n_prompts": len(recs),
               "completion/length_mean": mean([r["n_tokens"] for r in recs]),
               "completion/stopped_rate": mean([float(r["stopped"]) for r in recs])}
        tool = [r for r in recs if r["kind"] == "tool"]
        chat = [r for r in recs if r["kind"] == "chat"]
        if tool:
            out["completion/tool/format_rate"] = mean([float(r["tool_format_ok"]) for r in tool])
            out["completion/tool/name_match"] = mean([float(r["tool_name_match"]) for r in tool])
            out["completion/tool/args_match"] = mean([float(r["tool_args_match"]) for r in tool])
        if chat:
            out["completion/chat/length_mean"] = mean([r["n_tokens"] for r in chat])
            out["completion/chat/stopped_rate"] = mean([float(r["stopped"]) for r in chat])
        return {k: v for k, v in out.items() if v is not None}

    def _log_wandb(self, step: int, metrics: dict, latest: list[list]) -> None:
        if wandb_utils.run() is None:
            return
        import wandb
        wandb_utils.log(step, {**metrics,
                               "generations/latest": wandb.Table(columns=self.COLUMNS, data=latest),
                               "generations/history": wandb.Table(columns=self.COLUMNS, data=self.rows)})
