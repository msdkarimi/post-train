# post-train

Post-training `Qwen/Qwen3.5-2B-Base` into a tool-using assistant: SFT now, RL (GRPO) next.

## Layout

```
bootstrap.sh        set up a fresh GPU VM (torch cu128, transformers, peft, ...) + sanity checks; exit 0 = OK
train.sh            build data if missing, then run SFT
run.sh              bootstrap.sh, and only if it exits 0, train.sh
etl/                raw datasets -> canonical conversations -> tokenized data with assistant-only loss masks
  configs/          sft_mix.yaml (real mix), sft_debug.yaml (tiny, laptop-friendly)
  sources/          one adapter per dataset (hermes, smoltalk, local_jsonl); add yours with @register
  schema.py         canonical record format + validation
  transforms.py     clean, dedup, deterministic train/val split
  tokenization.py   chat-template rendering + loss masks   (read its docstring: Qwen3.5 template quirks)
  build_sft.py      CLI entrypoint
train/              SFT trainer
  configs/          sft_lora.yaml, sft_full.yaml, sft_debug.yaml
  sft.py            entrypoint;  config.py / data.py / callbacks.py
tests/              pytest suite + tiny random Qwen3.5-architecture model for pipeline checks
src/scripts/        standalone utilities (vllm_smoke_test.py)
data/  outputs/     generated, git-ignored
```

## Quick start

```bash
# 1) data  (downloads datasets + tokenizer, writes data/processed/sft_v1)
python -m etl.build_sft --config etl/configs/sft_mix.yaml

# 2) train  (LoRA by default; ~hours on one A100)
python -m train.sft --config train/configs/sft_lora.yaml
python -m train.sft --config train/configs/sft_lora.yaml --set learning_rate=2e-4 --set max_train_minutes=60
python -m train.sft --config train/configs/sft_full.yaml        # full fine-tune, needs ~45GB+

# or everything on a fresh VM:   bash run.sh
```

Outputs land in `outputs/<run>/`: `merged/` (bf16 weights for eval/RL), `adapter/`, `checkpoint-*` (resume with
`--set resume_from_checkpoint=auto`), `final_eval.json`, `run_config.json`.

## Design notes

- **Loss only on assistant tokens** (text, tool calls, `<|im_end|>`). The Qwen3.5 chat template renders earlier
  assistant turns differently from the last one, so multi-turn chats are split into one example per user query and
  the `<think>\n\n</think>\n\n` prefix is masked, exactly matching what the model sees at inference. Non-thinking mode only.
- **Tool calls are structured** in the data and rendered by the model's own template, so SFT text == inference text.
- **No sequence packing**: Qwen3.5's linear-attention layers would leak state across packed sequences.
- **Full fine-tuning** keeps fp32 master weights under bf16 autocast; **LoRA** keeps the base in bf16.
- `max_train_minutes` stops and saves when a paid-GPU time budget is used up.
- The saved model's `generation_config` stops on `<|im_end|>` (the base tokenizer's eos is `<|endoftext|>`).

## Logging (Weights & Biases)

`report_to: wandb` in the config; the key comes from `WANDB_API_KEY` in `.env` (without it, logging falls back to the
console). Project = `$WANDB_PROJECT` (default `post-train`), run name = `run_name`. Offline test: `WANDB_MODE=offline`.
Logging is done by `train/wandb_utils.py` (not the stock Hugging Face callback) so names form a hierarchy; W&B turns the
part before the first `/` into a collapsible section, and every point shares one `step` axis so curves line up.

| Section | Metrics |
|---|---|
| `train/` | `loss`, `ppl`, `epoch` |
| `optim/` | `learning_rate`, `grad_norm` |
| `eval/` | `loss`, `ppl` (overall, token-weighted); `by_source/<source>/loss` and `/ppl` (hermes = tool use, smoltalk = chat, ...) |
| `completion/` | `length_mean`, `stopped_rate`, `tool/{format_rate,name_match,args_match}`, `chat/{length_mean,stopped_rate}`, `gen_seconds` |
| `perf/` | `tokens_per_sec`, `gpu_mem_alloc_gb`, `gpu_mem_reserved_gb`, `train_runtime_s` |
| `data/` | `tokens_seen`, `trained_tokens_seen`, `loss_token_fraction` |
| `generations/` | tables: `latest` (this round), `history` (all rounds; filter by `step` / `prompt_id`), `final_merged` |
| `system/` | GPU / CPU / disk (collected by W&B itself) |
| run config | full `sft:` config, model, dataset stats (`data.by_source`, ...), `code.git` commit (+`uncommitted` if dirty) |

The runs table also shows best-so-far summaries: min `train/loss` and `eval/loss`, max `completion/tool/*` rates.

### Seeing what the model writes

Every `gen_log_steps` steps (plus once **before** training = baseline, and once at the end) the model greedy-decodes a
fixed set of validation prompts (`gen_prompts_per_source` per data source), so rows are comparable across steps.
Tool-call prompts are scored by parsing the output: `format_rate` (well-formed `<tool_call>` blocks), `name_match`
(same functions as the target), `args_match` (same functions and argument values). Rows are also appended to
`outputs/<run>/generations.jsonl`, so this works without W&B. Cost: roughly `gen_max_new_tokens` x prompts / decode
speed per round (see `completion/gen_seconds`); raise `gen_log_steps` or set it to `0` to turn it off.

## Tests

```bash
pytest -q tests          # ETL + config tests (tokenizer tests skip if the Hub is unreachable)
```

Pipeline smoke test on a laptop (tiny random model, a few minutes on CPU):

```bash
python -m etl.build_sft --config etl/configs/sft_debug.yaml
python -m tests.tiny_model --out data/tiny_model
python -m train.sft --config train/configs/sft_debug.yaml
```
