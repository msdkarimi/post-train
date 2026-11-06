"""Supervised fine-tuning of Qwen3.5-2B-Base on the data built by `python -m etl.build_sft`.

    python -m train.sft --config train/configs/sft_lora.yaml
    python -m train.sft --config train/configs/sft_lora.yaml --set learning_rate=2e-4 --set max_train_minutes=60

Loss is computed only on assistant tokens (text, tool calls, <|im_end|>); the masks come from the ETL step.
Outputs under output_dir:
    checkpoint-*/        periodic Trainer checkpoints (resumable)
    adapter/             LoRA adapter            (method=lora)
    merged/              merged bf16 weights     (method=lora, merge_lora=true)  <- use this for eval / RL
    final/               bf16 weights            (method=full)
    run_config.json      the exact config used
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from etl.tokenization import show_mask
from train.callbacks import TimeBudgetCallback
from train.config import SFTRunConfig, load_config
from train.data import PadCollator, load_tokenized, split_by_source
from train.generation import GenerationLogCallback
from train.trainer import SFTTrainer
from train.wandb_utils import StructuredWandbCallback, dataset_stats, finish, log_sample


def build_training_args(cfg: SFTRunConfig, report_to: str) -> TrainingArguments:
    """Version-tolerant construction (transformers renamed several arguments between 4.x and 5.x)."""
    valid = {f.name for f in dataclasses.fields(TrainingArguments)}
    kw = dict(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        lr_scheduler_kwargs={"min_lr_rate": cfg.min_lr_ratio} if cfg.lr_scheduler == "cosine_with_min_lr" else {},
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim,
        bf16=cfg.bf16 and torch.cuda.is_available(),
        tf32=cfg.tf32 if torch.cuda.is_available() else None,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        report_to=report_to,
        seed=cfg.seed,
        dataloader_num_workers=cfg.num_workers,
        remove_unused_columns=False,        # our collator needs input_ids/labels as stored
        use_cpu=not torch.cuda.is_available(),
    )
    # renamed arguments
    kw["eval_strategy" if "eval_strategy" in valid else "evaluation_strategy"] = "steps"
    if "warmup_ratio" in valid:
        kw["warmup_ratio"] = cfg.warmup_ratio
    else:                                   # transformers >= 5: float in [0,1) = ratio of total steps
        kw["warmup_steps"] = cfg.warmup_ratio
    if cfg.group_by_length:
        if "train_sampling_strategy" in valid:
            kw["train_sampling_strategy"] = "group_by_length"
        else:
            kw["group_by_length"] = True
    return TrainingArguments(**{k: v for k, v in kw.items() if k in valid and v is not None})


def load_model(cfg: SFTRunConfig):
    # Full fine-tuning: keep fp32 master weights and train under bf16 autocast (pure-bf16 weights lose
    # small AdamW updates at lr ~1e-5). LoRA: frozen base in bf16 (adapters are kept in fp32 by peft).
    dtype = torch.float32 if cfg.method == "full" else torch.bfloat16
    if not torch.cuda.is_available():
        dtype = torch.float32
    kwargs = {"dtype": dtype}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    model.config.use_cache = False          # incompatible with gradient checkpointing
    if cfg.method == "lora":
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=cfg.lora_target, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
    else:
        n = sum(p.numel() for p in model.parameters())
        print(f"full fine-tuning: {n / 1e6:.0f}M trainable parameters")
    return model


def finalize_generation_config(model, tok):
    """The base tokenizer's eos is <|endoftext|>, but chat turns end with <|im_end|>. Without this the
    fine-tuned model would not stop after its answer when used with generate()/vLLM."""
    gc = model.generation_config
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    gc.eos_token_id = [im_end, tok.eos_token_id]
    gc.pad_token_id = tok.pad_token_id


def save_outputs(cfg: SFTRunConfig, model, tok):
    out = Path(cfg.output_dir)
    if cfg.method == "lora":
        model.save_pretrained(out / "adapter")
        if cfg.merge_lora:
            merged = model.merge_and_unload()
            merged = merged.to(torch.bfloat16)
            merged.config.use_cache = True
            finalize_generation_config(merged, tok)
            merged.save_pretrained(out / "merged")
            tok.save_pretrained(out / "merged")
            print(f"saved merged model -> {out / 'merged'}")
            return merged
    else:
        model = model.to(torch.bfloat16)
        model.config.use_cache = True
        finalize_generation_config(model, tok)
        model.save_pretrained(out / "final")
        tok.save_pretrained(out / "final")
        print(f"saved model -> {out / 'final'}")
    return model


@torch.no_grad()
def sample_from_val(model, tok, val_ds, max_new_tokens: int = 128):
    """Generate from the first val example's prompt (everything before its first trained token)."""
    ex = val_ds[0]
    first = next(i for i, l in enumerate(ex["labels"]) if l != -100)
    prompt_ids = torch.tensor([ex["input_ids"][:first]], device=model.device)
    model.eval()
    model.config.use_cache = True
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    out = model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                         max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=[im_end, tok.eos_token_id],
                         pad_token_id=tok.pad_token_id)
    prompt = tok.decode(prompt_ids[0][-300:])
    model_out = tok.decode(out[0][first:])
    target = tok.decode([t for t in ex["labels"] if t != -100])[:500]
    print("\n----- sample generation (greedy) -----")
    print("PROMPT (tail):", prompt)
    print("MODEL :", model_out)
    print("TARGET:", target)
    return prompt, model_out, target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    if cfg.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    report_to = cfg.report_to
    if report_to == "wandb" and not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set -> logging to console only")
        report_to = "none"
    os.environ.setdefault("WANDB_PROJECT", "post-train")

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds, val_ds = load_tokenized(cfg.data_dir)
    n_tr = sum(sum(1 for l in x if l != -100) for x in train_ds["labels"])
    n_all = sum(len(x) for x in train_ds["input_ids"])
    print(f"train: {len(train_ds)} examples, {n_all / 1e6:.2f}M tokens ({100 * n_tr / max(n_all, 1):.1f}% receive loss) | val: {len(val_ds)}")
    print("\n----- loss-mask preview of one training example (⟦trained⟧) -----")
    print(show_mask(tok, train_ds[0]["input_ids"], train_ds[0]["labels"])[-900:])

    model = load_model(cfg)
    targs = build_training_args(cfg, "none")      # W&B is handled by StructuredWandbCallback (hierarchical names)
    callbacks = [TimeBudgetCallback(cfg.max_train_minutes)] if cfg.max_train_minutes > 0 else []
    # evaluate per data source (eval_hermes_loss, eval_smoltalk_loss, ...); the trainer also logs the overall eval_loss
    eval_sets = split_by_source(val_ds)
    eval_weights = {n: float(sum(sum(1 for l in x if l != -100) for x in d["labels"])) for n, d in eval_sets.items()}
    if report_to == "wandb":
        callbacks.append(StructuredWandbCallback(cfg, extra=dataset_stats(cfg.data_dir)))   # before GenerationLogCallback
    if cfg.gen_log_steps > 0:
        callbacks.append(GenerationLogCallback(tok, val_ds, cfg, out))
    trainer_kwargs = dict(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_sets,
                          eval_weights=eval_weights,
                          data_collator=PadCollator(pad_token_id=tok.pad_token_id), callbacks=callbacks)
    # `tokenizer` was renamed `processing_class` in newer transformers
    sig = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in sig else "tokenizer"] = tok
    trainer = SFTTrainer(**trainer_kwargs)

    resume = cfg.resume_from_checkpoint or None
    if resume == "auto":
        resume = get_last_checkpoint(str(out))
        print("resuming from:", resume)

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    print(f"training finished in {(time.time() - t0) / 60:.1f} min")
    metrics = trainer.evaluate()
    print("final eval:", metrics)
    (out / "final_eval.json").write_text(json.dumps(metrics, indent=2))

    final_model = save_outputs(cfg, trainer.model, tok)
    if cfg.sample_after_train:
        try:
            prompt, model_out, target = sample_from_val(final_model, tok, val_ds)
            log_sample(trainer.state.global_step, prompt, model_out, target)
        except Exception as e:  # never lose a finished run to a cosmetic step
            print("sample generation skipped:", type(e).__name__, e)
    finish()


if __name__ == "__main__":
    main()
