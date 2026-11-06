"""Build the SFT dataset: sources -> canonical -> clean/validate -> dedup -> split -> tokenize+mask -> save.

    python -m etl.build_sft --config etl/configs/sft_mix.yaml
    python -m etl.build_sft --config etl/configs/sft_mix.yaml --limit 50 --out data/processed/debug   # quick check

Output (under --out, default data/processed/<config name>):
    canonical/{train,val}.jsonl      human-readable conversations after cleaning/dedup/split
    tokenized/{train,val}            HF datasets (input_ids, labels, id, source) read by train/sft.py
    stats.json                       counts per source, drop reasons, sequence-length stats
    config.yaml                      copy of the config used (reproducibility)
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path

import yaml

from etl.sources import iter_source
from etl.tokenization import encode_record, show_mask
from etl.transforms import clean_and_validate, dedup, split_train_val


def _pct(xs: list[int], p: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="default: data/processed/<config name>")
    ap.add_argument("--limit", type=int, default=None, help="cap examples PER SOURCE (debug runs)")
    ap.add_argument("--no-tokenize", action="store_true", help="stop after canonical jsonl (no tokenizer download)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    name = cfg.get("name", Path(args.config).stem)
    out_dir = Path(args.out or f"data/processed/{name}")
    seed = int(cfg.get("seed", 42))
    max_len = int(cfg.get("max_seq_len", 4096))
    val_fraction = float(cfg.get("val_fraction", 0.02))

    # 1) load sources -------------------------------------------------------------------------
    per_source: dict[str, dict] = {}
    all_records: list[dict] = []
    drops_total: Counter = Counter()
    for src in cfg["sources"]:
        drops: Counter = Counter()
        cap = args.limit if args.limit is not None else src.get("max_examples")
        recs = []
        for rec in iter_source(src["adapter"], src.get("params", {}), seed, drops):
            recs.append(rec)
            if cap is not None and len(recs) >= cap:
                break
        for r in recs:
            r["_expand"] = bool(src.get("expand_turns", True))
            r["_max_exp"] = int(src.get("max_expansions", 4))
        per_source[src["name"]] = {"loaded": len(recs), "adapter_drops": dict(drops)}
        drops_total.update(drops)
        all_records.extend(recs)
        print(f"[{src['name']}] loaded {len(recs)} (adapter drops: {dict(drops)})")

    # 2) clean / validate / dedup / split -----------------------------------------------------
    flags = {r["id"]: (r.pop("_expand"), r.pop("_max_exp")) for r in all_records}
    drops_cv: Counter = Counter()
    records = clean_and_validate(all_records, drops_cv)
    records = dedup(records, drops_cv)
    train, val = split_train_val(records, val_fraction, seed)
    print(f"after clean/dedup: {len(records)} -> train {len(train)} / val {len(val)}  (drops: {dict(drops_cv)})")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "canonical").mkdir(parents=True)
    for split, recs in (("train", train), ("val", val)):
        with (out_dir / "canonical" / f"{split}.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(args.config, out_dir / "config.yaml")

    stats = {"name": name, "sources": per_source, "clean_dedup_drops": dict(drops_cv),
             "n_canonical": {"train": len(train), "val": len(val)}}

    # 3) tokenize + loss masks ----------------------------------------------------------------
    if not args.no_tokenize:
        from datasets import Dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "Qwen/Qwen3.5-2B-Base"))
        for split, recs in (("train", train), ("val", val)):
            drops_tok: Counter = Counter()
            rows = []
            for r in recs:
                expand, max_exp = flags[r["id"]]
                rows.extend(encode_record(tok, r, max_len, expand, max_exp, drops_tok))
            lens = [len(x["input_ids"]) for x in rows]
            trained = [sum(1 for l in x["labels"] if l != -100) for x in rows]
            stats[f"tokenized_{split}"] = {
                "examples": len(rows), "drops": dict(drops_tok),
                "tokens_total": sum(lens), "tokens_trained": sum(trained),
                "len_mean": round(statistics.mean(lens), 1) if lens else 0,
                "len_p50": _pct(lens, 0.5), "len_p95": _pct(lens, 0.95), "len_max": max(lens, default=0),
                "by_source": dict(Counter(x["source"].split("/")[0] for x in rows)),
            }
            print(f"[{split}] tokenized {len(rows)} examples; drops {dict(drops_tok)}")
            ds = Dataset.from_list(rows) if rows else Dataset.from_dict(
                {"input_ids": [], "labels": [], "id": [], "source": []})
            ds.save_to_disk(str(out_dir / "tokenized" / split))
            if split == "train" and rows:
                for ex in (rows[0], rows[len(rows) // 2]):
                    print("\n----- loss-mask preview (⟦trained⟧) -----")
                    print(show_mask(tok, ex["input_ids"], ex["labels"])[:1800])

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nDone. Output in {out_dir}  (stats.json has the breakdown)")


if __name__ == "__main__":
    main()
