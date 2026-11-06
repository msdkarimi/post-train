"""Cleaning, dedup and deterministic train/val split on canonical records."""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable

from etl.schema import content_hash, validate


def clean(rec: dict) -> dict:
    """Strip stray whitespace; drop unknown keys."""
    msgs = []
    for m in rec["messages"]:
        out = {"role": m["role"], "content": m["content"].strip() if isinstance(m.get("content"), str) else m.get("content")}
        if m.get("tool_calls"):
            out["tool_calls"] = m["tool_calls"]
        msgs.append(out)
    return {"id": rec["id"], "source": rec["source"], "messages": msgs, "tools": rec.get("tools")}


def clean_and_validate(records: Iterable[dict], drops: Counter) -> list[dict]:
    out = []
    for rec in records:
        rec = clean(rec)
        reason = validate(rec)
        if reason:
            drops[f"invalid:{reason}"] += 1
            continue
        out.append(rec)
    return out


def dedup(records: list[dict], drops: Counter) -> list[dict]:
    """Exact-duplicate removal on conversation content (keeps first occurrence)."""
    seen: set[str] = set()
    out = []
    for rec in records:
        h = content_hash(rec)
        if h in seen:
            drops["duplicate"] += 1
            continue
        seen.add(h)
        out.append(rec)
    return out


def split_train_val(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic split by content hash: the same conversation always lands in the same split,
    independent of ordering, so re-running the pipeline never leaks val examples into train."""
    train, val = [], []
    threshold = int(val_fraction * 10_000)
    for rec in records:
        bucket = int(hashlib.sha1(f"{seed}:{content_hash(rec)}".encode()).hexdigest(), 16) % 10_000
        (val if bucket < threshold else train).append(rec)
    return train, val
