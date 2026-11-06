"""Loading the tokenized SFT data produced by etl/build_sft.py, and batch collation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk


def load_tokenized(data_dir: str) -> tuple[Dataset, Dataset]:
    root = Path(data_dir) / "tokenized"
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Build it first:  python -m etl.build_sft --config etl/configs/sft_mix.yaml")
    return load_from_disk(str(root / "train")), load_from_disk(str(root / "val"))


@dataclass
class PadCollator:
    """Right-pads input_ids / labels / attention_mask to the longest sequence in the batch (multiple of 8).

    No sequence packing on purpose: Qwen3.5 has linear-attention (Gated DeltaNet) layers whose recurrent
    state would leak across packed sequences unless special cu_seqlens plumbing is used.
    """
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        longest = max(len(x["input_ids"]) for x in batch)
        m = self.pad_to_multiple_of
        longest = (longest + m - 1) // m * m
        ids = torch.full((len(batch), longest), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"])
            labels[i, :n] = torch.tensor(x["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def split_by_source(ds: Dataset) -> dict[str, Dataset]:
    """Group a tokenized dataset by the top-level source name (hermes, smoltalk, ...) for per-source eval metrics."""
    src = [s.split("/")[0] for s in ds["source"]]
    groups = {}
    for name in sorted(set(src)):
        groups[name] = ds.select([i for i, s in enumerate(src) if s == name])
    return groups or {"val": ds}
