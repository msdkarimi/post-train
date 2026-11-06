"""Local JSONL files that are already in canonical format (see etl/schema.py).

Use this for your OWN data, e.g. teacher-distilled sandbox trajectories that passed their checker.
Each line: {"id": optional, "messages": [...], "tools": optional}

params:
  paths: list of .jsonl paths (relative to the repo root or absolute)
  source_name: label stored in the record (default: file stem)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from etl.schema import SkipExample, make_record
from etl.sources import register


@register("local_jsonl")
def local_jsonl(params: dict, seed: int) -> Iterator:
    for p in params["paths"]:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"local_jsonl path not found: {path}")
        label = params.get("source_name", path.stem)
        with path.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    yield SkipExample("jsonl_bad_json")
                    continue
                yield make_record(str(row.get("id", f"{label}/{i}")), label, row.get("messages", []), row.get("tools"))
