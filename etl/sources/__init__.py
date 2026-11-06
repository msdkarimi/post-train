"""Source adapters: turn a raw dataset into canonical records (see etl/schema.py).

An adapter is a generator function  adapter(params: dict, seed: int) -> Iterator[dict | SkipExample-raising]
registered under a short name. It yields canonical records. To drop a single bad row, adapters
`raise SkipExample("reason")` from a per-row helper; `iter_source` catches and counts it.

Add a dataset = add one file here with @register("name") and reference it from etl/configs/*.yaml.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Iterator

from etl.schema import SkipExample

ADAPTERS: dict[str, Callable[..., Iterator]] = {}


def register(name: str):
    def deco(fn):
        ADAPTERS[name] = fn
        return fn
    return deco


def iter_source(adapter: str, params: dict, seed: int, drops: Counter) -> Iterator[dict]:
    """Run an adapter; adapters yield either records or SkipExample instances (counted, not raised)."""
    if adapter not in ADAPTERS:
        raise KeyError(f"unknown adapter {adapter!r}; known: {sorted(ADAPTERS)}")
    for item in ADAPTERS[adapter](params, seed):
        if isinstance(item, SkipExample):
            drops[str(item)] += 1
        else:
            yield item


# import for side effects (registration)
from etl.sources import hermes, local_jsonl, smoltalk  # noqa: E402,F401
