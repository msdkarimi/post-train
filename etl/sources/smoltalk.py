"""HuggingFaceTB/smoltalk  ->  canonical records (general chat / instruction following).

Mixed in so the tool-use SFT does not erase general chat and instruction-following ability.
Rows are {"messages": [{"role","content"}...], "source": str} with roles system/user/assistant.

params:
  subsets:   mapping  subset_name -> max_examples, e.g. {smol-magpie-ultra: 3000, smol-constraints: 2000}
             (subset names are HF configs: smol-magpie-ultra, smol-constraints, smol-rewrite, smol-summarize,
              everyday-conversations, systemchats-30k, ...)
  shuffle_buffer: streaming shuffle buffer size (default 10000)
"""
from __future__ import annotations

from typing import Iterator

from etl.schema import SkipExample, make_record
from etl.sources import register


@register("smoltalk")
def smoltalk(params: dict, seed: int) -> Iterator:
    from datasets import load_dataset

    subsets: dict = params["subsets"]
    buf = int(params.get("shuffle_buffer", 10_000))
    for subset, cap in subsets.items():
        ds = load_dataset("HuggingFaceTB/smoltalk", subset, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=buf)
        n = 0
        for idx, row in enumerate(ds):
            if cap is not None and n >= cap:
                break
            msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
            if any(m["role"] not in ("system", "user", "assistant") for m in msgs):
                yield SkipExample("smoltalk_unexpected_role")
                continue
            n += 1
            yield make_record(f"smoltalk/{subset}/{idx}", f"smoltalk/{subset}", msgs, None)
