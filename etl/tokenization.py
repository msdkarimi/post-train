"""Render conversations with the model's own chat template and build assistant-only loss masks.

Why this is not a one-liner (Qwen3.5 template behaviour, verified against its tokenizer_config):
  * Only assistant messages AFTER the last user query are rendered with a  <think>\\n\\n</think>\\n\\n  prefix;
    earlier assistant turns are rendered without it. So a multi-turn chat is NOT prefix-stable.
  * At inference, add_generation_prompt=True appends  <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n  so the
    model never generates that think prefix itself.
To train on exactly what the model sees/produces at inference we therefore:
  1. build ONE example per user query (history before it, assistant segment after it), and
  2. put loss only on the assistant text / tool calls / <|im_end|> of that last segment, masking the
     think prefix, the user/tool turns and the template scaffolding.
This trains NON-thinking mode (the template default).
"""
from __future__ import annotations

from typing import Optional

from etl.schema import SkipExample

IM_END = "<|im_end|>"


def render(tok, messages: list[dict], tools: Optional[list[dict]], add_generation_prompt: bool) -> str:
    try:
        return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception as e:  # jinja raise_exception(...) etc.
        raise SkipExample(f"render_error:{type(e).__name__}")


def _encode_segment(tok, msgs: list[dict], q_idx: int, tools, max_len: int) -> dict:
    """msgs = history + user query at q_idx + the assistant/tool messages that answer it."""
    full = render(tok, msgs, tools, add_generation_prompt=False)

    spans: list[tuple[int, int]] = []
    for a in range(q_idx + 1, len(msgs)):
        if msgs[a]["role"] != "assistant":
            continue
        prefix = render(tok, msgs[:a], tools, add_generation_prompt=True)   # ends with the think prefix
        upto = render(tok, msgs[: a + 1], tools, add_generation_prompt=False)
        if not (upto.startswith(prefix) and full.startswith(upto)):
            raise SkipExample("template_not_prefix_stable")
        if not upto.endswith(IM_END + "\n"):
            raise SkipExample("unexpected_assistant_end")
        spans.append((len(prefix), len(upto) - 1))   # include <|im_end|>, exclude the trailing newline
    if not spans:
        raise SkipExample("no_assistant_tokens")

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_len:
        raise SkipExample("too_long")

    labels = [-100] * len(ids)
    si = 0
    for t, (start, _end) in enumerate(offsets):
        while si < len(spans) and start >= spans[si][1]:
            si += 1
        if si < len(spans) and spans[si][0] <= start < spans[si][1]:
            labels[t] = ids[t]
    if all(l == -100 for l in labels):
        raise SkipExample("no_trainable_tokens")
    return {"input_ids": ids, "labels": labels}


def encode_record(tok, rec: dict, max_len: int, expand_turns: bool, max_expansions: int, drops) -> list[dict]:
    """Return 0..N tokenized examples for one canonical record; reasons for drops go into `drops` (Counter)."""
    msgs, tools = rec["messages"], rec.get("tools")
    user_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    if not user_idx:
        drops["no_user"] += 1
        return []
    targets = list(enumerate(user_idx))
    if not expand_turns:
        targets = targets[-1:]
    elif max_expansions and len(targets) > max_expansions:
        targets = targets[-max_expansions:]

    out = []
    for k, q in targets:
        end = user_idx[k + 1] if k + 1 < len(user_idx) else len(msgs)
        try:
            ex = _encode_segment(tok, msgs[:end], q, tools, max_len)
        except SkipExample as e:
            drops[str(e)] += 1
            continue
        ex["id"] = f"{rec['id']}#q{k}"
        ex["source"] = rec["source"]
        out.append(ex)
    return out


def show_mask(tok, input_ids: list[int], labels: list[int], open_mark: str = "⟦", close_mark: str = "⟧") -> str:
    """Human-readable view of what receives loss: trained spans are wrapped in ⟦ ⟧."""
    parts, buf, trained = [], [], False
    for i, l in zip(input_ids, labels):
        is_tr = l != -100
        if is_tr != trained and buf:
            txt = tok.decode(buf)
            parts.append(f"{open_mark}{txt}{close_mark}" if trained else txt)
            buf = []
        trained = is_tr
        buf.append(i)
    if buf:
        txt = tok.decode(buf)
        parts.append(f"{open_mark}{txt}{close_mark}" if trained else txt)
    return "".join(parts)
