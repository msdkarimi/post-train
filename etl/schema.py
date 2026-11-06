"""Canonical conversation schema shared by every stage after the source adapters.

record = {
    "id":       str,
    "source":   str,                       # which adapter/mix entry produced it
    "messages": [ {"role": "system"|"user"|"assistant"|"tool",
                   "content": str,
                   "tool_calls": [ {"type": "function",
                                    "function": {"name": str, "arguments": dict}} ]   # assistant only, optional
                 }, ... ],
    "tools":    [ {"type": "function", "function": {...json schema...}}, ... ] | None,
}

Tool calls are STRUCTURED here (not text). The model's own chat template renders them into
its native format (<tool_call><function=...>), so training text matches inference text.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

ROLES = {"system", "user", "assistant", "tool"}


class SkipExample(Exception):
    """Raise to drop one example. The message is the reason and is counted in stats.json."""


def make_record(rec_id: str, source: str, messages: list[dict], tools: Optional[list[dict]] = None) -> dict:
    return {"id": rec_id, "source": source, "messages": messages, "tools": tools or None}


def validate(rec: dict) -> Optional[str]:
    """Return None if the record is well formed, otherwise a short reason string."""
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return "empty_messages"

    tools = rec.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
            return "bad_tools"

    prev_role = None
    prev_had_calls = False
    seen_user = False
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = m.get("content")
        if role not in ROLES:
            return "bad_role"
        if not isinstance(content, str):
            return "non_string_content"
        calls = m.get("tool_calls")

        if role == "system":
            if i != 0:
                return "system_not_first"
            if not content.strip():
                return "empty_system"
        elif role == "user":
            if prev_role in ("user",) or (prev_role == "assistant" and prev_had_calls):
                return "bad_order_user"
            if not content.strip():
                return "empty_user"
            seen_user = True
        elif role == "assistant":
            if not seen_user or prev_role not in ("user", "tool"):
                return "bad_order_assistant"
            if not content.strip() and not calls:
                return "empty_assistant"
            if calls:
                if not isinstance(calls, list):
                    return "bad_tool_calls"
                for c in calls:
                    fn = c.get("function") if isinstance(c, dict) else None
                    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
                        return "bad_tool_call_name"
                    if not isinstance(fn.get("arguments"), dict):
                        return "bad_tool_call_args"
        elif role == "tool":
            if not (prev_role == "tool" or (prev_role == "assistant" and prev_had_calls)):
                return "tool_without_call"
            if calls:
                return "tool_with_calls"

        prev_role = role
        prev_had_calls = bool(calls) if role == "assistant" else False

    if not seen_user:
        return "no_user"
    if msgs[-1]["role"] not in ("assistant", "tool"):
        return "ends_with_user"
    return None


def content_hash(rec: dict) -> str:
    """Stable hash of the conversation content (ignores id/source) for dedup and splitting."""
    payload = json.dumps({"m": rec["messages"], "t": rec.get("tools")}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
