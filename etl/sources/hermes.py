"""NousResearch/hermes-function-calling-v1  ->  canonical records.

Raw format (ShareGPT-like): conversations = [{"from": system|human|gpt|tool, "value": str}], tools = JSON string.
  - gpt turns embed calls as text:   <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>   (one or more)
  - tool turns embed results as:     <tool_response>\n{...}\n</tool_response>                        (one or more)
We parse those into STRUCTURED tool_calls / separate `tool` messages, and drop the dataset's own
system prompt (it describes a JSON tool-call format that conflicts with the Qwen3.5 template, which
renders tools + the call format itself).

params:
  configs:       list of HF configs, default [func_calling_singleturn, func_calling, glaive_func_calling]
  max_examples:  optional cap per config
  system_prompt: optional string to use as the system message (default: none)
"""
from __future__ import annotations

import ast
import json
import re
from typing import Iterator

from etl.schema import SkipExample, make_record
from etl.sources import register

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_TOOL_RESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)
_LITERAL_NL_EDGES = re.compile(r"^(?:\\n|\s)+|(?:\\n|\s)+$")  # some rows wrap the JSON in literal "\n" text

DEFAULT_CONFIGS = ["func_calling_singleturn", "func_calling", "glaive_func_calling"]


def _parse_call(raw: str) -> dict:
    raw = _LITERAL_NL_EDGES.sub("", raw)
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            raise SkipExample("tool_call_unparseable")
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        raise SkipExample("tool_call_bad_shape")
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            raise SkipExample("tool_call_args_not_json")
    if not isinstance(args, dict):
        raise SkipExample("tool_call_args_not_dict")
    return {"type": "function", "function": {"name": obj["name"], "arguments": args}}


def _convert(row: dict, cfg_name: str, idx: int, system_prompt):
    try:
        tools = json.loads(row["tools"]) if row.get("tools") else None
    except Exception:
        raise SkipExample("tools_unparseable")

    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for turn in row["conversations"]:
        role, value = turn["from"], turn["value"]
        if role == "system":
            continue
        if role == "human":
            msgs.append({"role": "user", "content": value.strip()})
        elif role == "gpt":
            calls = [_parse_call(b) for b in _TOOL_CALL_RE.findall(value)]
            text = _TOOL_CALL_RE.sub("", value).strip()
            m = {"role": "assistant", "content": text}
            if calls:
                m["tool_calls"] = calls
            msgs.append(m)
        elif role == "tool":
            for r in (_TOOL_RESP_RE.findall(value) or [value.strip()]):
                msgs.append({"role": "tool", "content": r})
        else:
            raise SkipExample("unknown_from_role")
    rid = f"hermes/{cfg_name}/{row.get('id', idx)}"
    return make_record(rid, "hermes", msgs, tools)


@register("hermes")
def hermes(params: dict, seed: int) -> Iterator:
    from datasets import load_dataset

    cap = params.get("max_examples")
    for cfg_name in params.get("configs", DEFAULT_CONFIGS):
        ds = load_dataset("NousResearch/hermes-function-calling-v1", cfg_name, split="train")
        for idx, row in enumerate(ds):
            if cap is not None and idx >= cap:
                break
            try:
                yield _convert(row, cfg_name, idx, params.get("system_prompt"))
            except SkipExample as e:
                yield e
