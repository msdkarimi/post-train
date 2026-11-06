"""ETL unit tests. The tokenization tests need the Qwen3.5 tokenizer files (small download, cached by HF);
they are skipped automatically when the Hub is unreachable.

    pytest -q tests/test_etl.py
"""
from collections import Counter

import pytest

from etl.schema import SkipExample, content_hash, validate
from etl.sources.hermes import _convert, _parse_call
from etl.tokenization import encode_record
from etl.transforms import dedup, split_train_val


def rec(messages, tools=None, rid="t/1"):
    return {"id": rid, "source": "test", "messages": messages, "tools": tools}


CALL = {"type": "function", "function": {"name": "get_x", "arguments": {"a": 1}}}
TOOLS = [{"type": "function", "function": {"name": "get_x", "description": "d",
                                           "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}}}}]


# ---- schema -----------------------------------------------------------------------------------
def test_validate_accepts_tool_trajectory():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [CALL]},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": "done"}]
    assert validate(rec(msgs, TOOLS)) is None


@pytest.mark.parametrize("msgs,reason", [
    ([], "empty_messages"),
    ([{"role": "user", "content": "q"}], "ends_with_user"),
    ([{"role": "assistant", "content": "a"}], "bad_order_assistant"),
    ([{"role": "user", "content": "q"}, {"role": "tool", "content": "x"}, {"role": "assistant", "content": "a"}],
     "tool_without_call"),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}], "empty_assistant"),
    ([{"role": "user", "content": "q"}, {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a"}],
     "bad_order_user"),
    ([{"role": "user", "content": "q"},
      {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]}],
     "bad_tool_call_args"),
])
def test_validate_rejects(msgs, reason):
    assert validate(rec(msgs)) == reason


# ---- hermes adapter ---------------------------------------------------------------------------
def test_parse_call_variants():
    assert _parse_call('{"name": "f", "arguments": {"x": 1}}')["function"]["arguments"] == {"x": 1}
    # python-literal quoting and literal "\n" wrappers seen in the raw data
    assert _parse_call('\\n{"name": "f", "arguments": {"q": [\'a\', "b"]}}\\n')["function"]["arguments"] == {"q": ["a", "b"]}
    assert _parse_call('{"name": "f", "arguments": "{\\"x\\": 2}"}')["function"]["arguments"] == {"x": 2}
    with pytest.raises(SkipExample):
        _parse_call("not json at all")


def test_hermes_convert_splits_calls_and_responses():
    row = {"id": "r1", "tools": '[{"type": "function", "function": {"name": "f"}}]', "conversations": [
        {"from": "system", "value": "ignored hermes system prompt"},
        {"from": "human", "value": "do it"},
        {"from": "gpt", "value": '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>\n'
                                 '<tool_call>\n{"name": "g", "arguments": {}}\n</tool_call>'},
        {"from": "tool", "value": "<tool_response>\n{\"r\": 1}\n</tool_response>\n<tool_response>\n{\"r\": 2}\n</tool_response>"},
        {"from": "gpt", "value": "all done"}]}
    r = _convert(row, "cfg", 0, None)
    roles = [m["role"] for m in r["messages"]]
    assert roles == ["user", "assistant", "tool", "tool", "assistant"]      # system dropped, 2 responses split
    assert [c["function"]["name"] for c in r["messages"][1]["tool_calls"]] == ["f", "g"]
    assert r["messages"][1]["content"] == ""
    assert validate(r) is None


# ---- transforms -------------------------------------------------------------------------------
def test_dedup_and_split_are_deterministic():
    a = rec([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}], rid="a")
    b = rec([{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}], rid="b")
    dup = dict(a, id="a-dup")
    d = Counter()
    out = dedup([a, b, dup], d)
    assert [r["id"] for r in out] == ["a", "b"] and d["duplicate"] == 1
    many = [rec([{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "x"}], rid=str(i)) for i in range(400)]
    t1, v1 = split_train_val(many, 0.1, seed=7)
    t2, v2 = split_train_val(list(reversed(many)), 0.1, seed=7)
    assert {r["id"] for r in v1} == {r["id"] for r in v2}                      # order independent
    assert 10 < len(v1) < 80 and len(t1) + len(v1) == 400
    assert content_hash(a) == content_hash(dict(a, id="zzz"))


# ---- tokenization / loss masks (need the Qwen3.5 tokenizer) -------------------------------------
@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
    except Exception as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")


def trained_text(tok, ex):
    return tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])


def test_mask_trains_only_assistant_text_and_im_end(tok):
    r = rec([{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "It is 4."}])
    [ex] = encode_record(tok, r, 512, True, 4, Counter())
    assert trained_text(tok, ex) == "It is 4.<|im_end|>"        # no think prefix, no role tags, no user text
    assert len(ex["input_ids"]) == len(ex["labels"])


def test_mask_tool_trajectory(tok):
    msgs = [{"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [CALL]},
            {"role": "tool", "content": '{"temp": 3}'},
            {"role": "assistant", "content": "It is 3 degrees."}]
    [ex] = encode_record(tok, rec(msgs, TOOLS), 2048, True, 4, Counter())
    t = trained_text(tok, ex)
    assert "<function=get_x>" in t and "It is 3 degrees.<|im_end|>" in t
    assert "tool_response" not in t and '"temp"' not in t        # tool output is context, not a target
    assert "<think>" not in t and "# Tools" not in t


def test_multi_turn_expands_one_example_per_user_query(tok):
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}]
    exs = encode_record(tok, rec(msgs), 512, True, 4, Counter())
    assert [trained_text(tok, e) for e in exs] == ["hello<|im_end|>", "goodbye<|im_end|>"]
    assert len(encode_record(tok, rec(msgs), 512, False, 4, Counter())) == 1


def test_too_long_is_dropped_not_truncated(tok):
    msgs = [{"role": "user", "content": "word " * 600}, {"role": "assistant", "content": "ok"}]
    drops = Counter()
    assert encode_record(tok, rec(msgs), 128, True, 4, drops) == [] and drops["too_long"] == 1
