"""Tool-call parser / metric tests. The round-trip test renders a call with the REAL Qwen3.5 chat template and checks
that the parser reads it back, so a template format change would be caught."""
import pytest

from train.generation import parse_tool_calls, tool_metrics

CALL = ("<tool_call>\n<function=get_x>\n<parameter=a>\n1\n</parameter>\n<parameter=city>\nParis\n</parameter>\n"
        "</function>\n</tool_call>")


def test_parse_single_and_parallel_calls():
    calls, n = parse_tool_calls(CALL + "\n" + CALL.replace("get_x", "get_y") + "<|im_end|>")
    assert n == 2 and [c[0] for c in calls] == ["get_x", "get_y"]
    assert calls[0][1] == {"a": "1", "city": "Paris"}


def test_parse_multiline_and_missing_args():
    calls, _ = parse_tool_calls("<tool_call>\n<function=f>\n<parameter=text>\nline1\nline2\n</parameter>\n</function>\n</tool_call>")
    assert calls == [("f", {"text": "line1\nline2"})]
    calls, _ = parse_tool_calls("<tool_call>\n<function=noargs>\n</function>\n</tool_call>")
    assert calls == [("noargs", {})]


def test_metrics_exact_partial_and_malformed():
    assert tool_metrics(CALL, CALL) == {"format_ok": True, "name_match": True, "args_match": True}
    wrong_arg = CALL.replace("Paris", "Rome")
    assert tool_metrics(wrong_arg, CALL) == {"format_ok": True, "name_match": True, "args_match": False}
    wrong_name = CALL.replace("get_x", "other")
    assert tool_metrics(wrong_name, CALL)["name_match"] is False
    truncated = CALL[: CALL.index("</function>")]                       # cut off mid-call (hit max_new_tokens)
    m = tool_metrics(truncated, CALL)
    assert m == {"format_ok": False, "name_match": False, "args_match": False}
    plain = tool_metrics("The weather is nice.<|im_end|>", CALL)         # answered in prose instead of calling
    assert plain == {"format_ok": False, "name_match": False, "args_match": False}


def test_roundtrip_with_real_chat_template():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 3, "opts": {"u": "c"}}}},
                {"type": "function", "function": {"name": "get_time", "arguments": {"tz": "CET"}}}]}]
    rendered = tok.apply_chat_template(msgs, tokenize=False)
    calls, n = parse_tool_calls(rendered)
    assert n == 2 and [c[0] for c in calls] == ["get_weather", "get_time"]
    assert calls[0][1]["city"] == "Paris" and calls[0][1]["days"] == "3" and calls[0][1]["opts"] == '{"u": "c"}'
    assert tool_metrics(rendered, rendered) == {"format_ok": True, "name_match": True, "args_match": True}
