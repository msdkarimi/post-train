"""Config loader tests: YAML 1.1 reads `1e-5` as a string; the loader must coerce it to float."""
import pytest

from train.config import load_config


def test_scientific_notation_without_dot_is_float(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("learning_rate: 1e-5\nadam_epsilon: 1e-8\nnum_epochs: 2\nbf16: false\n")
    cfg = load_config(str(p), [])
    assert isinstance(cfg.learning_rate, float) and cfg.learning_rate == 1e-5
    assert isinstance(cfg.adam_epsilon, float) and cfg.num_epochs == 2.0 and cfg.bf16 is False


def test_overrides_are_typed():
    cfg = load_config(None, ["learning_rate=3e-3", "max_steps=12", "bf16=false", "run_name=abc", "report_to=none"])
    assert cfg.learning_rate == 3e-3 and cfg.max_steps == 12 and cfg.bf16 is False and cfg.run_name == "abc"


def test_unknown_keys_and_bad_method_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_config(None, ["nope=1"])
    with pytest.raises(ValueError):
        load_config(None, ["method=weird"])
    p = tmp_path / "c.yaml"
    p.write_text("learnng_rate: 1e-4\n")          # typo must not be silently ignored
    with pytest.raises(ValueError):
        load_config(str(p), [])
