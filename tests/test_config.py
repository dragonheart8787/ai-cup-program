import pytest

from src.utils.config import config_hash, load_config


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\npaths:\n  data_root: data\n", encoding="utf-8")
    assert load_config(p) == {"seed": 1, "paths": {"data_root": "data"}}


def test_load_empty_and_non_mapping(tmp_path):
    empty = tmp_path / "e.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_config(empty) == {}
    bad = tmp_path / "b.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_hash_ignores_key_order_and_detects_change():
    a = {"x": 1, "y": {"a": 1, "b": 2}}
    b = {"y": {"b": 2, "a": 1}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({"x": 2, "y": {"a": 1, "b": 2}})
    assert len(config_hash(a)) == 12


def test_default_config_loads():
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert cfg["seed"] == 42
