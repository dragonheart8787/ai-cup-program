import csv
import hashlib

import pytest

from src.utils.experiments import COLUMNS, append_experiment, file_sha256, get_git_commit


def test_columns_match_spec():
    assert COLUMNS == (
        "exp_id", "timestamp", "git_commit", "config_hash", "dataset_version", "fold", "seed",
        "model", "checkpoint_sha256", "input_modality", "resolution", "postprocess",
        "localization_score", "text_score", "combined_score", "notes",
    )


def test_append_writes_header_once(tmp_path):
    p = tmp_path / "sub" / "experiments.csv"
    r1 = append_experiment(p, {"seed": 1, "model": "m1", "notes": "a,b \"quoted\"\nline2"})
    append_experiment(p, {"seed": 2, "fold": 0})
    with p.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert list(rows[0].keys()) == list(COLUMNS)
    assert rows[0]["exp_id"] == r1["exp_id"] and rows[0]["exp_id"] != ""
    assert rows[0]["timestamp"] != ""
    assert rows[0]["notes"] == "a,b \"quoted\"\nline2"
    assert rows[1]["model"] == "" and rows[1]["fold"] == "0"
    assert rows[0]["exp_id"] != rows[1]["exp_id"]


def test_unknown_column_raises(tmp_path):
    with pytest.raises(ValueError):
        append_experiment(tmp_path / "e.csv", {"bogus": 1})
    assert not (tmp_path / "e.csv").exists()


def test_file_sha256(tmp_path):
    p = tmp_path / "w.bin"
    p.write_bytes(b"abc" * 1000)
    assert file_sha256(p) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_git_commit_unknown_outside_repo(tmp_path):
    assert get_git_commit(tmp_path) == "UNKNOWN"
