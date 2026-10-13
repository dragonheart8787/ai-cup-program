import pytest

from src.dataset.paths import DatasetPaths


def make_tree(tmp_path):
    for rel in ("a/a/train", "b/b/test"):
        (tmp_path / rel).mkdir(parents=True)
    (tmp_path / "a/a/make_submission.py").write_text("")
    (tmp_path / "a/a/sample_submission.json").write_text("{}")


CFG = {"dataset": {"data_root": None, "train_root": "a/a/train", "test_root": "b/b/test",
                   "make_submission_py": "a/a/make_submission.py", "sample_submission_json": "a/a/sample_submission.json"}}


def test_env_overrides_and_resolves(tmp_path):
    make_tree(tmp_path)
    p = DatasetPaths.from_config(CFG, env={"AICUP_DATA_ROOT": str(tmp_path)})
    assert p.train_root == tmp_path / "a/a/train" and p.image_dir("test", "IR") == tmp_path / "b/b/test/IR/image"


def test_missing_root_raises(tmp_path):
    with pytest.raises(ValueError, match="AICUP_DATA_ROOT"):
        DatasetPaths.from_config(CFG, env={})
    with pytest.raises(FileNotFoundError):
        DatasetPaths.from_config(CFG, env={"AICUP_DATA_ROOT": str(tmp_path / "nope")})


def test_missing_subdir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DatasetPaths.from_config(CFG, env={"AICUP_DATA_ROOT": str(tmp_path)})


def test_default_config_has_no_machine_path():
    from pathlib import Path

    from src.utils.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert cfg["dataset"]["data_root"] is None
    assert not any(":" in str(v) for k, v in cfg["dataset"].items() if k != "data_root" and v)
