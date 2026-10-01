from pathlib import Path

import pytest

from oncolens.config import load_config
from oncolens.utils.storage import Paths

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = ("root", "raw", "shards", "checkpoints", "artifacts", "runs", "tmp")


@pytest.mark.parametrize("profile", ["local", "kaggle", "ci"])
def test_paths_profiles_compose(profile, monkeypatch):
    monkeypatch.delenv("ONCOLENS_ROOT", raising=False)
    cfg = load_config([f"paths={profile}"])
    paths = Paths.from_cfg(cfg.paths)
    for key in PATH_KEYS:
        p = Path(getattr(paths, key)).resolve()
        assert p.is_absolute()
        assert not p.is_relative_to(REPO_ROOT), f"{profile}.{key}={p} points inside the repo"


def test_local_default_root_avoids_case_collision(monkeypatch):
    # /mnt/g/oncolens is the repo folder on a case-insensitive Windows drive.
    monkeypatch.delenv("ONCOLENS_ROOT", raising=False)
    cfg = load_config(["paths=local"])
    assert cfg.paths.root == "/mnt/g/oncolens-data"
    assert cfg.paths.require_mount is True


def test_local_root_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ONCOLENS_ROOT", str(tmp_path))
    cfg = load_config(["paths=local"])
    assert cfg.paths.checkpoints == f"{tmp_path}/checkpoints"


def test_kaggle_paths():
    cfg = load_config(["paths=kaggle"])
    assert cfg.paths.raw == "/kaggle/input"
    assert cfg.paths.root == "/kaggle/working"
    assert cfg.paths.require_mount is False


def test_defaults_compose():
    cfg = load_config()
    assert cfg.model.init == "scratch"
    assert cfg.model.num_classes == cfg.data.num_classes
    assert cfg.trainer.strategy == "single"
    assert cfg.trainer.batch_size_per_gpu > 0
