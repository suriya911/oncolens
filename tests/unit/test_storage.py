from pathlib import Path

import pytest

from oncolens.utils.storage import Paths, StorageError, mount_for, require_storage


def make_cfg(root: Path, require_mount: bool = True) -> dict:
    return {
        "root": root,
        "raw": root / "raw",
        "shards": root / "shards",
        "checkpoints": root / "checkpoints",
        "artifacts": root / "artifacts",
        "runs": root / "runs",
        "tmp": root / "tmp",
        "data_format": "webdataset",
        "require_mount": require_mount,
    }


def test_passes_on_windows_mount(tmp_path, mounts_file):
    root = tmp_path / "oncolens-data"
    root.mkdir()
    paths = require_storage(make_cfg(root), mounts_file=mounts_file("9p"))
    assert isinstance(paths, Paths)
    assert paths.root == root
    assert list((root / "tmp").iterdir()) == []  # probe file cleaned up


def test_missing_root_raises_with_mount_hint(tmp_path, mounts_file):
    with pytest.raises(StorageError, match="sudo mount -t drvfs G: /mnt/g"):
        require_storage(make_cfg(tmp_path / "absent"), mounts_file=mounts_file())


def test_root_not_on_windows_drive_raises(tmp_path, mounts_file):
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(StorageError, match="not on a mounted Windows drive"):
        require_storage(make_cfg(root), mounts_file=mounts_file("ext4"))


def test_unwritable_root_raises(tmp_path, mounts_file):
    root = tmp_path / "data"
    root.mkdir()
    (root / "tmp").write_text("a file where the tmp dir should be")
    with pytest.raises(StorageError, match="not writable"):
        require_storage(make_cfg(root), mounts_file=mounts_file())


def test_skip_when_mount_not_required(tmp_path):
    paths = require_storage(make_cfg(tmp_path / "absent", require_mount=False))
    assert paths.require_mount is False


def test_mount_for_picks_longest_prefix(tmp_path):
    f = tmp_path / "mounts"
    f.write_text("/dev/sdc / ext4 rw 0 0\nG:\\134 /mnt/g 9p rw 0 0\n")
    assert mount_for(Path("/mnt/g/oncolens-data/raw"), f) == ("/mnt/g", "9p")
    assert mount_for(Path("/mnt/gx/data"), f) == ("/", "ext4")
    assert mount_for(Path("/home/user"), f) == ("/", "ext4")
