"""Storage guard and path helpers.

Every script that touches data or checkpoints calls ``require_storage()`` first, so a missing or
sleeping external drive fails fast with a clear message instead of corrupting a run.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MIN_FREE_GB_WARN = 20
WINDOWS_FS_TYPES = {"9p", "drvfs"}
MOUNT_HINT = (
    "My Passport (G:) not mounted at /mnt/g. "
    "Plug it in, then run: sudo mkdir -p /mnt/g && sudo mount -t drvfs G: /mnt/g"
)


class StorageError(RuntimeError):
    """Raised when the data root is missing, not a Windows drive mount, or not writable."""


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    shards: Path
    checkpoints: Path
    artifacts: Path
    runs: Path
    tmp: Path
    data_format: str
    require_mount: bool

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> Paths:
        dirs = ("root", "raw", "shards", "checkpoints", "artifacts", "runs", "tmp")
        values: dict[str, Any] = {k: Path(str(cfg[k])) for k in dirs}
        values["data_format"] = str(cfg["data_format"])
        values["require_mount"] = bool(cfg["require_mount"])
        return cls(**values)


def mount_for(path: Path, mounts_file: Path = Path("/proc/mounts")) -> tuple[str, str] | None:
    """Return (mount point, fs type) of the longest mount point that contains ``path``."""
    best: tuple[str, str] | None = None
    target = str(path)
    for line in mounts_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point, fs_type = parts[1].replace("\\040", " "), parts[2]
        inside = target == mount_point or target.startswith(mount_point.rstrip("/") + "/")
        if inside and (best is None or len(mount_point) > len(best[0])):
            best = (mount_point, fs_type)
    return best


def require_storage(
    paths_cfg: Mapping[str, Any] | Paths,
    mounts_file: Path = Path("/proc/mounts"),
) -> Paths:
    """Fail fast with a clear message when the external drive is missing or read-only."""
    paths = paths_cfg if isinstance(paths_cfg, Paths) else Paths.from_cfg(paths_cfg)
    if not paths.require_mount:
        return paths

    root = paths.root
    if not root.is_dir():
        raise StorageError(f"Data root {root} does not exist. {MOUNT_HINT}")

    mount = mount_for(root.resolve(), mounts_file)
    if mount is None or mount[1] not in WINDOWS_FS_TYPES:
        raise StorageError(f"Data root {root} is not on a mounted Windows drive. {MOUNT_HINT}")

    probe = paths.tmp / f".probe-{uuid.uuid4().hex}"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        raise StorageError(f"Data root {root} is not writable ({exc}). {MOUNT_HINT}") from exc

    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < MIN_FREE_GB_WARN:
        logger.warning("Only %.1f GB free on %s", free_gb, root)
    return paths
