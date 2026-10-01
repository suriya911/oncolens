"""Phase 1 gate: verify GPU, storage, and that login credential files exist.

Credential files are checked for existence only. Their contents are never read.
Usage: python scripts/check_env.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MIN_FREE_GB = 150
EXPECTED_DRIVE = "/mnt/g"

CREDENTIAL_FILES = {
    "Kaggle": [Path.home() / ".kaggle" / "kaggle.json"],
    "Hugging Face": [Path.home() / ".cache" / "huggingface" / "token"],
    "Weights & Biases": [Path.home() / ".netrc", Path.home() / ".config" / "wandb" / "settings"],
    "GitHub CLI": [Path.home() / ".config" / "gh" / "hosts.yml"],
}

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        check("PyTorch installed", False, "pip install torch (see pytorch.org)")
        return
    check("PyTorch installed", True, torch.__version__)
    cuda = torch.cuda.is_available()
    check("CUDA available", cuda, torch.cuda.get_device_name(0) if cuda else "no CUDA device")
    check("bf16 supported", cuda and torch.cuda.is_bf16_supported())


def check_storage() -> None:
    root_env = os.environ.get("ONCOLENS_ROOT")
    check("ONCOLENS_ROOT set", bool(root_env), root_env or "run setup/storage_setup.sh")
    if not root_env:
        return
    root = Path(root_env)
    check("ONCOLENS_ROOT exists", root.is_dir(), str(root))
    if not root.is_dir():
        return
    check("ONCOLENS_ROOT on /mnt/g", str(root.resolve()).startswith(EXPECTED_DRIVE + "/"))
    probe = root / "tmp" / ".check_env_probe"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        probe.unlink()
        check("ONCOLENS_ROOT writable", True)
    except OSError as exc:
        check("ONCOLENS_ROOT writable", False, str(exc))
    free_gb = shutil.disk_usage(root).free / 1e9
    check(f"Free space >= {MIN_FREE_GB} GB", free_gb >= MIN_FREE_GB, f"{free_gb:.0f} GB free")


def check_credentials() -> None:
    for service, candidates in CREDENTIAL_FILES.items():
        found = next((p for p in candidates if p.exists()), None)
        check(f"{service} credentials present", found is not None, "" if found else "log in")


def main() -> int:
    check_torch()
    check_storage()
    check_credentials()
    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name.ljust(width)}  {detail}")
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
