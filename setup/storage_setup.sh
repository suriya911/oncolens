#!/usr/bin/env bash
# Usage: bash setup/storage_setup.sh /mnt/g/oncolens-data
# Note: the default is NOT /mnt/g/oncolens, because Windows paths are case-insensitive and
# /mnt/g/oncolens would be the repo folder /mnt/g/OncoLens.
set -euo pipefail
ROOT="${1:-/mnt/g/oncolens-data}"
DRIVE="$(dirname "$ROOT")"

if ! grep -qs " $DRIVE " /proc/mounts; then
  echo "ERROR: $DRIVE is not mounted. Plug in My Passport (G:), then run:"
  echo "  sudo mkdir -p $DRIVE && sudo mount -t drvfs G: $DRIVE"
  exit 1
fi

mkdir -p "$ROOT"/{raw,shards,checkpoints,artifacts,runs/hydra,runs/wandb,runs/kaggle,cache/hf,cache/torch,cache/pip,dvc-remote,tmp}

echo "Write-speed test (1 GB) on $ROOT ..."
dd if=/dev/zero of="$ROOT/tmp/speedtest.bin" bs=1M count=1024 conv=fdatasync 2>&1 | tail -1
rm -f "$ROOT/tmp/speedtest.bin"

MARK_START="# >>> oncolens storage >>>"
MARK_END="# <<< oncolens storage <<<"
if ! grep -q "$MARK_START" ~/.bashrc; then
  cat >> ~/.bashrc <<EOT
$MARK_START
export ONCOLENS_ROOT="$ROOT"
export HF_HUB_CACHE="\$ONCOLENS_ROOT/cache/hf"
export TORCH_HOME="\$ONCOLENS_ROOT/cache/torch"
export PIP_CACHE_DIR="\$ONCOLENS_ROOT/cache/pip"
export WANDB_DIR="\$ONCOLENS_ROOT/runs/wandb"
$MARK_END
EOT
  echo "Added environment variables to ~/.bashrc"
fi
df -h "$ROOT" | tail -1
echo "Done. Run: source ~/.bashrc"
