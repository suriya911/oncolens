"""Hydra config helpers usable outside a Hydra entry point (tests, notebooks)."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=overrides or [])
