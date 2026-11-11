"""Configuration loading and global seeding.

All tunable parameters live in ``config.yaml`` at the project root. Paths in the
config are relative to the project root and are resolved here, so no module
needs absolute paths.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None, set_seed: bool = True) -> dict[str, Any]:
    """Load the YAML config, resolve paths and (optionally) fix all seeds."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    cfg = copy.deepcopy(cfg)
    cfg["paths"] = {key: str(resolve_path(value)) for key, value in cfg["paths"].items()}
    for value in cfg["paths"].values():
        Path(value).mkdir(parents=True, exist_ok=True)

    if set_seed:
        set_seeds(int(cfg["seed"]))
    return cfg


def resolve_path(relative: str | Path) -> Path:
    """Resolve a project-relative path to an absolute one."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seeds(seed: int) -> None:
    """Fix the seed of every random source used in the project."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_rng(cfg: dict[str, Any], offset: int = 0) -> np.random.Generator:
    """Return an independent, reproducible NumPy generator derived from the config seed."""
    return np.random.default_rng(int(cfg["seed"]) + offset)
