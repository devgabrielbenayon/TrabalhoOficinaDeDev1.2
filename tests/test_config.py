"""Smoke tests for configuration loading and seeding."""

import random
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, load_config


def test_config_loads_and_resolves_paths() -> None:
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["recommend"]["n_values"] == [5, 10]
    for value in cfg["paths"].values():
        assert Path(value).is_absolute()
        assert Path(value).is_relative_to(PROJECT_ROOT)


def test_seed_is_reproducible() -> None:
    load_config()
    a = (random.random(), np.random.rand())
    load_config()
    b = (random.random(), np.random.rand())
    assert a == b
