"""Run the complete data, feature, split, and evaluation pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
STAGES = [
    ("load", [sys.executable, "-m", "src.load"]),
    ("features", [sys.executable, "-m", "src.features"]),
    ("split", [sys.executable, "-m", "src.split"]),
    ("evaluation", [sys.executable, "-m", "src.evaluate"]),
]


def run_stage(name: str, command: list[str]) -> float:
    """Run one pipeline command and report its elapsed time."""
    print(f"\n== Etapa: {name}", flush=True)
    started = time.perf_counter()
    try:
        # Unbuffered child output: progress shows up live even when redirected to a file.
        env = os.environ | {"PYTHONUNBUFFERED": "1"}
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)
    finally:
        elapsed = time.perf_counter() - started
        print(f"== {name}: {elapsed:.1f} s", flush=True)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="use the raw files already present in data/raw",
    )
    args = parser.parse_args()

    stages = list(STAGES)
    if not args.skip_download:
        stages.insert(
            0,
            ("download", [sys.executable, "-m", "data.download"]),
        )

    total_started = time.perf_counter()
    for name, command in stages:
        run_stage(name, command)
    print(f"\nPipeline concluído em {time.perf_counter() - total_started:.1f} s.")


if __name__ == "__main__":
    main()
