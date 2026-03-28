#!/usr/bin/env python3
"""
Run the full Part 1 pipeline from a clean state:

  1. Preprocessing  — images under data/asl_dataset → X.npy, y.npy, label_map.npy
  2. Training       — SVM, Random Forest, MLP, and CNN (--mode all)
  3. Evaluation     — metrics, confusion matrices, comparison chart in results/

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-preprocessing   # reuse existing .npy files
    python run_pipeline.py --skip-training        # only evaluate existing models
    python run_pipeline.py --no-download          # fail if ASL images are missing (no Kaggle fetch)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "part1_letter_classifier" / "src"
DATA_DIR = REPO_ROOT / "part1_letter_classifier" / "data"
DATA_ASL = DATA_DIR / "asl_dataset"

from dataset_download import ensure_dataset
PREPROCESSING = SRC / "preprocessing.py"
TRAIN = SRC / "train.py"
EVALUATE = SRC / "evaluate.py"


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def check_dataset() -> None:
    if not DATA_ASL.is_dir():
        die(
            f"Dataset folder not found:\n  {DATA_ASL}\n\n"
            "Add the ASL image dataset there with one subfolder per class "
            "(single character: 0-9, a-z)."
        )
    valid = [
        p
        for p in DATA_ASL.iterdir()
        if p.is_dir() and len(p.name) == 1 and p.name.isalnum()
    ]
    if not valid:
        die(
            f"No class folders found under:\n  {DATA_ASL}\n\n"
            "Expected subfolders named with a single letter or digit each."
        )


def run_step(title: str, script: Path, extra_args: list[str] | None = None) -> None:
    if not script.is_file():
        die(f"Missing script: {script}")
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + prev if prev else "")
    args = [sys.executable, str(script), *(extra_args or [])]
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(" ", " ".join(args), "\n", flush=True)
    proc = subprocess.run(args, cwd=REPO_ROOT, env=env)
    if proc.returncode != 0:
        die(f"Step failed: {title} (exit {proc.returncode})", proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing, training, and evaluation.")
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip landmark extraction (requires existing X.npy, y.npy, label_map.npy).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (only run evaluation on saved models).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download the ASL dataset from Kaggle if images are missing.",
    )
    args = parser.parse_args()

    if not args.skip_preprocessing:
        ensure_dataset(DATA_ASL, no_download=args.no_download)
        check_dataset()

    if not args.skip_preprocessing:
        run_step("1/3 Preprocessing (MediaPipe landmarks)", PREPROCESSING)

    if not args.skip_training:
        run_step("2/3 Training (SVM, RF, MLP, CNN)", TRAIN, ["--mode", "all"])

    run_step("3/3 Evaluation (metrics + plots)", EVALUATE)

    print(f"\n{'=' * 60}\nPipeline finished successfully.\n{'=' * 60}")


if __name__ == "__main__":
    main()
