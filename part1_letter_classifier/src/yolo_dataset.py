"""
yolo_dataset.py
---------------
Convert the existing **class-folder** ASL dataset into Ultralytics **classification** layout.

Source layout (same as ``preprocessing.py``)::

    data/asl_dataset/<one char class>/*.jpg

Output layout (Ultralytics ``yolo classify train``)::

    data/yolo_cls_dataset/train/<class>/*
    data/yolo_cls_dataset/val/<class>/*

Each image is copied into either ``train`` or ``val`` with a per-class random split
so both sets contain every class that has enough images.

Usage::

    python yolo_dataset.py
    python yolo_dataset.py --val-ratio 0.25 --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from yolo_utils import DEFAULT_ASL_DIR, DEFAULT_YOLO_CLS_DATA

# Image extensions (aligned with preprocessing.py)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_class_folders(data_dir: Path) -> list[str]:
    """
    Return sorted single-character alphanumeric folder names (0-9, a-z).

    Matches the filtering logic in ``preprocessing.build_dataset`` so MediaPipe and
    YOLO see the same class list.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"ASL dataset folder not found:\n  {data_dir}\n\n"
            "Add images under data/asl_dataset/ with one subfolder per class, "
            "or run dataset_download.py / run_pipeline.py to fetch data."
        )

    folders = sorted(
        [
            p.name
            for p in data_dir.iterdir()
            if p.is_dir() and len(p.name) == 1 and p.name.isalnum()
        ]
    )
    if not folders:
        raise ValueError(
            f"No class folders found under {data_dir}. "
            "Expected subfolders named with a single letter or digit each."
        )
    return folders


def collect_images(class_dir: Path) -> list[Path]:
    """Return image paths for one class folder."""
    out: list[Path] = []
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return sorted(out)


def prepare_yolo_cls_dataset(
    src_dir: Path,
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    clean: bool = False,
) -> dict[str, int]:
    """
    Split each class into train/val and copy files into Ultralytics classify layout.

    Parameters
    ----------
    src_dir : Path
        ``asl_dataset`` root.
    out_dir : Path
        Will contain ``train/`` and ``val/`` subfolders.
    val_ratio : float
        Fraction of images per class reserved for validation (default 0.2).
    seed : int
        RNG seed for reproducible splits.
    clean : bool
        If True, delete ``out_dir`` before writing.

    Returns
    -------
    dict
        Statistics: ``n_train``, ``n_val``, ``n_classes``, ``skipped_classes``.
    """
    if not (0.0 < val_ratio < 0.95):
        raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

    class_names = list_class_folders(src_dir)

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    train_root = out_dir / "train"
    val_root = out_dir / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    n_train = 0
    n_val = 0
    skipped = 0

    for cls in class_names:
        files = collect_images(src_dir / cls)
        if not files:
            skipped += 1
            continue

        rng.shuffle(files)
        n_val_cls = max(1, int(round(len(files) * val_ratio)))
        # If only one image, put it in train (val would be empty otherwise)
        if len(files) == 1:
            n_val_cls = 0

        val_files = set(files[-n_val_cls:]) if n_val_cls else set()
        train_files = [f for f in files if f not in val_files]

        (train_root / cls).mkdir(parents=True, exist_ok=True)
        (val_root / cls).mkdir(parents=True, exist_ok=True)

        for fpath in tqdm(train_files, desc=f"train [{cls}]", leave=False):
            dest = train_root / cls / fpath.name
            shutil.copy2(fpath, dest)
            n_train += 1

        for fpath in tqdm(list(val_files), desc=f"val [{cls}]", leave=False):
            dest = val_root / cls / fpath.name
            shutil.copy2(fpath, dest)
            n_val += 1

    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_classes": len(class_names) - skipped,
        "skipped_classes": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build YOLO classification dataset (train/val) from asl_dataset class folders."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_ASL_DIR,
        help=f"Source ASL class-folder root (default: {DEFAULT_ASL_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_YOLO_CLS_DATA,
        help=f"Output root with train/ and val/ (default: {DEFAULT_YOLO_CLS_DATA})",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of images per class for validation (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output folder before rebuilding (recommended if classes changed).",
    )
    args = parser.parse_args()

    print(f"Source: {args.src.resolve()}")
    print(f"Output: {args.out.resolve()}")
    print(f"val_ratio={args.val_ratio}, seed={args.seed}")

    stats = prepare_yolo_cls_dataset(
        args.src, args.out, val_ratio=args.val_ratio, seed=args.seed, clean=args.clean
    )

    print("\nDone.")
    print(f"  Train images : {stats['n_train']}")
    print(f"  Val images   : {stats['n_val']}")
    print(f"  Classes      : {stats['n_classes']}")
    if stats["skipped_classes"]:
        print(f"  Empty classes skipped: {stats['skipped_classes']}")

    # Minimal YAML for documentation (Ultralytics uses directory layout; this is optional)
    yaml_path = args.out.parent / "yolo_cls_dataset.yaml"
    class_names = list_class_folders(args.src)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# YOLO classification dataset (reference only).\n")
        f.write("# Training command uses --data pointing to the folder that contains train/ and val/.\n")
        f.write(f"path: {args.out.as_posix()}\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names: [" + ", ".join(repr(c) for c in class_names) + "]\n")
    print(f"\nWrote reference config: {yaml_path}")


if __name__ == "__main__":
    main()
