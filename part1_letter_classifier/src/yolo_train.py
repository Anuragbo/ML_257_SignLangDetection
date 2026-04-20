"""
yolo_train.py
-------------
Train an Ultralytics YOLO **classification** model on ``data/yolo_cls_dataset``.

Prerequisite: run ``yolo_dataset.py`` once so ``train/`` and ``val/`` exist.

Example::

    python yolo_train.py --epochs 30 --imgsz 224
    python yolo_train.py --model yolov8s-cls.pt --batch 32 --device 0

The best weights are copied to ``part1_letter_classifier/models/yolo_cls_best.pt``.
Ultralytics also writes run artifacts under ``results/yolo_runs/``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from yolo_utils import (
    DEFAULT_YOLO_CLS_DATA,
    DEFAULT_YOLO_PROJECT,
    DEFAULT_YOLO_WEIGHTS,
    MODELS_DIR,
    require_ultralytics,
)


def _die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def main() -> None:
    require_ultralytics()
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Train YOLO classification on ASL class-folder data.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_YOLO_CLS_DATA,
        help="Folder containing train/ and val/ class subfolders (default: data/yolo_cls_dataset).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-cls.pt",
        help="Ultralytics cls checkpoint to start from (default: yolov8n-cls.pt). "
        "Other options: yolov8s-cls.pt, yolov8m-cls.pt, yolov11n-cls.pt, ...",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30).")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size (default: 224).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: '' (auto), cpu, 0, 0,1, ... (default: auto).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_YOLO_PROJECT,
        help="Ultralytics project directory (default: results/yolo_runs).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="cls",
        help="Run name under project (default: cls).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (Ultralytics default behavior; default: 15).",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting an existing run folder with the same name.",
    )
    parser.add_argument(
        "--weights-out",
        type=Path,
        default=DEFAULT_YOLO_WEIGHTS,
        help=f"Where to copy best.pt after training (default: {DEFAULT_YOLO_WEIGHTS.name}).",
    )
    args = parser.parse_args()

    data = args.data.resolve()
    train_dir = data / "train"
    val_dir = data / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        _die(
            f"Invalid YOLO cls data directory:\n  {data}\n\n"
            "Expected:\n"
            f"  {train_dir}/<class>/images...\n"
            f"  {val_dir}/<class>/images...\n\n"
            "Build it with:\n"
            "  python part1_letter_classifier/src/yolo_dataset.py\n"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    args.project.mkdir(parents=True, exist_ok=True)

    print("Starting YOLO classification training")
    print(f"  data    : {data}")
    print(f"  model   : {args.model}")
    print(f"  epochs  : {args.epochs}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  batch   : {args.batch}")
    print(f"  device  : {args.device or '(auto)'}")
    print(f"  project : {args.project}")
    print(f"  name    : {args.name}")

    model = YOLO(args.model)
    train_kw = dict(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        exist_ok=args.exist_ok,
    )
    if args.device:
        train_kw["device"] = args.device

    # Ultralytics handles train/val folders and metrics.
    model.train(**train_kw)

    run_dir = Path(args.project) / args.name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.is_file():
        _die(f"Training finished but best weights not found:\n  {best}")

    shutil.copy2(best, args.weights_out)
    print(f"\nCopied best weights -> {args.weights_out.resolve()}")
    print(f"Ultralytics run dir  -> {run_dir.resolve()}")
    if last.is_file():
        print(f"last.pt available   -> {last}")


if __name__ == "__main__":
    main()
