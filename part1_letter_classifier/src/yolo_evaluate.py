"""
yolo_evaluate.py
----------------
Evaluate a trained YOLO **classification** checkpoint on the ``val/`` split.

Reports accuracy, macro precision/recall/F1, per-class metrics, and saves:
  - ``results/yolo_cls_confusion.png``
  - ``results/yolo_cls_metrics.txt`` (human-readable summary)

Usage::

    python yolo_evaluate.py
    python yolo_evaluate.py --weights part1_letter_classifier/models/yolo_cls_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from yolo_utils import (
    DEFAULT_YOLO_CLS_DATA,
    DEFAULT_YOLO_WEIGHTS,
    RESULTS_DIR,
    ensure_weights_exist,
    load_yolo_model,
    require_ultralytics,
)


def _list_classes(val_root: Path) -> list[str]:
    """Sorted class folder names under val/."""
    names = sorted(
        p.name for p in val_root.iterdir() if p.is_dir() and len(p.name) == 1 and p.name.isalnum()
    )
    if not names:
        raise FileNotFoundError(f"No class folders under {val_root}")
    return names


def _gather_val_paths(val_root: Path, classes: list[str]) -> tuple[list[Path], list[str]]:
    """Return (file_paths, true_labels) for all images in val/."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths: list[Path] = []
    labels: list[str] = []
    for c in classes:
        cdir = val_root / c
        if not cdir.is_dir():
            continue
        for p in sorted(cdir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
                labels.append(c)
    if not paths:
        raise FileNotFoundError(f"No validation images found under {val_root}")
    return paths, labels


def evaluate_weights(
    weights: Path,
    data_root: Path,
    imgsz: int | None = None,
    out_dir: Path | None = None,
) -> dict:
    """
    Run inference on every image in ``data_root/val`` and compute sklearn metrics.

    Class index order follows ``sorted(val class folders)`` and must match training.
    """
    require_ultralytics()
    import cv2

    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    val_root = data_root / "val"
    if not val_root.is_dir():
        print(f"ERROR: Missing val folder: {val_root}", file=sys.stderr)
        raise SystemExit(1)

    classes = _list_classes(val_root)
    paths, y_true = _gather_val_paths(val_root, classes)

    model = load_yolo_model(weights)
    # Map name -> index for consistent confusion matrix ordering
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_pred: list[str] = []
    y_true_filtered: list[str] = []
    for p, t in zip(paths, y_true):
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"WARNING: could not read {p}; skipping.", file=sys.stderr)
            continue
        y_true_filtered.append(t)
        kwargs = {"verbose": False}
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        results = model.predict(bgr, **kwargs)
        if not results or results[0].probs is None:
            y_pred.append(classes[0])
            continue
        r = results[0]
        top1 = int(r.probs.top1)
        names = r.names or {}
        pred_name = str(names.get(top1, top1))
        y_pred.append(pred_name)

    y_true = y_true_filtered

    # Encode labels for sklearn metrics
    y_t = np.array([class_to_idx.get(t, -1) for t in y_true])
    y_p = np.array([class_to_idx.get(p, -1) for p in y_pred])
    if (y_t < 0).any() or (y_p < 0).any():
        print("WARNING: Unknown class label encountered; check val folder names vs model.names.", file=sys.stderr)

    acc = accuracy_score(y_t, y_p)
    macro_p = precision_score(y_t, y_p, average="macro", zero_division=0)
    macro_r = recall_score(y_t, y_p, average="macro", zero_division=0)
    macro_f1 = f1_score(y_t, y_p, average="macro", zero_division=0)

    report = classification_report(
        y_t,
        y_p,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("YOLO classification — validation metrics")
    print("=" * 60)
    print(f"Weights : {weights}")
    print(f"Val root: {val_root}")
    print(f"Samples : {len(y_t)}")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision (μ)  : {macro_p:.4f}")
    print(f"Recall (μ)       : {macro_r:.4f}")
    print(f"F1 (macro)       : {macro_f1:.4f}")
    print("\n" + report)

    cm = confusion_matrix(y_t, y_p, labels=list(range(len(classes))))

    fig_h = max(8, len(classes) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_h + 4, fig_h))
    sns.heatmap(
        cm,
        annot=len(classes) <= 15,
        fmt="d" if len(classes) <= 15 else "",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.2,
        ax=ax,
    )
    ax.set_title(f"YOLO cls — Confusion (acc={acc:.4f})", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = out_dir / "yolo_cls_confusion.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot -> {cm_path}")

    txt_path = out_dir / "yolo_cls_metrics.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"weights: {weights}\n")
        f.write(f"val_dir: {val_root}\n")
        f.write(f"accuracy: {acc:.6f}\n")
        f.write(f"precision_macro: {macro_p:.6f}\n")
        f.write(f"recall_macro: {macro_r:.6f}\n")
        f.write(f"f1_macro: {macro_f1:.6f}\n\n")
        f.write(report)
    print(f"Metrics text -> {txt_path}")

    return {
        "accuracy": acc,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "confusion_path": str(cm_path),
        "metrics_path": str(txt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO classification on val/ folder.")
    parser.add_argument("--weights", type=Path, default=None, help=f"Default: {DEFAULT_YOLO_WEIGHTS}")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_YOLO_CLS_DATA,
        help="Dataset root containing val/ (default: data/yolo_cls_dataset).",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference size.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Where to save plots and metrics text (default: results/).",
    )
    args = parser.parse_args()

    w = ensure_weights_exist(args.weights)
    evaluate_weights(w, args.data.resolve(), imgsz=args.imgsz, out_dir=args.output_dir.resolve())


if __name__ == "__main__":
    main()
