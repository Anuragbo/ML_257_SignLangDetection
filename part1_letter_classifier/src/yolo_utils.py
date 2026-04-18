"""
yolo_utils.py
-------------
Shared paths and small helpers for the YOLO **classification** pipeline (Ultralytics).

This project uses **YOLO classification** (not detection) because the ASL dataset is
organized as one folder per class with full-image crops — no bounding-box labels.
Detection would require manual box annotation; classification reuses existing images.

Typical layout after conversion (see ``yolo_dataset.py``)::

    part1_letter_classifier/data/yolo_cls_dataset/
      train/<class_name>/*.jpg
      val/<class_name>/*.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Repo-relative roots (same pattern as preprocessing.py / demo.py)
_SRC_DIR = Path(__file__).resolve().parent
PART1_ROOT = _SRC_DIR.parent
DATA_DIR = PART1_ROOT / "data"
MODELS_DIR = PART1_ROOT / "models"
RESULTS_DIR = PART1_ROOT / "results"

# Default ASL image tree (same as preprocessing)
DEFAULT_ASL_DIR = DATA_DIR / "asl_dataset"

# Ultralytics classify expects a folder containing ``train/`` and ``val/`` subfolders.
DEFAULT_YOLO_CLS_DATA = DATA_DIR / "yolo_cls_dataset"

# Single canonical checkpoint copied after training (see ``yolo_train.py``)
DEFAULT_YOLO_WEIGHTS = MODELS_DIR / "yolo_cls_best.pt"

# Training runs (Ultralytics project directory) — kept under results/ for easy cleanup
DEFAULT_YOLO_PROJECT = RESULTS_DIR / "yolo_runs"


def require_ultralytics():
    """
    Import Ultralytics YOLO with a clear error if the dependency is missing.

    Returns
    -------
    module
        The ``ultralytics`` package (for ``from ultralytics import YOLO``).
    """
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        print(
            "ERROR: The YOLO pipeline requires the `ultralytics` package.\n"
            "Install it with:\n"
            "  pip install ultralytics\n"
            "Or install all project requirements:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    return ultralytics


def default_weights_path() -> Path:
    """Return the path to the saved best weights (may not exist yet)."""
    return DEFAULT_YOLO_WEIGHTS


def ensure_weights_exist(weights: Path | str | None) -> Path:
    """
    Resolve ``weights`` to a path and verify the file exists.

    Parameters
    ----------
    weights : Path, str, or None
        Checkpoint ``.pt`` file. If None, uses ``DEFAULT_YOLO_WEIGHTS``.
    """
    p = Path(weights) if weights is not None else DEFAULT_YOLO_WEIGHTS
    if not p.is_file():
        print(
            f"ERROR: YOLO weights not found:\n  {p}\n\n"
            "Train a model first, for example:\n"
            "  python part1_letter_classifier/src/yolo_dataset.py\n"
            "  python part1_letter_classifier/src/yolo_train.py --epochs 30 --imgsz 224\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return p


def load_yolo_model(weights: Path | str):
    """
    Load a YOLO **classification** model from a ``.pt`` checkpoint.

    Parameters
    ----------
    weights : Path or str
        Path to ``yolo_cls_best.pt`` (or any Ultralytics cls checkpoint).
    """
    require_ultralytics()
    from ultralytics import YOLO

    return YOLO(str(weights))


def topk_from_result(result: Any, k: int = 8) -> list[dict[str, Any]]:
    """
    Build a list of ``{letter, confidence}`` dicts from one Ultralytics result.

    Works for classification tasks where ``result.probs`` is set.
    """
    if not hasattr(result, "probs") or result.probs is None:
        return []

    probs = result.probs
    names = getattr(result, "names", None) or {}
    # top5 is available; for k>5 we take raw tensor
    data = probs.data
    if data is None:
        return []

    import torch

    if isinstance(data, torch.Tensor):
        flat = data.flatten().float()
    else:
        flat = torch.as_tensor(data).flatten().float()

    k = min(k, int(flat.numel()))
    vals, idx = torch.topk(flat, k=k)
    out: list[dict[str, Any]] = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        name = names.get(int(i), str(int(i)))
        out.append({"letter": str(name), "confidence": float(v)})
    return out


def predict_letter_bgr(model, frame_bgr, imgsz: int | None = None) -> tuple[str | None, float, list[dict]]:
    """
    Run classification on a single OpenCV BGR frame.

    Returns
    -------
    letter : str or None
        Predicted class label (folder name, e.g. ``a``).
    confidence : float
        Probability of the top class.
    top_predictions : list of dict
        Up to 8 entries for API / UI display.
    """
    # Ultralytics accepts numpy HWC BGR uint8
    kwargs: dict[str, Any] = {"verbose": False}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    results = model.predict(frame_bgr, **kwargs)
    if not results:
        return None, 0.0, []

    r = results[0]
    if r.probs is None:
        return None, 0.0, []

    top1_idx = int(r.probs.top1)
    conf = float(r.probs.top1conf)
    names = r.names or {}
    letter = str(names.get(top1_idx, top1_idx))
    top = topk_from_result(r, k=8)
    return letter, conf, top
