"""
preprocessing.py
----------------
Extracts and normalizes MediaPipe hand landmarks from the ASL static image dataset.
Uses the new MediaPipe Tasks API (0.10+).

Usage:
    python preprocessing.py
    # Defaults: data from part1_letter_classifier/data/asl_dataset, output to .../data

Output (under --output_dir, default part1_letter_classifier/data):
    X.npy         - shape (n_samples, 63)  float32 landmark features
    y.npy         - shape (n_samples,)     int   label indices
    label_map.npy - dict mapping index -> letter/digit
"""

import os
import argparse
from pathlib import Path
import urllib.request
import numpy as np
import cv2
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Model download ────────────────────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
_MODEL_DIR = Path(__file__).resolve().parent
_PART1_ROOT = _MODEL_DIR.parent
MODEL_PATH = str(_MODEL_DIR / "hand_landmarker.task")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand landmarker model -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


# ── Landmark extraction ───────────────────────────────────────────────────────

def make_detector(num_hands: int = 1):
    base_options = mp_python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def pad_image(image_bgr, pad_ratio=0.3):
    """Add white padding around a tightly-cropped hand image so MediaPipe has context."""
    h, w = image_bgr.shape[:2]
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)
    return cv2.copyMakeBorder(image_bgr, pad_h, pad_h, pad_w, pad_w,
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _hand_label(result, idx: int) -> str | None:
    """
    Best-effort handedness label ("Left"/"Right") for hand idx.
    MediaPipe Tasks returns a per-hand list of classifications under result.handedness.
    """
    try:
        handed = getattr(result, "handedness", None)
        if not handed:
            return None
        # handed[idx] is a list of Classification objects; take the top one.
        top = handed[idx][0]
        return getattr(top, "category_name", None) or getattr(top, "display_name", None)
    except Exception:
        return None


def _pick_hand_index(result, prefer: str = "any") -> int:
    prefer = (prefer or "any").lower().strip()
    n = len(result.hand_landmarks or [])
    if n <= 1 or prefer == "any":
        return 0
    prefer_norm = "left" if prefer.startswith("l") else "right"
    for i in range(n):
        lab = (_hand_label(result, i) or "").lower()
        if prefer_norm in lab:
            return i
    return 0


def extract_landmarks(
    image_bgr,
    detector,
    *,
    hand: str = "any",
    canonicalize_left_to_right: bool = True,
):
    """
    Run MediaPipe Hands on a single BGR image.
    Tries the original image first, then a padded version for tight crops.
    Returns a (63,) float32 array of normalized (x,y,z) landmarks,
    or None if no hand is detected.
    """
    result = None
    for img in [image_bgr, pad_image(image_bgr)]:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            break

    if result is None or not result.hand_landmarks:
        return None

    hand_idx = _pick_hand_index(result, prefer=hand)
    landmarks = result.hand_landmarks[hand_idx]   # list of 21 NormalizedLandmark
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)  # (21, 3)

    # Normalize: translate so wrist (landmark 0) is at origin
    coords -= coords[0]

    # Canonicalize handedness: mirror left-hand x to match right-hand geometry.
    # This lets one model handle both hands without retraining.
    if canonicalize_left_to_right:
        lab = (_hand_label(result, hand_idx) or "").lower()
        if "left" in lab:
            coords[:, 0] *= -1.0

    # Scale: divide by wrist-to-middle-finger-MCP distance (landmark 9)
    # Makes features invariant to hand size and camera distance
    scale = np.linalg.norm(coords[9])
    if scale > 1e-6:
        coords /= scale

    return coords.flatten()  # (63,)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(data_dir):
    """
    Walk data_dir expecting subdirectories named by letter/digit (0-9, a-z).
    Returns X (n, 63), y (n,), label_map {idx: letter}.
    """
    label_map = {}
    X, y = [], []
    skipped = 0

    # Only keep single-character alphanumeric folders (skips nested 'asl_dataset' dirs)
    folders = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and len(d) == 1 and d.isalnum()
    ])

    print(f"Found {len(folders)} classes: {folders}")

    # Create ONE detector (reused across all images for efficiency)
    detector = make_detector()

    for idx, folder in enumerate(folders):
        label_map[idx] = folder
        folder_path = os.path.join(data_dir, folder)
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        for fname in tqdm(image_files, desc=f"[{folder}]", leave=False):
            img_path = os.path.join(folder_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            features = extract_landmarks(img, detector)
            if features is None:
                skipped += 1
                continue

            X.append(features)
            y.append(idx)

    detector.close()
    print(f"\nTotal samples: {len(X)}  |  Skipped (no hand detected / bad image): {skipped}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_map


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(_PART1_ROOT / "data" / "asl_dataset"),
                        help="Root folder with one subfolder per letter/digit")
    parser.add_argument("--output_dir", default=str(_PART1_ROOT / "data"),
                        help="Where to save X.npy, y.npy, label_map.npy")
    args = parser.parse_args()

    ensure_model()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Extracting landmarks...")
    X, y, label_map = build_dataset(args.data_dir)

    np.save(os.path.join(args.output_dir, "X.npy"),         X)
    np.save(os.path.join(args.output_dir, "y.npy"),         y)
    np.save(os.path.join(args.output_dir, "label_map.npy"), label_map)

    print(f"\nSaved to {args.output_dir}/")
    print(f"  X.npy        shape={X.shape}")
    print(f"  y.npy        shape={y.shape}")
    print(f"  label_map    {label_map}")


if __name__ == "__main__":
    main()
