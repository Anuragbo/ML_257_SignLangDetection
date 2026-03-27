"""
preprocessing.py
----------------
Extracts and normalizes MediaPipe hand landmarks from the ASL static image dataset.
Uses the new MediaPipe Tasks API (0.10+).

Usage:
    python preprocessing.py --data_dir ../data/asl_dataset --output_dir ../data

Output:
    ../data/X.npy       - shape (n_samples, 63)  float32 landmark features
    ../data/y.npy       - shape (n_samples,)     int   label indices
    ../data/label_map.npy - dict mapping index -> letter/digit
"""

import os
import argparse
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
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand landmarker model -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


# ── Landmark extraction ───────────────────────────────────────────────────────

def make_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
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


def extract_landmarks(image_bgr, detector):
    """
    Run MediaPipe Hands on a single BGR image.
    Tries the original image first, then a padded version for tight crops.
    Returns a (63,) float32 array of normalized (x,y,z) landmarks,
    or None if no hand is detected.
    """
    for img in [image_bgr, pad_image(image_bgr)]:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result    = detector.detect(mp_image)
        if result.hand_landmarks:
            break

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]   # list of 21 NormalizedLandmark
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)  # (21, 3)

    # Normalize: translate so wrist (landmark 0) is at origin
    coords -= coords[0]

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
    parser.add_argument("--data_dir",   default="../data/asl_dataset",
                        help="Root folder with one subfolder per letter/digit")
    parser.add_argument("--output_dir", default="../data",
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
