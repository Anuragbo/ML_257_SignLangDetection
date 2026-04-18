"""
preprocessing.py
----------------
Extracts MediaPipe Holistic landmark sequences from WLASL videos.

For each video, uniform-samples 30 frames from the signed segment, then runs
MediaPipe Holistic to get left hand (63-D) + right hand (63-D) + pose (99-D)
= 225-D feature vector per frame.  Result per video: (30, 225) float32.

Uses the same hand normalization as Part 1 (wrist→origin, scale by wrist-MCP9).
Pose normalization: hip-midpoint→origin, scale by torso height.

Usage:
    python preprocessing.py
    python preprocessing.py --json ../data/available_videos.json \\
                             --video_dir ../data/videos \\
                             --output_dir ../data/sequences \\
                             --seq_len 30

Output (in output_dir/):
    X.npy          (N, 30, 225) float32  — landmark sequences
    y.npy          (N,)         int32    — class indices
    label_map.npy  dict  {idx: word}
    splits.npy     dict  {video_id: 'train'|'val'|'test'}
    video_ids.npy  list  parallel to X rows
"""

import os
import json
import argparse
import urllib.request
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# ── Model paths ──────────────────────────────────────────────────────────────
HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
)
HOLISTIC_MODEL_PATH = os.path.join(os.path.dirname(__file__), "holistic_landmarker.task")

# Default paths
DEFAULT_JSON       = os.path.join(os.path.dirname(__file__), "..", "data", "available_videos.json")
DEFAULT_VIDEO_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "videos")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sequences")
SEQ_LEN            = 30   # frames per video

# ── Feature dimensions ────────────────────────────────────────────────────────
DIM_HAND  = 63   # 21 landmarks × 3
DIM_POSE  = 99   # 33 landmarks × 3
DIM_TOTAL = DIM_HAND * 2 + DIM_POSE  # 225


# ── Model setup ───────────────────────────────────────────────────────────────

def ensure_holistic_model():
    """Download holistic_landmarker.task if not already present."""
    if not os.path.exists(HOLISTIC_MODEL_PATH):
        print(f"Downloading holistic landmarker model -> {HOLISTIC_MODEL_PATH}")
        urllib.request.urlretrieve(HOLISTIC_MODEL_URL, HOLISTIC_MODEL_PATH)
        print("Download complete.")


def make_holistic_detector():
    """
    Create a HolisticLandmarker in IMAGE running mode.
    IMAGE mode processes each frame independently — no timestamp tracking
    required, and a single detector instance can be reused across all videos,
    which is significantly faster than creating a new detector per video.
    """
    base_options = mp_python.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH)
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.IMAGE,
        min_face_detection_confidence=0.1,
        min_pose_detection_confidence=0.1,
        min_pose_landmarks_confidence=0.1,
        min_hand_landmarks_confidence=0.1,
    )
    return mp_vision.HolisticLandmarker.create_from_options(options)


# ── Normalization helpers (reused from Part 1 for hands) ─────────────────────

def normalize_hand(landmarks_list):
    """
    Convert 21 MediaPipe NormalizedLandmark objects → (63,) float32.
    Normalization:
      - Translate so wrist (landmark 0) is at origin.
      - Scale by wrist-to-middle-finger-MCP (landmark 9) distance.
    Returns a zeros array if landmarks_list is None or empty.
    """
    if not landmarks_list:
        return np.zeros(DIM_HAND, dtype=np.float32)

    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list], dtype=np.float32)  # (21,3)
    coords -= coords[0]                     # translate wrist to origin
    scale = np.linalg.norm(coords[9])
    if scale > 1e-6:
        coords /= scale
    return coords.flatten()                 # (63,)


def normalize_pose(landmarks_list):
    """
    Convert 33 MediaPipe NormalizedLandmark objects → (99,) float32.
    Normalization:
      - Translate so hip midpoint (landmarks 23 & 24) is at origin.
      - Scale by torso height: midpoint(shoulders 11,12) to midpoint(hips 23,24).
    Returns a zeros array if landmarks_list is None or empty.
    """
    if not landmarks_list:
        return np.zeros(DIM_POSE, dtype=np.float32)

    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list], dtype=np.float32)  # (33,3)

    hip_mid      = (coords[23] + coords[24]) / 2.0
    shoulder_mid = (coords[11] + coords[12]) / 2.0
    coords      -= hip_mid                               # translate hip to origin

    torso_height = np.linalg.norm(shoulder_mid - hip_mid)
    if torso_height > 1e-6:
        coords /= torso_height

    return coords.flatten()                              # (99,)


# ── Per-video extraction ──────────────────────────────────────────────────────

def extract_sequence(video_path, frame_start, frame_end, detector, seq_len=SEQ_LEN):
    """
    Extract a (seq_len, 225) landmark sequence from a video file.

    Uniformly samples seq_len frame indices from [frame_start, frame_end],
    seeks to each frame, runs HolisticLandmarker, and concatenates features.

    Returns:
        np.ndarray of shape (seq_len, 225) float32, or None if the video
        could not be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Clamp frame_start / frame_end to valid range
    frame_start = max(0, frame_start - 1)        # JSON is 1-indexed → 0-indexed
    if frame_end <= 0 or frame_end >= total_frames:
        frame_end = total_frames - 1
    else:
        frame_end = frame_end - 1                 # 1-indexed → 0-indexed

    frame_end = max(frame_end, frame_start)       # ensure valid range

    # Uniform sample indices
    sample_indices = np.linspace(frame_start, frame_end, seq_len, dtype=int)

    sequence = []
    for t, fidx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            sequence.append(np.zeros(DIM_TOTAL, dtype=np.float32))
            continue

        # Resize to fixed resolution so detector state stays consistent
        # across videos of different sizes (avoids segmentation smoother crash)
        frame_bgr = cv2.resize(frame_bgr, (640, 480))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # IMAGE mode: each frame is independent, no timestamp needed
        result = detector.detect(mp_image)

        lh   = normalize_hand(result.left_hand_landmarks)   # (63,)
        rh   = normalize_hand(result.right_hand_landmarks)  # (63,)
        pose = normalize_pose(result.pose_landmarks)         # (99,)

        sequence.append(np.concatenate([lh, rh, pose]))     # (225,)

    cap.release()
    return np.stack(sequence, axis=0).astype(np.float32)    # (30, 225)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(json_path, video_dir, seq_len=SEQ_LEN):
    """
    Process all records in available_videos.json.

    Returns:
        X          np.ndarray (N, seq_len, 225)
        y          np.ndarray (N,)
        label_map  dict {label_idx: gloss_word}
        splits     dict {video_id: split_str}
        video_ids  list of video_id strings parallel to X
    """
    with open(json_path, "r") as f:
        records = json.load(f)

    # Drop glosses with no downloaded videos and re-index contiguously (0 to N-1)
    active_glosses   = sorted({r["gloss"] for r in records})
    gloss_to_new_idx = {g: i for i, g in enumerate(active_glosses)}
    for r in records:
        r["label_idx"] = gloss_to_new_idx[r["gloss"]]

    label_map = {i: g for g, i in gloss_to_new_idx.items()}
    print(f"Active glosses (with downloaded videos): {len(active_glosses)}")

    splits    = {}
    video_ids = []
    X_list    = []
    y_list    = []

    skipped = 0
    zero_frame_count = defaultdict(int)  # counts how many frames were all-zero per video

    ensure_holistic_model()

    # Single detector reused across all videos (IMAGE mode allows this)
    detector = make_holistic_detector()
    try:
        for rec in tqdm(records, desc="Extracting sequences", unit="video"):
            video_path = os.path.join(video_dir, f"{rec['video_id']}.mp4")
            if not os.path.exists(video_path):
                skipped += 1
                continue

            seq = extract_sequence(
                video_path,
                rec["frame_start"],
                rec["frame_end"],
                detector,
                seq_len=seq_len,
            )

            if seq is None:
                skipped += 1
                continue

            # Count all-zero frames (diagnostic)
            zero_frames = int((seq.sum(axis=1) == 0).sum())
            zero_frame_count[rec["video_id"]] = zero_frames

            X_list.append(seq)
            y_list.append(rec["label_idx"])
            splits[rec["video_id"]] = rec["split"]
            video_ids.append(rec["video_id"])
    finally:
        detector.close()

    print(f"\nProcessed {len(X_list)} sequences  |  Skipped: {skipped}")

    # Diagnostic: zero-frame rate
    if X_list:
        all_zero = sum(zero_frame_count.values())
        total_frames = len(X_list) * seq_len
        print(f"Zero-detection frames: {all_zero}/{total_frames} "
              f"({100*all_zero/total_frames:.1f}%)")

    X = np.stack(X_list, axis=0).astype(np.float32)   # (N, 30, 225)
    y = np.array(y_list, dtype=np.int32)               # (N,)

    return X, y, label_map, splits, video_ids


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract WLASL landmark sequences")
    parser.add_argument("--json",       default=DEFAULT_JSON,
                        help="Path to available_videos.json")
    parser.add_argument("--video_dir",  default=DEFAULT_VIDEO_DIR,
                        help="Folder containing downloaded MP4s")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where to save .npy files")
    parser.add_argument("--seq_len",    type=int, default=SEQ_LEN,
                        help="Number of frames to sample per video")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X, y, label_map, splits, video_ids = build_dataset(
        args.json, args.video_dir, seq_len=args.seq_len
    )

    # Save
    np.save(os.path.join(args.output_dir, "X.npy"),         X)
    np.save(os.path.join(args.output_dir, "y.npy"),         y)
    np.save(os.path.join(args.output_dir, "label_map.npy"), label_map)
    np.save(os.path.join(args.output_dir, "splits.npy"),    splits)
    np.save(os.path.join(args.output_dir, "video_ids.npy"), np.array(video_ids))

    print(f"\nSaved to {args.output_dir}/")
    print(f"  X.npy        shape={X.shape}")
    print(f"  y.npy        shape={y.shape}")
    print(f"  label_map    {len(label_map)} classes")

    # Split breakdown
    from collections import Counter
    split_counts = Counter(splits.values())
    print("\n  Split counts:")
    for split, cnt in sorted(split_counts.items()):
        print(f"    {split:8s}: {cnt}")

    # Per-class sample counts
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Samples per class — min: {counts.min()}, max: {counts.max()}, "
          f"mean: {counts.mean():.1f}")


if __name__ == "__main__":
    main()
