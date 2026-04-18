"""
yolo_predict.py
---------------
Run YOLO **classification** inference on a still image or webcam.

Examples::

    python yolo_predict.py --source path/to/image.jpg
    python yolo_predict.py --source 0
    python yolo_predict.py --weights part1_letter_classifier/models/yolo_cls_best.pt --imgsz 224

This script is independent of MediaPipe; it classifies the whole image. For best results,
use crops that mostly contain the hand (similar to the training images).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2

from yolo_utils import default_weights_path, ensure_weights_exist, load_yolo_model, predict_letter_bgr

BUFFER_SIZE = 7


def predict_image(weights: Path, source: Path, imgsz: int | None) -> int:
    """Print prediction for one image file. Returns process exit code."""
    if not source.is_file():
        print(f"ERROR: File not found: {source}", file=sys.stderr)
        return 1

    model = load_yolo_model(weights)
    bgr = cv2.imread(str(source))
    if bgr is None:
        print(f"ERROR: Could not read image: {source}", file=sys.stderr)
        return 1

    letter, conf, top = predict_letter_bgr(model, bgr, imgsz=imgsz)
    print(f"Top prediction: {letter}  ({conf:.4f})")
    print("Top predictions:")
    for row in top[:8]:
        print(f"  {row['letter']}: {row['confidence']:.4f}")
    return 0


def run_webcam(weights: Path, camera: int, imgsz: int | None, threshold: float) -> int:
    """Open webcam and overlay stabilized predictions."""
    model = load_yolo_model(weights)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pred_buffer: deque[str | None] = deque(maxlen=BUFFER_SIZE)
    prev_time = time.time()
    fps = 0.0

    print(f"Camera {camera} opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        letter, conf, _ = predict_letter_bgr(model, frame, imgsz=imgsz)

        display_letter = None
        display_conf = 0.0
        if letter is not None and conf >= threshold:
            pred_buffer.append(letter)
        else:
            pred_buffer.append(None)

        valid = [p for p in pred_buffer if p is not None]
        if valid:
            stable, count = Counter(valid).most_common(1)[0]
            if count >= 3:
                display_letter = stable
                display_conf = float(conf)

        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (340, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "YOLO classification", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        if display_letter:
            color = (0, 200, 0) if display_conf > 0.7 else (0, 230, 255)
            cv2.putText(frame, display_letter.upper(), (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)
            cv2.putText(
                frame,
                f"{display_conf:.0%}",
                (min(200, w - 120), 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(frame, "No confident class", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 255), 2)

        curr = time.time()
        fps = 0.85 * fps + 0.15 * (1.0 / max(curr - prev_time, 1e-6))
        prev_time = curr
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        cv2.putText(frame, "q = quit", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("YOLO ASL — Part 1", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO classification inference (image or webcam).")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=f"Path to yolo_cls_best.pt (default: {default_weights_path()})",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image path, or a camera index like 0 for webcam.",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference size (multiple of 32).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Min confidence to accept a label in webcam mode (default: 0.35).",
    )
    args = parser.parse_args()

    weights = ensure_weights_exist(args.weights)

    src = str(args.source).strip()
    if src.isdigit():
        raise SystemExit(run_webcam(weights, int(src), args.imgsz, args.threshold))

    raise SystemExit(predict_image(weights, Path(src), args.imgsz))


if __name__ == "__main__":
    main()
