"""
demo.py
-------
Real-time ASL letter recognition using your webcam.

Opens the camera, detects hand landmarks via MediaPipe, feeds them to a
trained sklearn model, and displays the predicted letter on screen.

Usage:
    python demo.py                        # default: MLP model, camera 0
    python demo.py --model svm            # use SVM
    python demo.py --model rf             # use Random Forest
    python demo.py --threshold 0.5        # lower confidence threshold
    python demo.py --camera 1             # use a different camera

Controls:
    q  - quit
    1  - switch to SVM
    2  - switch to Random Forest
    3  - switch to MLP
"""

import os
import sys
import time
import argparse
from collections import deque, Counter

import cv2
import numpy as np
import joblib

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Reuse the landmark extraction logic from preprocessing
from preprocessing import extract_landmarks, ensure_model, MODEL_PATH

# ── Constants ────────────────────────────────────────────────────────────────
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
BUFFER_SIZE = 7          # rolling window for prediction stabilization
COLORS = {
    "green":  (0, 200, 0),
    "red":    (0, 0, 200),
    "blue":   (200, 120, 0),
    "white":  (255, 255, 255),
    "black":  (0, 0, 0),
    "yellow": (0, 230, 255),
    "gray":   (180, 180, 180),
}


# ── MediaPipe detector (VIDEO mode for webcam) ──────────────────────────────

def make_video_detector():
    """Create a HandLandmarker configured for single-image mode (works frame-by-frame)."""
    ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ── Drawing helpers ──────────────────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle  (0→9 = wrist to MCP)
    (0,13),(13,14),(14,15),(15,16), # ring    (0→13 = wrist to MCP)
    (0,17),(17,18),(18,19),(19,20), # pinky   (0→17 = wrist to MCP)
    (5,9),(9,13),(13,17),           # palm cross-connections
]


def draw_hand_landmarks(frame, result):
    """Draw the 21 hand landmarks and connections on the frame."""
    if not result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    landmarks = result.hand_landmarks[0]

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
        x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), COLORS["blue"], 2)

    # Draw landmark points
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, COLORS["yellow"], -1)
        cv2.circle(frame, (cx, cy), 4, COLORS["black"], 1)


def draw_bounding_box(frame, result):
    """Draw a bounding box around the detected hand."""
    if not result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    landmarks = result.hand_landmarks[0]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    margin = 20
    x1, y1 = int(min(xs)) - margin, int(min(ys)) - margin
    x2, y2 = int(max(xs)) + margin, int(max(ys)) + margin
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["green"], 2)


def draw_prediction(frame, letter, confidence, model_name):
    """Draw the predicted letter, confidence bar, and model name."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay for the info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 130), COLORS["black"], -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Model name
    cv2.putText(frame, f"Model: {model_name.upper()}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["gray"], 1)

    # Predicted letter (large)
    if letter:
        color = COLORS["green"] if confidence > 0.7 else COLORS["yellow"]
        cv2.putText(frame, letter.upper(), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)

        # Confidence bar
        bar_w = int(180 * confidence)
        cv2.rectangle(frame, (100, 60), (280, 80), COLORS["gray"], 1)
        cv2.rectangle(frame, (100, 60), (100 + bar_w, 80), color, -1)
        cv2.putText(frame, f"{confidence:.0%}", (100, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["white"], 1)
    else:
        cv2.putText(frame, "No hand", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS["red"], 2)


def draw_fps(frame, fps):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["green"], 2)


def draw_instructions(frame):
    h, w = frame.shape[:2]
    instructions = "1=SVM  2=RF  3=MLP  |  q=Quit"
    cv2.putText(frame, instructions, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["gray"], 1)


# ── Main loop ────────────────────────────────────────────────────────────────

def load_model(model_name):
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        print(f"ERROR: Model file not found: {path}")
        print("Run train.py first to train the models.")
        sys.exit(1)
    return joblib.load(path)


def main():
    parser = argparse.ArgumentParser(description="Real-time ASL letter recognition")
    parser.add_argument("--model", choices=["svm", "rf", "mlp"], default="mlp",
                        help="Which model to use (default: mlp)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Minimum confidence to display prediction (default: 0.6)")
    args = parser.parse_args()

    # Load label map
    label_map_path = os.path.join(DATA_DIR, "label_map.npy")
    if not os.path.exists(label_map_path):
        print(f"ERROR: label_map.npy not found at {label_map_path}")
        print("Run preprocessing.py first.")
        sys.exit(1)
    label_map = np.load(label_map_path, allow_pickle=True).item()

    # Load model
    model_name = args.model
    model = load_model(model_name)
    print(f"Loaded model: {model_name.upper()}")

    # MediaPipe detector
    detector = make_video_detector()
    print("MediaPipe HandLandmarker ready.")

    # Prediction buffer for stabilization
    pred_buffer = deque(maxlen=BUFFER_SIZE)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Camera {args.camera} opened. Press 'q' to quit.")
    print(f"Keys: 1=SVM  2=RF  3=MLP")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)

        # ── Extract landmarks ────────────────────────────────────────────
        # Run full detection for drawing landmarks on frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)

        # Extract normalized 63-dim features using preprocessing logic
        features = extract_landmarks(frame, detector)

        # ── Predict ──────────────────────────────────────────────────────
        display_letter = None
        display_conf = 0.0

        if features is not None:
            features_2d = features.reshape(1, -1)
            proba = model.predict_proba(features_2d)[0]
            pred_idx = np.argmax(proba)
            confidence = proba[pred_idx]

            if confidence >= args.threshold:
                pred_letter = label_map[pred_idx]
                pred_buffer.append(pred_letter)
            else:
                pred_buffer.append(None)

            # Stabilize: take most common non-None prediction from buffer
            valid_preds = [p for p in pred_buffer if p is not None]
            if valid_preds:
                counter = Counter(valid_preds)
                stable_letter, count = counter.most_common(1)[0]
                # Only show if at least 3 out of BUFFER_SIZE agree
                if count >= 3:
                    display_letter = stable_letter
                    display_conf = confidence
        else:
            pred_buffer.append(None)

        # ── Draw ─────────────────────────────────────────────────────────
        draw_hand_landmarks(frame, result)
        draw_bounding_box(frame, result)
        draw_prediction(frame, display_letter, display_conf, model_name)

        # FPS calculation
        curr_time = time.time()
        fps = 0.8 * fps + 0.2 * (1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time
        draw_fps(frame, fps)
        draw_instructions(frame)

        # ── Display ──────────────────────────────────────────────────────
        cv2.imshow("ASL Letter Recognition - Part 1 Demo", frame)

        # ── Key handling ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            model_name = "svm"
            model = load_model(model_name)
            pred_buffer.clear()
            print(f"Switched to: SVM")
        elif key == ord('2'):
            model_name = "rf"
            model = load_model(model_name)
            pred_buffer.clear()
            print(f"Switched to: Random Forest")
        elif key == ord('3'):
            model_name = "mlp"
            model = load_model(model_name)
            pred_buffer.clear()
            print(f"Switched to: MLP")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Demo ended.")


if __name__ == "__main__":
    main()
