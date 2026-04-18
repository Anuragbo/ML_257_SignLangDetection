"""
demo.py
-------
Real-time webcam ASL word recognition using trained BiLSTM or Transformer.

State machine:
    IDLE       -> rolling 30-frame buffer, dim live guess every 10 frames
    RECORDING  -> fresh 30-frame collection with red REC indicator + progress bar
    INFERRING  -> single forward pass
    DISPLAYING -> top-3 predictions with confidence bars (3 seconds) -> IDLE

Controls:
    SPACE  - capture (start fresh 30-frame recording)
    l      - switch to BiLSTM model
    t      - switch to Transformer model
    c      - clear current prediction
    q      - quit

Usage:
    python demo.py
    python demo.py --model transformer
    python demo.py --model bilstm --models_dir ../models --camera 0
"""

import os
import sys
import argparse
import time
import collections
import numpy as np
import cv2

import torch
import torch.nn.functional as F

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

sys.path.insert(0, os.path.dirname(__file__))
from train import BiLSTMClassifier, TransformerClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_MODELS    = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_DATA      = os.path.join(os.path.dirname(__file__), "..", "data", "sequences")
HOLISTIC_MODEL_PATH = os.path.join(os.path.dirname(__file__), "holistic_landmarker.task")

# ── Feature dims (must match preprocessing.py) ────────────────────────────────
DIM_HAND  = 63
DIM_POSE  = 99
DIM_TOTAL = DIM_HAND * 2 + DIM_POSE   # 225
SEQ_LEN   = 30

# ── States ────────────────────────────────────────────────────────────────────
IDLE       = "IDLE"
RECORDING  = "RECORDING"
INFERRING  = "INFERRING"
DISPLAYING = "DISPLAYING"

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,   0,   0)
C_RED    = (0,   0,   220)
C_GREEN  = (0,   200, 0)
C_CYAN   = (255, 200, 0)
C_MAGENTA= (200, 0,   200)
C_GRAY   = (120, 120, 120)
C_DARK   = (30,  30,  30)
C_BLUE   = (200, 100, 0)


# ── Normalization (same as preprocessing.py) ──────────────────────────────────

def normalize_hand(landmarks_list):
    if not landmarks_list:
        return np.zeros(DIM_HAND, dtype=np.float32)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list], dtype=np.float32)
    coords -= coords[0]
    scale = np.linalg.norm(coords[9])
    if scale > 1e-6:
        coords /= scale
    return coords.flatten()


def normalize_pose(landmarks_list):
    if not landmarks_list:
        return np.zeros(DIM_POSE, dtype=np.float32)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list], dtype=np.float32)
    hip_mid      = (coords[23] + coords[24]) / 2.0
    shoulder_mid = (coords[11] + coords[12]) / 2.0
    coords      -= hip_mid
    torso_height = np.linalg.norm(shoulder_mid - hip_mid)
    if torso_height > 1e-6:
        coords /= torso_height
    return coords.flatten()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(models_dir, model_type, device):
    fname = f"{model_type}_best.pt"
    path  = os.path.join(models_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    if "hidden_dim" in cfg:
        model = BiLSTMClassifier(**cfg)
    else:
        model = TransformerClassifier(**cfg)

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded {fname}  val_top1={ckpt['metrics'].get('val_top1', '?'):.3f}")
    return model


def load_label_map(data_dir):
    path = os.path.join(data_dir, "label_map.npy")
    return np.load(path, allow_pickle=True).item()


# ── MediaPipe detector ────────────────────────────────────────────────────────

def make_detector():
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


def extract_frame_features(frame_bgr, detector):
    """Run holistic detection on one frame, return (225,) feature vector."""
    frame_bgr = cv2.resize(frame_bgr, (640, 480))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result    = detector.detect(mp_image)

    lh   = normalize_hand(result.left_hand_landmarks)
    rh   = normalize_hand(result.right_hand_landmarks)
    pose = normalize_pose(result.pose_landmarks)

    return np.concatenate([lh, rh, pose]), result


# ── Inference ─────────────────────────────────────────────────────────────────

def infer(model, sequence, label_map, device, top_k=3):
    """
    sequence: list of (225,) arrays, length SEQ_LEN
    Returns list of (word, confidence) sorted by confidence desc.
    """
    x = torch.tensor(np.stack(sequence), dtype=torch.float32)
    x = x.unsqueeze(0).to(device)   # (1, 30, 225)

    with torch.no_grad():
        logits = model(x)            # (1, num_classes)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = probs.argsort()[::-1][:top_k]
    return [(label_map[int(i)], float(probs[i])) for i in top_idx]


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_landmarks(frame, result):
    """Draw hand and pose landmarks on frame."""
    h, w = frame.shape[:2]

    def pt(lm):
        return int(lm.x * w), int(lm.y * h)

    # Left hand — cyan
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks:
            cv2.circle(frame, pt(lm), 3, C_CYAN, -1)

    # Right hand — magenta
    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks:
            cv2.circle(frame, pt(lm), 3, C_MAGENTA, -1)

    # Pose — white dots on key joints only (shoulders, elbows, wrists, hips)
    KEY_POSE = [11, 12, 13, 14, 15, 16, 23, 24]
    if result.pose_landmarks:
        for i in KEY_POSE:
            if i < len(result.pose_landmarks):
                cv2.circle(frame, pt(result.pose_landmarks[i]), 5, C_WHITE, -1)


def draw_hud(frame, state, model_name, buffer_len, predictions, display_timer):
    """Draw the heads-up display overlay."""
    h, w = frame.shape[:2]

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 55), C_DARK, -1)

    # Model name
    cv2.putText(frame, f"Model: {model_name.upper()}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WHITE, 2)

    # Controls hint
    hint = "SPACE=capture  l=BiLSTM  t=Transformer  c=clear  q=quit"
    cv2.putText(frame, hint, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)

    # State-specific HUD
    if state == IDLE:
        cv2.putText(frame, "IDLE — press SPACE to sign",
                    (w // 2 - 160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_GRAY, 2)

    elif state == RECORDING:
        # Red REC badge
        cv2.circle(frame, (w - 30, 30), 10, C_RED, -1)
        cv2.putText(frame, "REC", (w - 70, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_RED, 2)
        # Progress bar
        bar_w = 300
        bar_x = w // 2 - bar_w // 2
        bar_y = h - 45
        filled = int(bar_w * buffer_len / SEQ_LEN)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), C_GRAY, 2)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 18), C_RED, -1)
        cv2.putText(frame, f"{buffer_len}/{SEQ_LEN}",
                    (bar_x + bar_w + 8, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1)

    elif state == INFERRING:
        cv2.putText(frame, "Recognizing...",
                    (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_GREEN, 2)

    elif state == DISPLAYING and predictions:
        # Prediction panel (bottom area)
        panel_y = h - 160
        cv2.rectangle(frame, (0, panel_y - 10), (w, h - 35), C_DARK, -1)

        bar_max_w = 250
        for i, (word, conf) in enumerate(predictions):
            y = panel_y + i * 38
            # Rank color
            color = [C_GREEN, (0, 180, 255), C_GRAY][i]
            rank  = ["#1", "#2", "#3"][i]
            cv2.putText(frame, f"{rank} {word.upper()}",
                        (15, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75 if i == 0 else 0.6, color, 2 if i == 0 else 1)
            # Confidence bar
            bar_w = int(bar_max_w * conf)
            cv2.rectangle(frame, (210, y + 4), (210 + bar_max_w, y + 22), C_GRAY, 1)
            cv2.rectangle(frame, (210, y + 4), (210 + bar_w, y + 22), color, -1)
            cv2.putText(frame, f"{conf*100:.1f}%",
                        (210 + bar_max_w + 8, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1)

        # Countdown timer
        remaining = max(0, 3.0 - (time.time() - display_timer))
        cv2.putText(frame, f"Clearing in {remaining:.1f}s",
                    (w - 180, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1)

    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time ASL word demo")
    parser.add_argument("--model",      choices=["bilstm", "transformer"],
                        default="transformer")
    parser.add_argument("--models_dir", default=DEFAULT_MODELS)
    parser.add_argument("--data_dir",   default=DEFAULT_DATA)
    parser.add_argument("--camera",     type=int, default=0)
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(args.data_dir)

    # Load both models upfront so switching is instant
    models = {}
    for mtype in ["bilstm", "transformer"]:
        try:
            models[mtype] = load_model(args.models_dir, mtype, device)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not models:
        print("No models found. Check --models_dir path.")
        return

    active_model = args.model if args.model in models else list(models.keys())[0]
    print(f"Active model: {active_model}")

    # MediaPipe detector
    detector = make_detector()

    # Webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        detector.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State
    state         = IDLE
    rolling_buf   = collections.deque(maxlen=SEQ_LEN)  # continuous rolling buffer
    record_buf    = []                                  # fresh buffer for RECORDING
    predictions   = []
    display_timer = 0.0
    frame_count   = 0
    live_guess    = ""

    print("\nDemo running. Press SPACE to capture a sign.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)   # mirror

        # Extract features from current frame
        features, mp_result = extract_frame_features(frame, detector)
        draw_landmarks(frame, mp_result)

        # ── State machine ──────────────────────────────────────────────────
        if state == IDLE:
            rolling_buf.append(features)
            frame_count += 1

            # Live guess every 10 frames when buffer is full
            if frame_count % 10 == 0 and len(rolling_buf) == SEQ_LEN and active_model in models:
                preds = infer(models[active_model], list(rolling_buf),
                              label_map, device, top_k=1)
                live_guess = preds[0][0] if preds else ""

            if live_guess:
                cv2.putText(frame, f"live: {live_guess}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GRAY, 1)

        elif state == RECORDING:
            record_buf.append(features)
            if len(record_buf) >= SEQ_LEN:
                state = INFERRING

        elif state == INFERRING:
            if active_model in models:
                predictions = infer(models[active_model], record_buf[:SEQ_LEN],
                                    label_map, device, top_k=3)
            else:
                predictions = []
            display_timer = time.time()
            state = DISPLAYING

        elif state == DISPLAYING:
            if time.time() - display_timer > 3.0:
                state      = IDLE
                predictions = []
                live_guess  = ""

        # ── Draw HUD ───────────────────────────────────────────────────────
        buf_len = len(record_buf) if state == RECORDING else 0
        frame = draw_hud(frame, state, active_model, buf_len,
                         predictions, display_timer)

        cv2.imshow("ASL Word Recognizer", frame)

        # ── Key handling ───────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            record_buf = []
            state      = RECORDING
            predictions = []
            live_guess  = ""
        elif key == ord("l") and "bilstm" in models:
            active_model = "bilstm"
            print("Switched to BiLSTM")
        elif key == ord("t") and "transformer" in models:
            active_model = "transformer"
            print("Switched to Transformer")
        elif key == ord("c"):
            state       = IDLE
            predictions = []
            live_guess  = ""
            record_buf  = []

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()
