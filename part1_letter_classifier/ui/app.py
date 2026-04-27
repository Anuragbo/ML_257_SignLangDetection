"""
Web UI for the ASL project.

Backends exposed in one frontend:
- MediaPipe + sklearn letter models (SVM / RF / MLP)
- PyTorch image letter models (CNN, MobileNetV2, ResNet-18, VGG-11-BN on 64×64 RGB)
- YOLO letter classification
- WLASL word recognition (BiLSTM / Transformer) from live camera frames
- Part 3 fingerspelling decoder (letters → words → sentence) on **live** or **uploaded video** MediaPipe / PyTorch image / YOLO streams

Run from repo root:
    python part1_letter_classifier/ui/app.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
from flask import Flask, jsonify, make_response, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

_UI_DIR = Path(__file__).resolve().parent
_PART1_ROOT = _UI_DIR.parent
_REPO_ROOT = _PART1_ROOT.parent
_SRC = _PART1_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing import ensure_model, extract_landmarks, make_detector  # noqa: E402
from yolo_utils import default_weights_path, load_yolo_model, topk_from_result  # noqa: E402

DATA_DIR = _PART1_ROOT / "data"
MODELS_DIR = _PART1_ROOT / "models"
LABEL_MAP_PATH = DATA_DIR / "label_map.npy"
DEFAULT_YOLO_WEIGHTS = default_weights_path()

PART2_ROOT = _REPO_ROOT / "part2_word_recognizer"
PART2_MODELS_DIR = PART2_ROOT / "models"
PART2_DATA_DIR = PART2_ROOT / "data" / "sequences"
PART2_DEMO_PATH = PART2_ROOT / "src" / "demo.py"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_VIDEO_UPLOAD_BYTES = int(os.environ.get("MAX_VIDEO_UPLOAD_MB", "128")) * 1024 * 1024
VIDEO_MAX_FRAMES_TO_SAMPLE = 300
WLASL_BUFFER_TTL_SEC = 20.0
LETTER_DECODER_BUFFER_MAX = 500
_fingerspell_decoder = None
_letter_decoder_buffers: dict[str, deque] = {}
_letter_decoder_updated: dict[str, float] = {}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = max(MAX_UPLOAD_BYTES, MAX_VIDEO_UPLOAD_BYTES)


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(_e):
    """Werkzeug otherwise returns an HTML 413 page; the UI expects JSON."""
    max_mb = int(app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024))
    return jsonify({
        "ok": False,
        "error": (
            f"Upload too large (this server accepts up to about {max_mb} MB). "
            "Try a shorter video, lower resolution, or set MAX_VIDEO_UPLOAD_MB when starting Flask."
        ),
    }), 413

_detector = None
_models: dict[str, object] = {}
_label_map: dict[int, str] | None = None
_yolo_model = None
_wlasl_module = None
_wlasl_models: dict[str, object] = {}
_wlasl_label_map: dict[int, str] | None = None
_wlasl_detector = None
_wlasl_buffers: dict[str, dict[str, Any]] = {}
_wlasl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_letter_image_models: dict[str, torch.nn.Module] = {}


def load_wlasl_module():
    """Load the Part 2 demo module with a stable custom name."""
    global _wlasl_module
    if _wlasl_module is None:
        spec = importlib.util.spec_from_file_location("wlasl_demo_module", PART2_DEMO_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load Part 2 demo module: {PART2_DEMO_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["wlasl_demo_module"] = module
        spec.loader.exec_module(module)
        _wlasl_module = module
    return _wlasl_module


def get_detector():
    global _detector
    if _detector is None:
        ensure_model()
        _detector = make_detector()
    return _detector


def get_label_map() -> dict[int, str]:
    global _label_map
    if _label_map is None:
        if not LABEL_MAP_PATH.is_file():
            raise FileNotFoundError(
                f"Missing {LABEL_MAP_PATH}. Run preprocessing (or the full pipeline) first."
            )
        _label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    return _label_map


def get_model(name: str):
    if name not in _models:
        path = MODELS_DIR / f"{name}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}. Run train.py first.")
        _models[name] = joblib.load(path)
    return _models[name]


def get_yolo_model(weights_path: Path | None = None):
    global _yolo_model
    path = Path(weights_path) if weights_path is not None else DEFAULT_YOLO_WEIGHTS
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing YOLO weights: {path}. Train with yolo_train.py before using YOLO in the UI."
        )
    if _yolo_model is None:
        _yolo_model = load_yolo_model(path)
    return _yolo_model


def _load_image(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _parse_bool(s: str | None, default: bool = False) -> bool:
    if s is None:
        return default
    v = str(s).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off", ""):
        return False
    return default


def _maybe_mirror_bgr(bgr: np.ndarray, mirror: bool) -> np.ndarray:
    return cv2.flip(bgr, 1) if mirror else bgr


def _mirror_landmark_features(feats: np.ndarray) -> np.ndarray:
    """
    Mirror landmark features across the x-axis (post wrist-centering).
    feats is a flattened (21,3) vector: [x0,y0,z0,x1,y1,z1,...].
    """
    v = np.asarray(feats, dtype=np.float32).reshape(21, 3).copy()
    v[:, 0] *= -1.0
    return v.reshape(-1)

def _flip_landmark_z(feats: np.ndarray) -> np.ndarray:
    """Flip landmark depth (z). Helps with palm/back orientation differences."""
    v = np.asarray(feats, dtype=np.float32).reshape(21, 3).copy()
    v[:, 2] *= -1.0
    return v.reshape(-1)


def _flip_landmark_xz(feats: np.ndarray) -> np.ndarray:
    """Flip both x and z (approx 180° rotation around y-axis in camera coords)."""
    v = np.asarray(feats, dtype=np.float32).reshape(21, 3).copy()
    v[:, 0] *= -1.0
    v[:, 2] *= -1.0
    return v.reshape(-1)


def _best_proba_over_variants(model, feats: np.ndarray, extra_variants: list[np.ndarray] | None = None) -> np.ndarray:
    """
    Evaluate predict_proba over multiple feature variants and return the one with
    the highest top-1 confidence.
    """
    # Inference-time augmentation over landmark symmetries:
    # - x flip handles left-vs-right
    # - z flip handles palm-vs-back (depth sign) differences
    # - x+z approximates a 180° turn around the vertical axis
    variants = [
        feats,
        _mirror_landmark_features(feats),
        _flip_landmark_z(feats),
        _flip_landmark_xz(feats),
    ]
    if extra_variants:
        variants.extend(extra_variants)
    best = None
    best_score = -1.0
    for v in variants:
        p = model.predict_proba(np.asarray(v, dtype=np.float32).reshape(1, -1))[0]
        s = float(np.max(p))
        if s > best_score:
            best_score = s
            best = p
    assert best is not None
    return best


def _extract_overlay_landmarks(image_bgr: np.ndarray, detector) -> list[list[float]] | None:
    """
    Return up to 21 normalized [x, y] points from the first detected hand.
    These are used only for client-side overlay drawing (demo visualization).
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    return [[float(lm.x), float(lm.y)] for lm in result.hand_landmarks[0]]


def predict_mediapipe_from_bytes(image_bytes: bytes, model_name: str, *, mirror: bool = False) -> dict:
    bgr = _load_image(image_bytes)
    if bgr is None:
        return {"ok": False, "error": "Could not read the image. Use JPG, PNG, or WebP."}
    bgr = _maybe_mirror_bgr(bgr, mirror)

    detector = get_detector()
    feats = extract_landmarks(bgr, detector)
    if feats is None:
        return {
            "ok": True,
            "backend": "mediapipe",
            "task": "letter",
            "hand_detected": False,
            "label": None,
            "confidence": None,
            "message": "No hand detected. Try another angle, lighting, or a clearer hand sign.",
        }

    model = get_model(model_name)

    # Extra-robust left-hand support:
    # - Also run MediaPipe on a horizontally flipped image, then score both that feature
    #   vector and its mirror. This avoids relying on MediaPipe handedness and helps when
    #   landmark quality differs between left/right appearances.
    flipped = cv2.flip(bgr, 1)
    feats_flip = extract_landmarks(flipped, detector)
    extra = [feats_flip] if feats_flip is not None else None
    proba = _best_proba_over_variants(model, feats, extra_variants=extra)
    label_map = get_label_map()
    order = np.argsort(proba)[::-1]
    top = [{"label": label_map[int(i)], "confidence": float(proba[i])} for i in order[:8]]
    best_idx = int(order[0])
    overlay_landmarks = _extract_overlay_landmarks(bgr, detector)
    return {
        "ok": True,
        "backend": "mediapipe",
        "task": "letter",
        "hand_detected": True,
        "label": label_map[best_idx],
        "confidence": float(proba[best_idx]),
        "top_predictions": top,
        "model": model_name,
        "landmarks": overlay_landmarks,
    }


def _get_letter_image_model(model_name: str) -> torch.nn.Module:
    """Load a saved PyTorch image classifier for 64×64 RGB letter prediction (cached)."""
    key = model_name.lower().strip()
    allowed = ("cnn", "mobilenet", "resnet", "vgg")
    if key not in allowed:
        raise ValueError(f"model_name must be one of {allowed}")
    if key not in _letter_image_models:
        from train import ASL_CNN, build_mobilenet_v2, build_resnet18, build_vgg11_bn

        label_map = get_label_map()
        n_cls = len(label_map)
        weights_files = {
            "cnn": "cnn_best.pt",
            "mobilenet": "mobilenet_best.pt",
            "resnet": "resnet18_best.pt",
            "vgg": "vgg11_bn_best.pt",
        }
        mode_hint = {
            "cnn": "cnn",
            "mobilenet": "mobilenet",
            "resnet": "resnet",
            "vgg": "vgg",
        }
        path = MODELS_DIR / weights_files[key]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing weights: {path}. Run train.py --mode {mode_hint[key]} or --mode all."
            )
        if key == "cnn":
            net = ASL_CNN(n_cls).to(_wlasl_device)
        elif key == "mobilenet":
            net = build_mobilenet_v2(n_cls, pretrained=False).to(_wlasl_device)
        elif key == "resnet":
            net = build_resnet18(n_cls, pretrained=False).to(_wlasl_device)
        else:
            net = build_vgg11_bn(n_cls, pretrained=False).to(_wlasl_device)
        net.load_state_dict(torch.load(path, map_location=_wlasl_device))
        net.eval()
        _letter_image_models[key] = net
    return _letter_image_models[key]


def predict_image_letter_from_bytes(image_bytes: bytes, model_name: str, *, mirror: bool = False) -> dict:
    """Full-frame 64×64 RGB classifier; no MediaPipe hand crop."""
    from train import IMG_SIZE, predict_torch_images_proba

    bgr = _load_image(image_bytes)
    if bgr is None:
        return {"ok": False, "error": "Could not read the image. Use JPG, PNG, or WebP."}
    bgr = _maybe_mirror_bgr(bgr, mirror)

    key = model_name.lower().strip()
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB).astype(
        np.float32
    ) / 255.0
    x = np.expand_dims(rgb, axis=0)

    try:
        model = _get_letter_image_model(key)
    except FileNotFoundError as e:
        return {
            "ok": False,
            "error": (
                f"{e} "
                "Train with: python part1_letter_classifier/src/train.py --mode transfer "
                "(or --mode mobilenet / resnet / vgg / all)."
            ),
        }

    proba = predict_torch_images_proba(model, x, batch_size=1)[0]
    label_map = get_label_map()
    order = np.argsort(proba)[::-1]
    top = [{"label": label_map[int(i)], "confidence": float(proba[i])} for i in order[:8]]
    best_idx = int(order[0])
    return {
        "ok": True,
        "backend": "image",
        "task": "letter",
        "hand_detected": True,
        "label": label_map[best_idx],
        "confidence": float(proba[best_idx]),
        "top_predictions": top,
        "model": key,
        "message": None,
    }


def predict_yolo_from_bytes(image_bytes: bytes, imgsz: int | None = None, *, mirror: bool = False) -> dict:
    bgr = _load_image(image_bytes)
    if bgr is None:
        return {"ok": False, "error": "Could not read the image. Use JPG, PNG, or WebP."}
    bgr = _maybe_mirror_bgr(bgr, mirror)

    model = get_yolo_model()
    kwargs: dict[str, object] = {"verbose": False}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    results = model.predict(bgr, **kwargs)
    if not results:
        return {
            "ok": True,
            "backend": "yolo",
            "task": "letter",
            "hand_detected": False,
            "label": None,
            "confidence": None,
            "top_predictions": [],
            "model": "yolo_cls",
            "message": "No prediction returned.",
        }

    r = results[0]
    # Normalize to the same JSON shape as the other backends: {label, confidence}
    top_predictions = [
        {"label": row.get("letter"), "confidence": float(row.get("confidence", 0.0))}
        for row in (topk_from_result(r, k=8) or [])
    ]
    if r.probs is None:
        return {
            "ok": True,
            "backend": "yolo",
            "task": "letter",
            "hand_detected": False,
            "label": None,
            "confidence": None,
            "top_predictions": top_predictions,
            "model": "yolo_cls",
            "message": "Low confidence. Try cleaner framing or center the hand.",
        }

    names = r.names or {}
    top1_idx = int(r.probs.top1)
    conf = float(r.probs.top1conf)
    label = str(names.get(top1_idx, top1_idx))
    return {
        "ok": True,
        "backend": "yolo",
        "task": "letter",
        "hand_detected": conf >= 0.15,
        "label": label if conf >= 0.15 else None,
        "confidence": conf if conf >= 0.15 else None,
        "top_predictions": top_predictions,
        "model": "yolo_cls",
        "message": None if conf >= 0.15 else "Low confidence. Try better lighting or a tighter crop of the hand.",
    }


def get_wlasl_label_map() -> dict[int, str]:
    global _wlasl_label_map
    if _wlasl_label_map is None:
        module = load_wlasl_module()
        if not PART2_DATA_DIR.is_dir():
            raise FileNotFoundError(f"Missing WLASL sequence data: {PART2_DATA_DIR}")
        _wlasl_label_map = module.load_label_map(str(PART2_DATA_DIR))
    return _wlasl_label_map


def get_wlasl_detector():
    global _wlasl_detector
    if _wlasl_detector is None:
        module = load_wlasl_module()
        _wlasl_detector = module.make_detector()
    return _wlasl_detector


def get_wlasl_model(model_name: str):
    if model_name not in ("bilstm", "transformer"):
        raise FileNotFoundError("WLASL model must be bilstm or transformer.")
    if model_name not in _wlasl_models:
        module = load_wlasl_module()
        path = PART2_MODELS_DIR / f"{model_name}_best.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing WLASL checkpoint: {path}. Train the Part 2 model first."
            )
        _wlasl_models[model_name] = module.load_model(str(PART2_MODELS_DIR), model_name, _wlasl_device)
    return _wlasl_models[model_name]


def _get_wlasl_state(client_id: str):
    now = time.time()
    stale = [
        cid for cid, state in _wlasl_buffers.items()
        if now - state["updated_at"] > WLASL_BUFFER_TTL_SEC
    ]
    for cid in stale:
        _wlasl_buffers.pop(cid, None)

    state = _wlasl_buffers.get(client_id)
    if state is None:
        module = load_wlasl_module()
        state = {"buffer": deque(maxlen=module.SEQ_LEN), "updated_at": now}
        _wlasl_buffers[client_id] = state
    state["updated_at"] = now
    return state


def reset_wlasl_state(client_id: str) -> None:
    if client_id in _wlasl_buffers:
        _wlasl_buffers.pop(client_id, None)


def reset_letter_decoder_state(client_id: str) -> None:
    """Clear Part 3 fingerspelling decoder history for this browser session."""
    _letter_decoder_buffers.pop(client_id, None)
    _letter_decoder_updated.pop(client_id, None)


def _prune_stale_letter_decoder_buffers() -> None:
    now = time.time()
    stale = [
        cid
        for cid, t in _letter_decoder_updated.items()
        if now - t > WLASL_BUFFER_TTL_SEC
    ]
    for cid in stale:
        _letter_decoder_buffers.pop(cid, None)
        _letter_decoder_updated.pop(cid, None)


def get_fingerspell_decoder():
    """Lazy singleton — Part 3 is optional at import time."""
    global _fingerspell_decoder
    if _fingerspell_decoder is None:
        from part3_decoder import FingerspellDecoder

        _fingerspell_decoder = FingerspellDecoder()
    return _fingerspell_decoder


def append_live_letter_frame(client_id: str, label: str | None, confidence: float | None) -> dict[str, Any]:
    """
    Buffer one live frame for letter backends and run the Part 3 decoder over the stream.

    Returns a JSON-serializable dict for the ``decoder`` field, or ``{"error": ...}``.
    """
    from part3_decoder.frame_utils import frame_from_top1

    _prune_stale_letter_decoder_buffers()
    buf = _letter_decoder_buffers.get(client_id)
    if buf is None:
        buf = deque(maxlen=LETTER_DECODER_BUFFER_MAX)
        _letter_decoder_buffers[client_id] = buf

    conf = float(confidence) if confidence is not None else 0.0
    if not label:
        fp = frame_from_top1(None, min(conf, 0.2))
    else:
        s = str(label).strip()
        ch = s[0] if s else None
        if ch and (ch.isalpha() or ch.isdigit()):
            fp = frame_from_top1(ch, conf)
        else:
            fp = frame_from_top1(None, conf)

    buf.append(fp)
    _letter_decoder_updated[client_id] = time.time()

    try:
        dec = get_fingerspell_decoder()
        pr = dec.decode_frames(list(buf))
    except Exception as e:
        return {"error": str(e)}

    return _pipeline_result_to_decoder_dict(pr, frames_count=len(buf))


def _pipeline_result_to_decoder_dict(pr: Any, frames_count: int | None = None) -> dict[str, Any]:
    """Serialize Part 3 ``PipelineResult`` for JSON (shared by live buffer and video)."""
    out = {
        "smoothed_letters": pr.letters_normalized,
        "cleaned_letters": pr.cleaned_letters_no_spaces,
        "sentence": pr.sentence,
        "word_fragments": pr.word_fragments,
        "beam_top": [
            {"words": h.words, "score": round(float(h.score), 4)} for h in pr.beam_hypotheses[:5]
        ],
    }
    if frames_count is not None:
        out["frames_buffered"] = frames_count
    return out


def _bgr_to_jpeg_bytes(bgr: np.ndarray, quality: int = 88) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return buf.tobytes()


def _predict_letter_jpeg_bytes(jpeg_bytes: bytes, backend: str, model_name: str) -> dict:
    """Run the selected letter backend on one JPEG-encoded frame."""
    if backend == "mediapipe":
        return predict_mediapipe_from_bytes(jpeg_bytes, model_name)
    if backend == "image":
        return predict_image_letter_from_bytes(jpeg_bytes, model_name)
    if backend == "yolo":
        return predict_yolo_from_bytes(jpeg_bytes)
    return {"ok": False, "error": f"Unknown letter backend: {backend}"}


def predict_video_from_bytes(
    video_bytes: bytes,
    original_filename: str,
    backend: str,
    model_name: str,
    *,
    mirror: bool = False,
) -> dict:
    """
    Sample frames from an uploaded video, run letter classification on each sample,
    decode the full stream with Part 3, and return the last frame's prediction plus decoder output.
    """
    from part3_decoder.frame_utils import frame_from_top1

    suffix = Path(original_filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        suffix = ".mp4"
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        Path(tmp_path).write_bytes(video_bytes)
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"ok": False, "error": "Could not open video (unsupported or corrupt file)."}
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        frame_count_guess = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_read = 0
        sampled = 0
        last_result: dict | None = None
        frame_preds: list[Any] = []

        # First pass: count frames if needed (some codecs report 0)
        if frame_count_guess <= 0:
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_read += 1
            cap.release()
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return {"ok": False, "error": "Could not rewind video."}
            n_total = total_read
            total_read = 0
        else:
            n_total = frame_count_guess

        step = max(1, (n_total + VIDEO_MAX_FRAMES_TO_SAMPLE - 1) // VIDEO_MAX_FRAMES_TO_SAMPLE)

        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if total_read % step == 0:
                if mirror:
                    bgr = cv2.flip(bgr, 1)
                jpeg = _bgr_to_jpeg_bytes(bgr)
                if jpeg:
                    one = _predict_letter_jpeg_bytes(jpeg, backend, model_name)
                    last_result = one
                    sampled += 1
                    conf = float(one["confidence"]) if one.get("confidence") is not None else 0.0
                    lab = one.get("label")
                    if not lab:
                        frame_preds.append(frame_from_top1(None, min(conf, 0.2)))
                    else:
                        s = str(lab).strip()
                        ch = s[0] if s else None
                        if ch and (ch.isalpha() or ch.isdigit()):
                            frame_preds.append(frame_from_top1(ch, conf))
                        else:
                            frame_preds.append(frame_from_top1(None, conf))
            total_read += 1
        cap.release()

        if sampled == 0 or last_result is None:
            return {
                "ok": False,
                "error": "No frames could be read from the video.",
            }

        decoder_payload: dict[str, Any] | None = None
        try:
            dec = get_fingerspell_decoder()
            pr = dec.decode_frames(frame_preds, video_mode=True)
            decoder_payload = _pipeline_result_to_decoder_dict(pr, frames_count=len(frame_preds))
        except Exception as e:
            decoder_payload = {"error": str(e)}

        duration = total_read / fps if fps > 0 else None
        out = {
            **last_result,
            "ok": last_result.get("ok", True),
            "task": "video",
            "stream_mode": "video",
            "message": (
                f"Video: {total_read} frames read, {sampled} sampled for the model"
                + (f", ~{duration:.1f}s." if duration is not None else ".")
            ),
            "video": {
                "filename": original_filename,
                "frames_read": total_read,
                "frames_sampled": sampled,
                "frame_step": step,
                "approx_fps": round(fps, 3),
                "approx_duration_sec": round(duration, 2) if duration is not None else None,
            },
        }
        if decoder_payload is not None:
            out["decoder"] = decoder_payload
        return out
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def predict_wlasl_live(image_bytes: bytes, model_name: str, client_id: str, *, mirror: bool = False) -> dict:
    bgr = _load_image(image_bytes)
    if bgr is None:
        return {"ok": False, "error": "Could not read the image. Use JPG, PNG, or WebP."}
    bgr = _maybe_mirror_bgr(bgr, mirror)

    module = load_wlasl_module()
    detector = get_wlasl_detector()
    model = get_wlasl_model(model_name)
    label_map = get_wlasl_label_map()
    state = _get_wlasl_state(client_id)

    features, result = module.extract_frame_features(bgr, detector)
    has_signal = bool(result.left_hand_landmarks or result.right_hand_landmarks)
    state["buffer"].append(features)
    n = len(state["buffer"])

    if n < module.SEQ_LEN:
        return {
            "ok": True,
            "backend": "wlasl",
            "task": "word",
            "hand_detected": has_signal,
            "label": None,
            "confidence": None,
            "top_predictions": [],
            "model": model_name,
            "message": f"Collecting frames for WLASL word model: {n}/{module.SEQ_LEN}.",
            "progress": {"current": n, "required": module.SEQ_LEN},
        }

    preds = module.infer(model, list(state["buffer"]), label_map, _wlasl_device, top_k=3)
    top_predictions = [{"label": word, "confidence": float(conf)} for word, conf in preds]
    label = top_predictions[0]["label"] if top_predictions else None
    conf = top_predictions[0]["confidence"] if top_predictions else None
    return {
        "ok": True,
        "backend": "wlasl",
        "task": "word",
        "hand_detected": has_signal,
        "label": label,
        "confidence": conf,
        "top_predictions": top_predictions,
        "model": model_name,
        "message": None if top_predictions else "No word prediction available.",
        "progress": {"current": module.SEQ_LEN, "required": module.SEQ_LEN},
    }


@app.route("/")
def index():
    # Avoid stale HTML in the browser after git pull / redeploy (model list lives in JS).
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/image-checkpoints")
def image_checkpoints():
    """Which PyTorch image letter checkpoints exist under ``models/`` (for UI hints)."""
    keys = {
        "cnn": "cnn_best.pt",
        "mobilenet": "mobilenet_best.pt",
        "resnet": "resnet18_best.pt",
        "vgg": "vgg11_bn_best.pt",
    }
    return jsonify({name: (MODELS_DIR / fname).is_file() for name, fname in keys.items()})


@app.route("/api/reset-wlasl", methods=["POST"])
def api_reset_wlasl():
    client_id = request.form.get("client_id", "").strip()
    if client_id:
        reset_wlasl_state(client_id)
        reset_letter_decoder_state(client_id)
    return jsonify({"ok": True})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No file field named 'image'."}), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename."}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"ok": False, "error": f"Unsupported type {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400

    backend = request.form.get("backend", "mediapipe").lower().strip()
    model_name = request.form.get("model", "mlp").lower().strip()
    client_id = request.form.get("client_id", "").strip() or request.remote_addr or "anonymous"
    stream_mode = request.form.get("stream_mode", "image").lower().strip()
    mirror = _parse_bool(request.form.get("mirror"), default=False)

    data = f.read()
    if not data:
        return jsonify({"ok": False, "error": "Empty file."}), 400

    try:
        if backend == "mediapipe":
            if model_name not in ("svm", "rf", "mlp"):
                return jsonify({"ok": False, "error": "MediaPipe model must be svm, rf, or mlp."}), 400
            result = predict_mediapipe_from_bytes(data, model_name, mirror=mirror)
        elif backend == "image":
            if model_name not in ("cnn", "mobilenet", "resnet", "vgg"):
                return jsonify({
                    "ok": False,
                    "error": "Image backend model must be cnn, mobilenet, resnet, or vgg.",
                }), 400
            result = predict_image_letter_from_bytes(data, model_name, mirror=mirror)
        elif backend == "yolo":
            result = predict_yolo_from_bytes(data, mirror=mirror)
        elif backend == "wlasl":
            if model_name not in ("bilstm", "transformer"):
                return jsonify({"ok": False, "error": "WLASL model must be bilstm or transformer."}), 400
            if stream_mode != "live":
                result = {
                    "ok": True,
                    "backend": "wlasl",
                    "task": "word",
                    "hand_detected": False,
                    "label": None,
                    "confidence": None,
                    "top_predictions": [],
                    "model": model_name,
                    "message": "The WLASL word model needs a live sequence of frames. Use Start camera instead of single-image upload.",
                }
            else:
                result = predict_wlasl_live(data, model_name, client_id, mirror=mirror)
        else:
            return jsonify({"ok": False, "error": "backend must be mediapipe, image, yolo, or wlasl."}), 400
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 503

    if (
        stream_mode == "live"
        and backend in ("mediapipe", "image", "yolo")
        and result.get("ok")
    ):
        result["decoder"] = append_live_letter_frame(
            client_id, result.get("label"), result.get("confidence")
        )

    return jsonify(result), 200 if result.get("ok") else 400


@app.route("/api/predict-video", methods=["POST"])
def api_predict_video():
    """Process an uploaded video: sample frames, letter-classify each, run Part 3 over the stream."""
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "No file field named 'video'."}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename."}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            "ok": False,
            "error": f"Unsupported video type {ext}. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        }), 400

    backend = request.form.get("backend", "mediapipe").lower().strip()
    model_name = request.form.get("model", "mlp").lower().strip()
    mirror = _parse_bool(request.form.get("mirror"), default=False)

    if backend == "wlasl":
        return jsonify({
            "ok": False,
            "error": (
                "Video upload is only supported for letter backends (MediaPipe, PyTorch image, YOLO). "
                "Use the live camera for WLASL."
            ),
        }), 400

    data = f.read()
    if not data:
        return jsonify({"ok": False, "error": "Empty file."}), 400
    if len(data) > MAX_VIDEO_UPLOAD_BYTES:
        return jsonify({
            "ok": False,
            "error": f"Video too large (max {MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)} MB).",
        }), 400

    try:
        if backend == "mediapipe":
            if model_name not in ("svm", "rf", "mlp"):
                return jsonify({"ok": False, "error": "MediaPipe model must be svm, rf, or mlp."}), 400
        elif backend == "image":
            if model_name not in ("cnn", "mobilenet", "resnet", "vgg"):
                return jsonify({
                    "ok": False,
                    "error": "Image backend model must be cnn, mobilenet, resnet, or vgg.",
                }), 400
        elif backend != "yolo":
            return jsonify({"ok": False, "error": "backend must be mediapipe, image, or yolo for video."}), 400

        result = predict_video_from_bytes(data, f.filename, backend, model_name, mirror=mirror)
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    except Exception as e:
        app.logger.exception("predict-video failed")
        return jsonify({"ok": False, "error": f"Video processing failed: {e!s}"}), 500

    return jsonify(result), 200 if result.get("ok") else 400


def _default_host() -> str:
    if os.environ.get("HOST"):
        return os.environ["HOST"]
    if os.path.exists("/.dockerenv"):
        return "0.0.0.0"
    return "127.0.0.1"


def main():
    port = int(os.environ.get("PORT", "5000"))
    host = _default_host()
    print(f"ASL UI: http://{host}:{port}/")
    # threaded=True: live camera posts overlap safely; first request can be slow (MediaPipe + decode).
    app.run(
        host=host,
        port=port,
        debug=os.environ.get("FLASK_DEBUG") == "1",
        threaded=True,
    )


if __name__ == "__main__":
    main()
