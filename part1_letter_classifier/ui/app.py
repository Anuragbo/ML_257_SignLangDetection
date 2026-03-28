"""
Web UI: upload a still image or send live camera frames (JPEG) and predict the ASL letter (SVM / RF / MLP).

Run from repo root (recommended):
    python part1_letter_classifier/ui/app.py

Or from part1_letter_classifier:
    python ui/app.py

Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

# ── Paths: ui/ → src/ for preprocessing + demo-style inference ───────────────
_UI_DIR = Path(__file__).resolve().parent
_PART1_ROOT = _UI_DIR.parent
_SRC = _PART1_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocessing import ensure_model, extract_landmarks, make_detector  # noqa: E402

DATA_DIR = _PART1_ROOT / "data"
MODELS_DIR = _PART1_ROOT / "models"
LABEL_MAP_PATH = DATA_DIR / "label_map.npy"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_detector = None
_models: dict[str, object] = {}
_label_map: dict[int, str] | None = None


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


def predict_from_bytes(image_bytes: bytes, model_name: str) -> dict:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"ok": False, "error": "Could not read the image. Use JPG, PNG, or WebP."}

    detector = get_detector()
    feats = extract_landmarks(bgr, detector)
    if feats is None:
        return {
            "ok": True,
            "hand_detected": False,
            "letter": None,
            "confidence": None,
            "message": "No hand detected. Try another angle, lighting, or a clearer hand sign.",
        }

    model = get_model(model_name)
    proba = model.predict_proba(feats.reshape(1, -1))[0]
    label_map = get_label_map()
    order = np.argsort(proba)[::-1]
    top = []
    for i in order[:8]:
        letter = label_map[int(i)]
        top.append({"letter": letter, "confidence": float(proba[i])})

    best_idx = int(order[0])
    return {
        "ok": True,
        "hand_detected": True,
        "letter": label_map[best_idx],
        "confidence": float(proba[best_idx]),
        "top_predictions": top,
        "model": model_name,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No file field named 'image'."}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename."}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(
            {"ok": False, "error": f"Unsupported type {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}
        ), 400

    model_name = request.form.get("model", "mlp").lower().strip()
    if model_name not in ("svm", "rf", "mlp"):
        return jsonify({"ok": False, "error": "model must be svm, rf, or mlp"}), 400

    data = f.read()
    if not data:
        return jsonify({"ok": False, "error": "Empty file."}), 400

    try:
        result = predict_from_bytes(data, model_name)
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


def _default_host() -> str:
    if os.environ.get("HOST"):
        return os.environ["HOST"]
    # Inside Docker, bind all interfaces so port publishing works without extra env.
    if os.path.exists("/.dockerenv"):
        return "0.0.0.0"
    return "127.0.0.1"


def main():
    port = int(os.environ.get("PORT", "5000"))
    host = _default_host()
    print(f"ASL letter UI: http://{host}:{port}/")
    app.run(host=host, port=port, debug=os.environ.get("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    main()
