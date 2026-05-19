# How to Run the ASL Web Application

This guide is only for **running the demo in your browser**. You do not need to train models if weight files are already on your machine (from a teammate or a previous training run).

**What you get:** a local website at `http://127.0.0.1:5000` with live webcam, image/video upload, letter classification, WLASL word recognition, and a fingerspelling decoder (letters → words).

---

## Before you start

| Requirement | Notes |
|---------------|--------|
| **Python 3.10 or newer** | [python.org](https://www.python.org/downloads/) — check with `python --version` |
| **Git** (optional) | To clone the repo |
| **Web browser** | Chrome, Edge, or Firefox recommended |
| **Webcam** | Needed for live camera mode (not needed for upload-only) |
| **Model files** | See [Step 2](#step-2-check-model-files) — not included in git |

Clone or download the project, then open a terminal in the **repository root** (the folder that contains `requirements.txt` and `part1_letter_classifier/`).

---

## Step 1: Install Python packages

### Windows (PowerShell)

```powershell
cd path\to\ML_257_SignLangDetection

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install ultralytics
```

If `Activate.ps1` is blocked, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS / Linux

```bash
cd path/to/ML_257_SignLangDetection

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install ultralytics
```

> `ultralytics` is only required if you use the **YOLO** backend. Installing it up front avoids errors later.

---

## Step 2: Check model files

The app loads trained weights from disk. **If these files are missing, prediction will fail** until you train models or copy them from someone who already ran training.

### Minimum to try letter recognition (MediaPipe)

| File | Location |
|------|----------|
| `label_map.npy` | `part1_letter_classifier/data/label_map.npy` |
| At least one of `svm.pkl`, `rf.pkl`, `mlp.pkl` | `part1_letter_classifier/models/` |

### All backends

| Backend in the UI | Pick in “Model” | Required files |
|-------------------|-----------------|----------------|
| **MediaPipe** | SVM, RF, or MLP | `part1_letter_classifier/data/label_map.npy` + `models/svm.pkl` (or `rf.pkl` / `mlp.pkl`) |
| **PyTorch image** | CNN, MobileNet, ResNet, or VGG | `label_map.npy` + one of `models/cnn_best.pt`, `mobilenet_best.pt`, `resnet18_best.pt`, `vgg11_bn_best.pt` |
| **YOLO** | (none) | `part1_letter_classifier/models/yolo_cls_best.pt` |
| **WLASL** | BiLSTM or Transformer | `part2_word_recognizer/models/bilstm_best.pt` or `transformer_best.pt` |
| | | `part2_word_recognizer/data/sequences/label_map.npy` |

**Don’t have weights?** Ask your project partner for the `models/` folders, or train them using the main [README.md](README.md) (`python run_pipeline.py` for Part 1).

On first run, MediaPipe may download small task files (`hand_landmarker.task`, `holistic_landmarker.task`) automatically.

---

## Step 3: Start the server

With the virtual environment **activated**, from the **repository root**:

```bash
python part1_letter_classifier/ui/app.py
```

You should see something like:

```text
ASL UI: http://127.0.0.1:5000/
```

Leave this terminal window open while you use the app.

### Optional settings

| Variable | Example | Effect |
|----------|---------|--------|
| `PORT` | `8080` | Use a different port |
| `HOST` | `0.0.0.0` | Allow access from other devices on your network |

**Windows (PowerShell):**

```powershell
$env:PORT=8080
python part1_letter_classifier/ui/app.py
```

**macOS / Linux:**

```bash
export PORT=8080
python part1_letter_classifier/ui/app.py
```

---

## Step 4: Open the app in your browser

1. Go to **http://127.0.0.1:5000** (or `http://127.0.0.1:8080` if you changed `PORT`).
2. Use **127.0.0.1** or **localhost** — some browsers block the webcam on other addresses.
3. Allow **camera access** when prompted (for live mode).

Quick health check: **http://127.0.0.1:5000/api/health** should return `{"status":"ok"}`.

---

## Step 5: Use the interface

### Choose backend and model

- **Backend** — MediaPipe, PyTorch image, YOLO, or WLASL.
- **Model** — depends on backend (e.g. SVM vs MLP, or BiLSTM vs Transformer).

### Live camera

1. Click **Start camera**.
2. Sign in front of the camera.

| Backend | What happens | Keyboard |
|---------|----------------|----------|
| **MediaPipe / PyTorch image / YOLO** | Live letter guess updates on screen | Press **Space** once per letter to add it to the **Part 3 decoder** (builds words below) |
| **WLASL** | Holistic skeleton overlay on video | Press **Space** once to record **30 frames** (~1 second), then the app predicts a **word** |

### Upload a file

- **Image** — single-frame prediction for the selected backend.
- **Video** (letter backends only) — decodes the video and runs the Part 3 decoder over frames.

### Part 3 decoder panel

Shows smoothed letters and decoded words when you use letter backends (live **Space** commits or video upload).

---

## Alternative: Run with Docker

Use this if you prefer not to install Python locally. You still need model files in the project folders on your computer.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it.
2. From the repository root:

```bash
docker build -t ml257-signlang:latest .
docker compose up signlang-ui
```

3. Open **http://127.0.0.1:5000** in your browser.

The webcam runs in the browser on your PC; Docker only runs the Flask server. Stop the app with **Ctrl+C** in the terminal.

**Different port (Windows PowerShell):**

```powershell
$env:UI_PORT=8080
docker compose up signlang-ui
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `python` not found | Install Python 3.10+ and use `python3` on Mac/Linux |
| `No module named ...` | Activate the venv and run `pip install -r requirements.txt` again |
| Error about missing `.pkl` or `.pt` | Add model files (Step 2) or switch to a backend you have weights for |
| Camera never starts | Use `http://127.0.0.1:5000`, allow camera in browser settings, close Zoom/Teams/other camera apps |
| YOLO backend fails | Run `pip install ultralytics` and ensure `yolo_cls_best.pt` exists |
| WLASL backend fails | Ensure Part 2 `.pt` files and `part2_word_recognizer/data/sequences/label_map.npy` exist |
| First prediction is slow | Normal — models load on the first request |
| Page won’t load | Confirm the server terminal still shows no crash; check firewall isn’t blocking the port |

---

## Stop the application

- **Local Python:** press **Ctrl+C** in the terminal where `app.py` is running.
- **Docker:** **Ctrl+C**, or `docker compose down` from the repo root.

---

## More documentation

- **Train models, datasets, full pipeline:** [README.md](README.md)
- **Part 3 decoder details:** [part3_decoder/README.md](part3_decoder/README.md)
