# Sign language detection (ML 257) — Part 1

ASL **letter** classification from static images: MediaPipe hand landmarks, then SVM / Random Forest / MLP / CNN. A **webcam demo** uses the trained sklearn models on your machine.

---

## Local Python vs Docker (read this first)

You do **not** have to install Python, a venv, and `pip install -r requirements.txt` if you do not want to.

| | **Docker** | **Local Python** |
|---|------------|------------------|
| **You install** | Docker Desktop (and Git) | Python 3.10+, Git, then `pip install -r requirements.txt` |
| **Preprocess → train → evaluate** | Yes — everything runs in the container | Yes — `run_pipeline.py` or manual commands |
| **Webcam `demo.py`** | Not practical without extra camera/display setup | Run on the host (recommended) |
| **Jupyter notebooks** | Possible but awkward | Natural on the host |

Docker is **not** “optional” in the sense of being unnecessary. It is **optional** in the sense that it is **one of two valid ways** to run the batch pipeline: either the container brings the dependencies, or your own Python environment does.

If your goal is “run the full pipeline without touching Python on my PC,” use **Docker** and mount your Kaggle credentials so `run_pipeline.py` can download data inside the container (see the Docker section). If you want to hack code, use notebooks, or use the webcam demo easily, use **local Python** (or use Docker for training and Python only for the demo).

---

## What everyone needs

- **This repository** on your machine (`git clone` or download).
- **ASL images** under `part1_letter_classifier/data/asl_dataset/` — either **placed manually** or **downloaded automatically** (see below).

---

## 1. Dataset (automatic download or manual)

### Automatic (recommended for `run_pipeline.py`)

The pipeline can download the Kaggle dataset **[ayuraj/asl-dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)** (same source as the project notebooks) if images are missing.

1. Create a Kaggle API token: **[kaggle.com → Settings → API → Create New Token](https://www.kaggle.com/settings)**.
2. Save `kaggle.json` to:

   - **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`
   - **macOS / Linux:** `~/.kaggle/kaggle.json`

3. Open the [dataset page](https://www.kaggle.com/datasets/ayuraj/asl-dataset) once and click **Download** or **Accept** if Kaggle asks you to agree to the dataset terms.

When you run `python run_pipeline.py`, it will:

- Check for class folders under `part1_letter_classifier/data/asl_dataset/`
- If they are missing, print progress while the **Kaggle CLI** downloads and unzips, then place files in that folder

Use **`python run_pipeline.py --no-download`** if you do not want any network download (the run will fail until you add data yourself).

You can also run only the downloader:

```bash
python dataset_download.py
```

### Manual

Put your own ASL images here:

```text
part1_letter_classifier/data/asl_dataset/
```

**Layout:** one subfolder per class, named with a **single** character (`0`–`9` or `a`–`z`). Inside each folder: `.jpg`, `.jpeg`, or `.png` images.

```text
part1_letter_classifier/data/asl_dataset/
  a/
  b/
  ...
  z/
```

The pipeline needs **at least one** such class folder before preprocessing can run.

---

## 2. Run the full pipeline with Docker (no local Python required)

Install **Docker Desktop** and start it. From the **repo root**:

**Build the image once:**

```bash
docker build -t ml257-signlang:latest .
```

If you **updated the Dockerfile** (or still see old errors like missing `libGLESv2.so.2`), rebuild without cache so Docker does not reuse an older layer:

```bash
docker build --no-cache -t ml257-signlang:latest .
```

The Dockerfile uses the full **`python:3.11-bookworm`** image plus Mesa/GLES packages and runs a quick MediaPipe import during the build so problems show up immediately.

**Run preprocessing, training, and evaluation in one shot** (same as `run_pipeline.py` on the host).

Mount your Kaggle API folder so automatic dataset download works (omit the `-v` line if you already copied `asl_dataset` into `part1_letter_classifier/data/`):

```bash
docker compose --profile pipeline run --rm -v "${HOME}/.kaggle:/root/.kaggle:ro" signlang python /app/run_pipeline.py
```

On **Windows PowerShell** (path to your user profile):

```powershell
docker compose --profile pipeline run --rm -v "${env:USERPROFILE}\.kaggle:/root/.kaggle:ro" signlang python /app/run_pipeline.py
```

Or run each step yourself:

```bash
docker compose --profile pipeline run --rm signlang python /app/part1_letter_classifier/src/preprocessing.py
docker compose --profile pipeline run --rm signlang python /app/part1_letter_classifier/src/train.py --mode all
docker compose --profile pipeline run --rm signlang python /app/part1_letter_classifier/src/evaluate.py
```

`docker-compose.yml` mounts `part1_letter_classifier/data`, `models`, and `results` into the container, so outputs stay on your computer under those folders. The **`signlang`** service uses **`--profile pipeline`** so a plain `docker compose up` does not start it by default.

The **`signlang-ui`** service runs the **Web UI** (Flask: upload + live camera in the browser) on port **5000** — see **Web UI (Docker)** below (`docker compose up signlang-ui`).

Training can take a long time (especially the CNN). The image uses **CPU** PyTorch to avoid huge CUDA downloads.

---

## 3. Run the full pipeline with local Python

Use this path if you prefer not to use Docker, or for day-to-day development.

### 3.1 Virtual environment (recommended)

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 3.3 One command for preprocess → train → evaluate

**Windows:** double-click **`run_all.bat`**, or:

```bat
python run_pipeline.py
```

**macOS / Linux:**

```bash
python3 run_pipeline.py
```

### Skip steps (if you already have outputs)

```bash
python run_pipeline.py --skip-preprocessing
python run_pipeline.py --skip-training
```

---

## 4. Where outputs go

| Output | Location |
|--------|----------|
| Preprocessed arrays (`X.npy`, `y.npy`, `label_map.npy`, test splits) | `part1_letter_classifier/data/` |
| Saved models (`.pkl`, `cnn_best.pt`) | `part1_letter_classifier/models/` |
| Plots (confusion matrices, comparison chart) | `part1_letter_classifier/results/` |
| MediaPipe model (downloaded once) | `part1_letter_classifier/src/hand_landmarker.task` |

---

## 5. Run steps manually (local Python only)

From the **repo root**, with venv activated and `PYTHONPATH` set.

**PowerShell**

```powershell
$env:PYTHONPATH = "part1_letter_classifier/src"
python part1_letter_classifier/src/preprocessing.py
python part1_letter_classifier/src/train.py --mode all
python part1_letter_classifier/src/evaluate.py
```

**Command Prompt**

```bat
set PYTHONPATH=part1_letter_classifier\src
python part1_letter_classifier\src\preprocessing.py
python part1_letter_classifier\src\train.py --mode all
python part1_letter_classifier\src\evaluate.py
```

**macOS / Linux**

```bash
export PYTHONPATH=part1_letter_classifier/src
python part1_letter_classifier/src/preprocessing.py
python part1_letter_classifier/src/train.py --mode all
python part1_letter_classifier/src/evaluate.py
```

---

## 6. Webcam demo (local Python — after training)

Uses your camera and the **SVM / RF / MLP** models in `part1_letter_classifier/models/`.

**Windows**

```bat
cd part1_letter_classifier\src
python demo.py
```

**macOS / Linux**

```bash
cd part1_letter_classifier/src
python3 demo.py
```

**Controls:** `q` quit; `1` SVM, `2` Random Forest, `3` MLP.

### Using Docker

- **Windows / macOS (Docker Desktop):** Passing the **webcam** and a **GUI window** (`cv2.imshow`) into a Linux container is **not practical** for most setups. Use **`docker compose up signlang-ui`** and use **Live camera** or upload stills in the browser, or run **`demo.py` with local Python** (venv) as above.
- **Linux (same machine display):** You can try the **`signlang-demo`** service (needs `/dev/video0`, `DISPLAY`, and the X11 socket so the window can open):

```bash
docker compose --profile demo run --rm -it \
  --device /dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  signlang-demo
```

Optional: `python /app/part1_letter_classifier/src/demo.py --model svm` by overriding `command` in a one-off run, or run `bash` first and invoke `demo.py` manually.

---

## 7. Web UI — upload or live camera (letter prediction)

Small Flask app under **`part1_letter_classifier/ui/`** — upload a still photo or use **Live camera** (browser `getUserMedia`) so frames are sent to the server as JPEGs; the trained **SVM / RF / MLP** models predict the letter (same pipeline as `demo.py`, but the camera runs in the browser, not in the container).

**Live camera + Docker:** The webcam is used by your **browser** on the host; only image bytes are posted to the Flask app. That works with **`docker compose up signlang-ui`** on Windows without passing `/dev/video0` into the container. Use **http://127.0.0.1** or **http://localhost** so the page is treated as a secure context for camera access.

**Layout**

```text
part1_letter_classifier/ui/
  app.py
  templates/index.html
  static/style.css
```

**Run** (from the **repository root**, after `pip install -r requirements.txt` and training so `models/*.pkl` and `data/label_map.npy` exist):

```bash
python part1_letter_classifier/ui/app.py
```

Open **http://127.0.0.1:5000** in your browser. Optional: `set PORT=8080` (Windows) or `export PORT=8080` to change the port.

### Run the Web UI in Docker

Uses the same **`ml257-signlang`** image as the training pipeline. Mounts **`data/`** (for `label_map.npy`) and **`models/`** (for `.pkl` files) from your repo.

**Build** (if you have not already):

```bash
docker build -t ml257-signlang:latest .
```

**Start the UI** (foreground; stop with Ctrl+C):

```bash
docker compose up signlang-ui
```

Detached (runs in the background):

```bash
docker compose up -d signlang-ui
```

Then open **http://127.0.0.1:5000** on the machine running Docker (use **http://localhost:5000** if you prefer).

**Run on another host port** (e.g. 8080):

```bash
UI_PORT=8080 docker compose up signlang-ui
```

**Windows PowerShell:**

```powershell
$env:UI_PORT=8080; docker compose up signlang-ui
```

The app listens on **`0.0.0.0`** inside the container so port mapping works; locally you still use `127.0.0.1` in the browser.

---

## 8. Extract text from a PDF

Default: `documents/project_proposal.pdf`. Requires local Python (or run a one-off container with the same image if you prefer).

```bash
python read_pdf.py
python read_pdf.py "path\to\your\file.pdf"
```

---

## 9. Jupyter notebooks

Open `part1_letter_classifier/notebooks/` in Jupyter on the host. The first cell assumes the notebook’s working directory is the `notebooks` folder.

---

## Quick checklists

**Docker only**

1. Install Docker Desktop.  
2. Add **`~/.kaggle/kaggle.json`** (or pre-fill **`part1_letter_classifier/data/asl_dataset/`** yourself).  
3. `docker build -t ml257-signlang:latest .`  
4. Run `run_pipeline.py` with the **`-v …/.kaggle:/root/.kaggle:ro`** mount shown in section 2 (or skip the mount if data is already on disk).  
5. Check **`part1_letter_classifier/results/`**.

**Local Python**

1. Create venv, activate it.  
2. `pip install -r requirements.txt`  
3. Add **`kaggle.json`** if you want automatic download (section 1), or add images under **`part1_letter_classifier/data/asl_dataset/`** manually.  
4. `python run_pipeline.py`  
5. Check **`part1_letter_classifier/results/`**.

If something fails, read the terminal message: missing dataset folders and missing packages are the most common issues. Training uses **CPU** in Docker by default; local `pip install torch` may install a GPU build depending on your platform, but the code runs on CPU if no GPU is available.
