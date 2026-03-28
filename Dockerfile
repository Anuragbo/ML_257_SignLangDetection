# CPU image for preprocessing, training, and evaluation.
# Webcam demo (demo.py) needs a local run or extra X11/device passthrough.
#
# Use full bookworm (not slim): MediaPipe’s native libs expect a complete GL/GLES
# stack; slim images often miss transitive deps even when libgles2 is listed.
FROM python:3.11-bookworm

# GLES/EGL + Mesa DRI + GLVND — required so libGLESv2.so.2 and friends resolve at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libglvnd0 \
    libgl1-mesa-dri \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

WORKDIR /app

# CPU wheels avoid multi‑GB CUDA dependencies in the default torch package.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-docker.txt \
    && python -c "from mediapipe.tasks.python import vision as _v; print('mediapipe: import ok')"

COPY . .

ENV PYTHONPATH=/app/part1_letter_classifier/src

# Override with e.g. docker compose run signlang python ...
CMD ["python", "/app/part1_letter_classifier/src/train.py", "--help"]
