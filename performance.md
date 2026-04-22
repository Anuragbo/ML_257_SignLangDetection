# Model performance (reference)

Offline evaluation metrics for **Part 1** (ASL letter classifiers) and **Part 2** (WLASL word recognizers). **Refresh** after training or data changes using [Reproduce](#reproduce).

**Last full evaluation run:** 2026-04-19 (`evaluate.py`, `yolo_evaluate.py`, Part 2 `evaluate.py`).

---

## Part 1 — ASL fingerspelling (letters)

### MediaPipe landmarks + classical models (test split: `X_test.npy` / `y_test.npy`)

| Model | Accuracy | Macro P | Macro R | Macro F1 |
|-------|----------|---------|---------|----------|
| Random Forest | 0.9066 | 0.9128 | 0.9039 | 0.9027 |
| SVM | 0.9011 | 0.9038 | 0.9005 | 0.8977 |
| MLP | 0.8242 | 0.8264 | 0.8211 | 0.8173 |

**Test samples:** 364 (landmark test split).

**Outputs:** `part1_letter_classifier/results/confusion_*.png`, `model_comparison.png`.

#### Chart — MediaPipe model accuracy

```text
Scale: 0.00 -> 1.00 (40 blocks)
🔵 Random Forest  0.9066 |████████████████████████████████████░░░░|
🟢 SVM            0.9011 |████████████████████████████████████░░░░|
🟠 MLP            0.8242 |█████████████████████████████████░░░░░░░|
```
**Key:** 🔵 Random Forest, 🟢 SVM, 🟠 MLP.  
**Values:** Random Forest `0.9066`, SVM `0.9011`, MLP `0.8242`.

### PyTorch image models (test split: `X_test_img.npy` / `y_test_img.npy`)

| Model | Accuracy | Macro P | Macro R | Macro F1 |
|-------|----------|---------|---------|----------|
| ResNet-18 | 0.9894 | 0.9905 | 0.9894 | 0.9892 |
| MobileNetV2 | 0.9868 | 0.9878 | 0.9869 | 0.9867 |
| VGG-11-BN | 0.9815 | 0.9841 | 0.9816 | 0.9816 |
| CNN (`ASL_CNN`) | 0.9392 | 0.9451 | 0.9402 | 0.9384 |

**Test samples:** 378 (image test split).

**FINAL RANKING (by macro F1, all Part 1 models in `evaluate.py`):** ResNet-18 → MobileNetV2 → VGG-11-BN → CNN → Random Forest → SVM → MLP.

**Checkpoints:** `cnn_best.pt`, `mobilenet_best.pt`, `resnet18_best.pt`, `vgg11_bn_best.pt` under `part1_letter_classifier/models/`.

**Outputs:** `confusion_cnn.png`, `confusion_mobilenetv2.png`, `confusion_resnet-18.png`, `confusion_vgg-11-bn.png`, `model_comparison.png`.

#### Chart — PyTorch image model accuracy

```text
Scale: 0.00 -> 1.00 (40 blocks)
🔵 ResNet-18      0.9894 |████████████████████████████████████████|
🟢 MobileNetV2    0.9868 |███████████████████████████████████████░|
🟠 VGG-11-BN      0.9815 |███████████████████████████████████████░|
🔴 CNN            0.9392 |██████████████████████████████████████░░|
```
**Key:** 🔵 ResNet-18, 🟢 MobileNetV2, 🟠 VGG-11-BN, 🔴 CNN.  
**Values:** ResNet-18 `0.9894`, MobileNetV2 `0.9868`, VGG-11-BN `0.9815`, CNN `0.9392`.

### YOLO classification (Ultralytics — validation split `data/yolo_cls_dataset/val/`)

| Samples | Accuracy | Macro precision | Macro recall | Macro F1 |
|---------|----------|-----------------|--------------|----------|
| 503 | 0.9622 | 0.9667 | 0.9623 | 0.9617 |

**Also saved in:** `part1_letter_classifier/results/yolo_cls_metrics.txt`, `yolo_cls_confusion.png`.

#### Chart — YOLO vs best image model

```text
Scale: 0.00 -> 1.00 (40 blocks)
🟦 YOLO           0.9622 |███████████████████████████████████████░|
🔵 ResNet-18      0.9894 |████████████████████████████████████████|
```
**Key:** 🟦 YOLO, 🔵 ResNet-18.  
**Values:** YOLO `0.9622`, ResNet-18 `0.9894`.

### Comparing Part 1 numbers

- Landmark models use the **landmark test** arrays.
- PyTorch image models use the **image test** arrays (378 samples; different split than landmarks).
- YOLO metrics are on the **YOLO val** split — not identical to the image test split above.

---

## Part 2 — WLASL words (100 classes)

Test set: **61** sequences (`part2_word_recognizer/data/sequences` test split).

| Model | Test top-1 | Test top-5 | Best val top-1 (checkpoint) |
|-------|------------|------------|------------------------------|
| Transformer | 34.4% | 70.5% | 41.0% (epoch 32) |
| BiLSTM | 24.6% | 59.0% | 28.0% (epoch 42) |

**Outputs:** `part2_word_recognizer/results/confusion_bilstm.png`, `confusion_transformer.png`, `per_class_accuracy.csv`, `model_comparison.png`.

#### Chart — Part 2 word recognition

```text
Scale: 0.00 -> 1.00 (40 blocks)
🟠 Transformer top-1  0.344 |██████████████░░░░░░░░░░░░░░░░░░░░░░░░|
🟠 BiLSTM top-1       0.246 |██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
🟦 Transformer top-5  0.705 |████████████████████████████░░░░░░░░░░░|
🟦 BiLSTM top-5       0.590 |████████████████████████░░░░░░░░░░░░░░░|
```
**Key:** 🟠 Top-1, 🟦 Top-5.  
**Model key:** Transformer first row pair, BiLSTM second row pair.  
**Values:** Transformer top-1 `0.344`, top-5 `0.705`; BiLSTM top-1 `0.246`, top-5 `0.590`.

---

## Reproduce

From the repository root (PowerShell):

```powershell
cd <path-to-repo>
$env:PYTHONPATH = (Resolve-Path "part1_letter_classifier\src").Path

python part1_letter_classifier\src\evaluate.py
python part1_letter_classifier\src\yolo_evaluate.py
python part2_word_recognizer\src\evaluate.py
```

**Requirements:** prepared `data/` and checkpoints under `part1_letter_classifier/models/` and `part2_word_recognizer/models/` (see main `README.md`).

---

## Hardware note

These runs used **CPU** where applicable for Part 2 and for Part 1 PyTorch inference; GPU changes wall time, not saved metrics for fixed checkpoints.
