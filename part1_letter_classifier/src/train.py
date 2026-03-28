"""
train.py
--------
Trains and saves four classifiers on the preprocessed ASL landmark features.

Models:
  1. SVM (RBF kernel)
  2. Random Forest
  3. MLP (2-layer feed-forward)
  4. CNN (on raw 64x64 images)  — uses PyTorch

Usage:
    # Defaults resolve to part1_letter_classifier/{data,models} (see --help).

    # Landmark-based models (SVM, RF, MLP):
    python train.py --mode landmarks

    # CNN on raw images:
    python train.py --mode cnn

    # Train all:
    python train.py --mode all
"""

import os
import argparse
from pathlib import Path
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
from tqdm import tqdm


# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PART1_ROOT = Path(__file__).resolve().parent.parent


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_landmark_data(data_dir):
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    label_map = np.load(os.path.join(data_dir, "label_map.npy"), allow_pickle=True).item()
    return X, y, label_map


def load_image_data(image_dir):
    """Load raw images resized to IMG_SIZE x IMG_SIZE, normalized to [0,1]."""
    X, y = [], []
    label_map = {}
    # Only keep single-character alphanumeric folders (0-9, a-z) — skips nested dirs
    folders = sorted([d for d in os.listdir(image_dir)
                      if os.path.isdir(os.path.join(image_dir, d))
                      and len(d) == 1 and d.isalnum()])
    for idx, folder in enumerate(folders):
        label_map[idx] = folder
        folder_path = os.path.join(image_dir, folder)
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fname in tqdm(files, desc=f"[CNN data] {folder}", leave=False):
            img = cv2.imread(os.path.join(folder_path, fname))
            if img is None:
                continue
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
            X.append(img.astype(np.float32) / 255.0)
            y.append(idx)
    return np.array(X), np.array(y, dtype=np.int64), label_map


def split_data(X, y, val_size=0.15, test_size=0.15):
    """Stratified 70/15/15 split."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Sklearn model trainers ────────────────────────────────────────────────────

def train_svm(X_train, y_train, X_val, y_val):
    print("\n[SVM] Training...")
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=SEED)
    model.fit(X_train, y_train)
    print(f"[SVM] Val accuracy: {accuracy_score(y_val, model.predict(X_val)):.4f}")
    return model


def train_rf(X_train, y_train, X_val, y_val):
    print("\n[Random Forest] Training...")
    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    model.fit(X_train, y_train)
    print(f"[Random Forest] Val accuracy: {accuracy_score(y_val, model.predict(X_val)):.4f}")
    return model


def train_mlp(X_train, y_train, X_val, y_val):
    print("\n[MLP] Training...")
    model = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu',
                          solver='adam', max_iter=300, early_stopping=True,
                          validation_fraction=0.1, random_state=SEED, verbose=False)
    model.fit(X_train, y_train)
    print(f"[MLP] Val accuracy: {accuracy_score(y_val, model.predict(X_val)):.4f}")
    return model


# ── PyTorch CNN ───────────────────────────────────────────────────────────────

class ASL_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_cnn(X_train, y_train, X_val, y_val, num_classes, models_dir,
              epochs=50, batch_size=32, patience=8):
    print(f"\n[CNN] Training on {DEVICE}...")

    # (N, H, W, C) → (N, C, H, W)
    def to_tensor(X, y):
        xt = torch.tensor(X).permute(0, 3, 1, 2)
        yt = torch.tensor(y)
        return TensorDataset(xt, yt)

    train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(to_tensor(X_val, y_val),   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    model = ASL_CNN(num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, no_improve = 0.0, 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        val_acc = correct / total
        scheduler.step(1 - val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(models_dir, "cnn_best.pt"))
    print(f"[CNN] Best val accuracy: {best_val_acc:.4f}")
    return model


def predict_cnn(model, X, batch_size=64):
    model.eval()
    xt = torch.tensor(X).permute(0, 3, 1, 2)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(xt), batch_size):
            batch = xt[i:i+batch_size].to(DEVICE)
            all_preds.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(all_preds)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["landmarks", "cnn", "all"], default="all")
    parser.add_argument("--data_dir", default=str(_PART1_ROOT / "data"))
    parser.add_argument("--image_dir", default=str(_PART1_ROOT / "data" / "asl_dataset"))
    parser.add_argument("--models_dir", default=str(_PART1_ROOT / "models"))
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    results = {}

    # ── Landmark models ────────────────────────────────────────────────────
    if args.mode in ("landmarks", "all"):
        print("Loading landmark data...")
        X, y, label_map = load_landmark_data(args.data_dir)
        print(f"  X shape: {X.shape}  classes: {len(label_map)}")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

        for name, trainer, path in [
            ("SVM", train_svm, "svm.pkl"),
            ("Random Forest", train_rf, "rf.pkl"),
            ("MLP", train_mlp, "mlp.pkl"),
        ]:
            m = trainer(X_train, y_train, X_val, y_val)
            joblib.dump(m, os.path.join(args.models_dir, path))
            acc = accuracy_score(y_test, m.predict(X_test))
            results[name] = acc
            print(f"[{name}] Test accuracy: {acc:.4f}")

        np.save(os.path.join(args.data_dir, "X_test.npy"), X_test)
        np.save(os.path.join(args.data_dir, "y_test.npy"), y_test)

    # ── CNN ────────────────────────────────────────────────────────────────
    if args.mode in ("cnn", "all"):
        print("\nLoading image data for CNN...")
        X_img, y_img, label_map_img = load_image_data(args.image_dir)
        print(f"  X_img shape: {X_img.shape}")
        X_train_i, X_val_i, X_test_i, y_train_i, y_val_i, y_test_i = split_data(X_img, y_img)

        cnn = train_cnn(X_train_i, y_train_i, X_val_i, y_val_i,
                        num_classes=len(label_map_img), models_dir=args.models_dir)
        preds = predict_cnn(cnn, X_test_i)
        acc = accuracy_score(y_test_i, preds)
        results["CNN"] = acc
        print(f"[CNN] Test accuracy: {acc:.4f}")

        np.save(os.path.join(args.data_dir, "X_test_img.npy"), X_test_i)
        np.save(os.path.join(args.data_dir, "y_test_img.npy"), y_test_i)

    print("\n" + "="*40)
    print("MODEL COMPARISON (Test Accuracy)")
    print("="*40)
    for name, acc in results.items():
        print(f"  {name:<20} {acc:.4f}")
    if results:
        best = max(results, key=results.get)
        print(f"\nBest model: {best} ({results[best]:.4f})")


if __name__ == "__main__":
    main()
