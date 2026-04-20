"""
train.py
--------
Trains classifiers on MediaPipe landmark features and on raw ASL images.

Models:
  1. SVM (RBF kernel)
  2. Random Forest
  3. MLP (2-layer feed-forward)
  4. CNN (custom conv net on 64×64 RGB)
  5. MobileNetV2 (torchvision, ImageNet-pretrained backbone by default)
  6. ResNet-18 (torchvision)
  7. VGG-11-BN (torchvision; lighter than VGG-16)

Usage:
    # Defaults resolve to part1_letter_classifier/{data,models} (see --help).

    # Landmark-based models (SVM, RF, MLP):
    python train.py --mode landmarks

    # CNN on raw images:
    python train.py --mode cnn

    # MobileNetV2 (transfer learning):
    python train.py --mode mobilenet

    # ResNet-18 or VGG-11-BN:
    python train.py --mode resnet
    python train.py --mode vgg

    # Train all (landmarks + CNN + MobileNet + ResNet + VGG):
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
import torchvision.models as tvm_models
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
        for fname in tqdm(files, desc=f"[images] {folder}", leave=False):
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
    print(f"\n[CNN] Training on {DEVICE} (epochs={epochs})...")

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


def predict_torch_images_proba(model, X, batch_size=64) -> np.ndarray:
    """Return softmax class probabilities, shape (N, num_classes)."""
    model.eval()
    xt = torch.tensor(X).permute(0, 3, 1, 2)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(xt), batch_size):
            batch = xt[i:i + batch_size].to(DEVICE)
            logits = model(batch)
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def predict_cnn(model, X, batch_size=64):
    """Class indices for each row of ``X`` (N, H, W, C) in [0,1]."""
    proba = predict_torch_images_proba(model, X, batch_size=batch_size)
    return np.argmax(proba, axis=1)


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with a new classification head (``fc``) for ``num_classes``."""
    try:
        weights = tvm_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm_models.resnet18(weights=weights)
    except AttributeError:
        m = tvm_models.resnet18(pretrained=pretrained)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m


def build_vgg11_bn(num_classes: int, pretrained: bool = True) -> nn.Module:
    """VGG-11-BN with the last linear replaced for ``num_classes``."""
    try:
        weights = tvm_models.VGG11_BN_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm_models.vgg11_bn(weights=weights)
    except AttributeError:
        m = tvm_models.vgg11_bn(pretrained=pretrained)
    in_f = m.classifier[6].in_features
    m.classifier[6] = nn.Linear(in_f, num_classes)
    return m


def build_mobilenet_v2(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    MobileNetV2 with a fresh classification head for ``num_classes`` labels.

    Parameters
    ----------
    pretrained
        If True, load ImageNet backbone weights (recommended for small datasets).
    """
    try:
        weights = tvm_models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    except AttributeError:
        # Older torchvision: fall back without enum
        m = tvm_models.mobilenet_v2(pretrained=pretrained)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        return m
    m = tvm_models.mobilenet_v2(weights=weights)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def _train_transfer_vision_model(
    model: nn.Module,
    weights_filename: str,
    display_name: str,
    X_train,
    y_train,
    X_val,
    y_val,
    models_dir: str,
    epochs=50,
    batch_size=32,
    patience=8,
    lr=3e-4,
    weight_decay=1e-4,
):
    """
    Shared fine-tuning loop for torchvision image classifiers (same 64×64 RGB tensors as CNN).
    Saves only ``state_dict`` to ``models_dir / weights_filename``.
    """
    print(f"\n[{display_name}] Training on {DEVICE}...")

    def to_tensor(X, y):
        xt = torch.tensor(X).permute(0, 3, 1, 2)
        yt = torch.tensor(y)
        return TensorDataset(xt, yt)

    train_loader = DataLoader(
        to_tensor(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        to_tensor(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
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
    torch.save(model.state_dict(), os.path.join(models_dir, weights_filename))
    print(f"[{display_name}] Best val accuracy: {best_val_acc:.4f}")
    return model


def train_mobilenet(
    X_train, y_train, X_val, y_val, num_classes, models_dir,
    epochs=50, batch_size=32, patience=8, pretrained=True,
):
    m = build_mobilenet_v2(num_classes, pretrained=pretrained)
    return _train_transfer_vision_model(
        m,
        "mobilenet_best.pt",
        "MobileNetV2",
        X_train,
        y_train,
        X_val,
        y_val,
        models_dir,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )


def train_resnet(
    X_train, y_train, X_val, y_val, num_classes, models_dir,
    epochs=50, batch_size=32, patience=8, pretrained=True,
):
    m = build_resnet18(num_classes, pretrained=pretrained)
    return _train_transfer_vision_model(
        m,
        "resnet18_best.pt",
        "ResNet-18",
        X_train,
        y_train,
        X_val,
        y_val,
        models_dir,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )


def train_vgg(
    X_train, y_train, X_val, y_val, num_classes, models_dir,
    epochs=50, batch_size=24, patience=8, pretrained=True,
):
    """Slightly smaller default batch for VGG memory use."""
    m = build_vgg11_bn(num_classes, pretrained=pretrained)
    return _train_transfer_vision_model(
        m,
        "vgg11_bn_best.pt",
        "VGG-11-BN",
        X_train,
        y_train,
        X_val,
        y_val,
        models_dir,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["landmarks", "cnn", "mobilenet", "resnet", "vgg", "transfer", "all"],
        default="all",
        help=(
            "landmarks | cnn | mobilenet | resnet | vgg | "
            "transfer (MobileNet+ResNet+VGG only, one image load) | "
            "all (landmarks + every image model)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Max epochs for CNN and torchvision transfer models (early stopping may stop sooner).",
    )
    parser.add_argument(
        "--no-imagenet-weights",
        action="store_true",
        help="For MobileNet/ResNet/VGG: train from random init instead of ImageNet-pretrained backbone.",
    )
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

    # ── CNN + torchvision transfer models (same image tensors) ─────────────
    if args.mode in ("cnn", "mobilenet", "resnet", "vgg", "transfer", "all"):
        print("\nLoading image data for PyTorch image models...")
        X_img, y_img, label_map_img = load_image_data(args.image_dir)
        print(f"  X_img shape: {X_img.shape}")
        X_train_i, X_val_i, X_test_i, y_train_i, y_val_i, y_test_i = split_data(X_img, y_img)
        n_cls = len(label_map_img)
        use_pretrained = not args.no_imagenet_weights
        ep = args.epochs

        run_cnn = args.mode in ("cnn", "all")
        run_mb = args.mode in ("mobilenet", "all", "transfer")
        run_rn = args.mode in ("resnet", "all", "transfer")
        run_vg = args.mode in ("vgg", "all", "transfer")

        if run_cnn:
            cnn = train_cnn(
                X_train_i, y_train_i, X_val_i, y_val_i,
                num_classes=n_cls, models_dir=args.models_dir,
                epochs=ep,
            )
            preds = predict_cnn(cnn, X_test_i)
            acc = accuracy_score(y_test_i, preds)
            results["CNN"] = acc
            print(f"[CNN] Test accuracy: {acc:.4f}")

        if run_mb:
            mb = train_mobilenet(
                X_train_i, y_train_i, X_val_i, y_val_i,
                num_classes=n_cls, models_dir=args.models_dir,
                pretrained=use_pretrained,
                epochs=ep,
            )
            preds_m = predict_cnn(mb, X_test_i)
            acc_m = accuracy_score(y_test_i, preds_m)
            results["MobileNetV2"] = acc_m
            print(f"[MobileNetV2] Test accuracy: {acc_m:.4f}")

        if run_rn:
            rn = train_resnet(
                X_train_i, y_train_i, X_val_i, y_val_i,
                num_classes=n_cls, models_dir=args.models_dir,
                pretrained=use_pretrained,
                epochs=ep,
            )
            preds_r = predict_cnn(rn, X_test_i)
            acc_r = accuracy_score(y_test_i, preds_r)
            results["ResNet-18"] = acc_r
            print(f"[ResNet-18] Test accuracy: {acc_r:.4f}")

        if run_vg:
            vg = train_vgg(
                X_train_i, y_train_i, X_val_i, y_val_i,
                num_classes=n_cls, models_dir=args.models_dir,
                pretrained=use_pretrained,
                epochs=ep,
            )
            preds_v = predict_cnn(vg, X_test_i)
            acc_v = accuracy_score(y_test_i, preds_v)
            results["VGG-11-BN"] = acc_v
            print(f"[VGG-11-BN] Test accuracy: {acc_v:.4f}")

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
