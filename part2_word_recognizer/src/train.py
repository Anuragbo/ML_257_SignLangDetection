"""
train.py
--------
Trains BiLSTM and Transformer classifiers on WLASL landmark sequences.

Designed to run on Google Colab (GPU) after uploading data/sequences/.
Can also run locally on CPU (slower).

Usage (Colab):
    # Mount Drive and set paths, then:
    python train.py --data_dir /content/drive/MyDrive/wlasl/sequences \
                    --output_dir /content/drive/MyDrive/wlasl/models \
                    --model both

Usage (local):
    python train.py --data_dir ../data/sequences --output_dir ../models --model both

Outputs (in output_dir/):
    bilstm_best.pt       best BiLSTM checkpoint  {model_state, config, metrics}
    transformer_best.pt  best Transformer checkpoint
    training_curves.png  loss + accuracy plots for both models
"""

import os
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

class WLASLDataset(Dataset):
    """
    Wraps the (N, 30, 225) landmark sequences with optional augmentation.

    Augmentations (training only):
      - Gaussian noise  : add N(0, 0.01) to all landmark values
      - Frame dropout   : zero out 1-2 random frames (p=0.5)
      - Temporal scaling: resample at 0.8-1.2x speed then resize back to 30
    """

    def __init__(self, X, y, augment=False):
        self.X       = torch.from_numpy(X).float()   # (N, 30, 225)
        self.y       = torch.from_numpy(y).long()    # (N,)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()   # (30, 225)
        y = self.y[idx]

        if self.augment:
            # Gaussian noise
            x = x + torch.randn_like(x) * 0.01

            # Frame dropout (zero out 1-2 frames)
            if torch.rand(1).item() < 0.5:
                n_drop = torch.randint(1, 3, (1,)).item()
                drop_idx = torch.randperm(30)[:n_drop]
                x[drop_idx] = 0.0

            # Temporal scaling (0.8–1.2×)
            if torch.rand(1).item() < 0.4:
                scale   = 0.8 + torch.rand(1).item() * 0.4   # in [0.8, 1.2]
                src_len = max(2, int(round(30 * scale)))
                # Sample src_len evenly-spaced frames, then interpolate back to 30
                src_idx = torch.linspace(0, 29, src_len).long().clamp(0, 29)
                sampled = x[src_idx]                          # (src_len, 225)
                # Interpolate: (1, 225, src_len) → (1, 225, 30)
                sampled_t = sampled.T.unsqueeze(0)            # (1, 225, src_len)
                resampled = F.interpolate(sampled_t, size=30,
                                          mode="linear", align_corners=False)
                x = resampled.squeeze(0).T                    # (30, 225)

        return x, y


def make_datasets(data_dir):
    """Load npy files and split into train/val/test WLASLDatasets."""
    X         = np.load(os.path.join(data_dir, "X.npy"))
    y         = np.load(os.path.join(data_dir, "y.npy"))
    splits    = np.load(os.path.join(data_dir, "splits.npy"),
                        allow_pickle=True).item()
    video_ids = np.load(os.path.join(data_dir, "video_ids.npy"),
                        allow_pickle=True)
    label_map = np.load(os.path.join(data_dir, "label_map.npy"),
                        allow_pickle=True).item()

    split_arr = np.array([splits.get(vid, "train") for vid in video_ids])

    tr = split_arr == "train"
    va = split_arr == "val"
    te = split_arr == "test"

    print(f"Dataset  |  Train: {tr.sum()}  Val: {va.sum()}  Test: {te.sum()}"
          f"  Classes: {len(label_map)}")

    ds_train = WLASLDataset(X[tr], y[tr], augment=True)
    ds_val   = WLASLDataset(X[va], y[va], augment=False)
    ds_test  = WLASLDataset(X[te], y[te], augment=False)

    return ds_train, ds_val, ds_test, label_map


def make_weighted_sampler(dataset):
    """WeightedRandomSampler so every class is seen equally per epoch."""
    labels      = dataset.y.numpy()
    class_counts = np.bincount(labels)
    weights      = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )


# ── Models ────────────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    2-layer Bidirectional LSTM over 30-frame landmark sequences.

    Architecture:
        BiLSTM(225→256×2, layers=2, dropout=0.3)
        → mean pool over T=30
        → Linear(512→256) + ReLU + Dropout(0.3)
        → Linear(256→num_classes)
    """

    def __init__(self, input_dim=225, hidden_dim=256, num_layers=2,
                 num_classes=100, dropout=0.3):
        super().__init__()
        self.config = dict(input_dim=input_dim, hidden_dim=hidden_dim,
                           num_layers=num_layers, num_classes=num_classes,
                           dropout=dropout)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 225)
        out, _ = self.lstm(x)          # (B, T, 512)
        pooled = out.mean(dim=1)       # (B, 512)
        return self.classifier(pooled) # (B, num_classes)


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=30):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    """
    Transformer encoder over 30-frame landmark sequences.

    Architecture:
        Linear(225→256)          [input projection]
        SinusoidalPE(256)
        TransformerEncoder(d=256, heads=8, ffn=512, layers=4, Pre-LN, dropout=0.1)
        → mean pool over T=30
        → Linear(256→num_classes)
    """

    def __init__(self, input_dim=225, d_model=256, nhead=8,
                 num_layers=4, dim_feedforward=512,
                 dropout=0.1, num_classes=100):
        super().__init__()
        self.config = dict(input_dim=input_dim, d_model=d_model, nhead=nhead,
                           num_layers=num_layers, dim_feedforward=dim_feedforward,
                           dropout=dropout, num_classes=num_classes)

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = SinusoidalPE(d_model, max_len=30)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm for stability on small datasets
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, 225)
        x = self.input_proj(x)          # (B, T, 256)
        x = self.pos_enc(x)
        x = self.encoder(x)             # (B, T, 256)
        pooled = x.mean(dim=1)          # (B, 256)
        return self.classifier(pooled)  # (B, num_classes)


# ── Metrics ───────────────────────────────────────────────────────────────────

def topk_accuracy(logits, targets, k=5):
    """Compute top-k accuracy."""
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.unsqueeze(1).expand_as(pred))
    return correct.any(dim=1).float().mean().item()


# ── Train / Eval loops ────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, top1_sum, top5_sum, n = 0.0, 0.0, 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs       = y.size(0)
            total_loss += loss.item() * bs
            top1_sum   += (logits.argmax(1) == y).float().sum().item()
            top5_sum   += topk_accuracy(logits, y, k=min(5, logits.size(1))) * bs
            n          += bs

    return total_loss / n, top1_sum / n, top5_sum / n


# ── Training orchestration ────────────────────────────────────────────────────

def train_model(model, ds_train, ds_val, args, device, model_name):
    """Full training loop with early stopping and LR scheduling."""

    sampler = make_weighted_sampler(ds_train)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [],
               "train_top5": [], "val_top5": []}

    print(f"\n{'='*60}")
    print(f"  Training {model_name}  |  device={device}")
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")
    print(f"{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  {'TrTop5':>7}"
          f"  {'VaLoss':>8}  {'VaAcc':>7}  {'VaTop5':>7}  {'LR':>9}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_top5 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc, va_top5 = run_epoch(
            model, val_loader,   criterion, optimizer, device, train=False)

        scheduler.step(va_acc)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_top5"].append(tr_top5)
        history["val_top5"].append(va_top5)

        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_acc:>7.3f}  {tr_top5:>7.3f}"
              f"  {va_loss:>8.4f}  {va_acc:>7.3f}  {va_top5:>7.3f}  {lr_now:>9.2e}")

        # Save best checkpoint
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_ctr = 0
            ckpt_path = os.path.join(args.output_dir,
                                     f"{model_name.lower().replace(' ', '_')}_best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config":      model.config,
                "metrics":     {"val_top1": va_acc, "val_top5": va_top5,
                                "epoch": epoch},
            }, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val_acc={best_val_acc:.3f})")
                break

    print(f"\n  Best val top-1: {best_val_acc:.3f}")
    return history


# ── Training curves ───────────────────────────────────────────────────────────

def plot_curves(histories, output_dir):
    """Plot loss + top-1 accuracy curves for all trained models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"BiLSTM": "#2196F3", "Transformer": "#FF5722"}

    for name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        c = colors.get(name, "gray")
        axes[0].plot(epochs, hist["train_loss"], "--", color=c, alpha=0.6,
                     label=f"{name} train")
        axes[0].plot(epochs, hist["val_loss"],   "-",  color=c,
                     label=f"{name} val")
        axes[1].plot(epochs, hist["train_acc"],  "--", color=c, alpha=0.6,
                     label=f"{name} train")
        axes[1].plot(epochs, hist["val_acc"],    "-",  color=c,
                     label=f"{name} val")

    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Top-1 Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"\nSaved training curves -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train WLASL sequence classifiers")
    parser.add_argument("--data_dir",   default="../data/sequences",
                        help="Folder with X.npy, y.npy, splits.npy, …")
    parser.add_argument("--output_dir", default="../models",
                        help="Where to save checkpoints and plots")
    parser.add_argument("--model",      choices=["bilstm", "transformer", "both"],
                        default="both", help="Which model(s) to train")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience",   type=int, default=15,
                        help="Early-stopping patience (val_acc epochs)")
    parser.add_argument("--lr_bilstm",  type=float, default=1e-3)
    parser.add_argument("--lr_trans",   type=float, default=5e-4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    ds_train, ds_val, ds_test, label_map = make_datasets(args.data_dir)
    num_classes = len(label_map)

    histories = {}

    # ── BiLSTM ────────────────────────────────────────────────────────────────
    if args.model in ("bilstm", "both"):
        args.lr = args.lr_bilstm
        model   = BiLSTMClassifier(num_classes=num_classes).to(device)
        hist    = train_model(model, ds_train, ds_val, args, device, "BiLSTM")
        histories["BiLSTM"] = hist

    # ── Transformer ───────────────────────────────────────────────────────────
    if args.model in ("transformer", "both"):
        args.lr = args.lr_trans
        model   = TransformerClassifier(num_classes=num_classes).to(device)
        hist    = train_model(model, ds_train, ds_val, args, device, "Transformer")
        histories["Transformer"] = hist

    # ── Plots ─────────────────────────────────────────────────────────────────
    if histories:
        plot_curves(histories, args.output_dir)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Test-set evaluation")
    print("=" * 60)

    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
    criterion   = nn.CrossEntropyLoss()

    for model_name, fname in [("BiLSTM", "bilstm_best.pt"),
                               ("Transformer", "transformer_best.pt")]:
        ckpt_path = os.path.join(args.output_dir, fname)
        if not os.path.exists(ckpt_path):
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = ckpt["config"]

        if model_name == "BiLSTM":
            model = BiLSTMClassifier(**cfg).to(device)
        else:
            model = TransformerClassifier(**cfg).to(device)

        model.load_state_dict(ckpt["model_state"])

        te_loss, te_top1, te_top5 = run_epoch(
            model, test_loader, criterion, None, device, train=False)

        print(f"  {model_name:12s}  "
              f"loss={te_loss:.4f}  top-1={te_top1:.3f}  top-5={te_top5:.3f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
