"""
evaluate.py
-----------
Evaluates trained BiLSTM and Transformer models on the WLASL test split.

Reports:
  - Top-1 and Top-5 accuracy per model
  - Confusion matrix (top-20 most confused class pairs)
  - Per-class accuracy saved to results/per_class_accuracy.csv
  - Model comparison bar chart

Usage:
    python evaluate.py
    python evaluate.py --data_dir ../data/sequences --models_dir ../models
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Reuse model definitions and dataset from train.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train import BiLSTMClassifier, TransformerClassifier, WLASLDataset

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA   = os.path.join(os.path.dirname(__file__), "..", "data", "sequences")
DEFAULT_MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_RESULTS= os.path.join(os.path.dirname(__file__), "..", "results")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_split(data_dir):
    X         = np.load(os.path.join(data_dir, "X.npy"))
    y         = np.load(os.path.join(data_dir, "y.npy"))
    splits    = np.load(os.path.join(data_dir, "splits.npy"),
                        allow_pickle=True).item()
    video_ids = np.load(os.path.join(data_dir, "video_ids.npy"),
                        allow_pickle=True)
    label_map = np.load(os.path.join(data_dir, "label_map.npy"),
                        allow_pickle=True).item()

    split_arr = np.array([splits.get(vid, "train") for vid in video_ids])
    te = split_arr == "test"
    return X[te], y[te], label_map


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    if "hidden_dim" in cfg:
        model = BiLSTMClassifier(**cfg)
        name  = "BiLSTM"
    else:
        model = TransformerClassifier(**cfg)
        name  = "Transformer"

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, name, ckpt.get("metrics", {})


def run_inference(model, loader, device):
    all_preds  = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y)

    logits = torch.cat(all_logits)
    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return logits, preds, labels


def top1_acc(preds, labels):
    return (preds == labels).float().mean().item()


def top5_acc(logits, labels):
    k = min(5, logits.size(1))
    _, topk = logits.topk(k, dim=1)
    correct = topk.eq(labels.unsqueeze(1).expand_as(topk))
    return correct.any(dim=1).float().mean().item()


# ── Confusion matrix (top-20 confused pairs) ─────────────────────────────────

def plot_confusion(preds, labels, label_map, model_name, results_dir):
    num_classes = len(label_map)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds.numpy(), labels.numpy()):
        cm[l, p] += 1

    # Find top-20 most confused off-diagonal pairs
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    flat_idx = np.argsort(cm_copy.ravel())[::-1][:20]
    row_idx, col_idx = np.unravel_index(flat_idx, cm.shape)

    confused_classes = sorted(set(row_idx) | set(col_idx))
    sub_labels = [label_map[i] for i in confused_classes]
    sub_cm = cm[np.ix_(confused_classes, confused_classes)]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sub_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sub_labels, yticklabels=sub_labels,
                linewidths=0.5, ax=ax)
    ax.set_title(f"{model_name} — Top-20 Most Confused Classes", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    fname = f"confusion_{model_name.lower()}.png"
    out   = os.path.join(results_dir, fname)
    plt.savefig(out, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  Saved: {out}")
    return cm


# ── Per-class accuracy ────────────────────────────────────────────────────────

def per_class_accuracy(preds, labels, label_map):
    rows = []
    for idx, word in label_map.items():
        mask    = labels == idx
        total   = mask.sum().item()
        correct = (preds[mask] == idx).sum().item() if total > 0 else 0
        acc     = correct / total if total > 0 else float("nan")
        rows.append({"class_idx": idx, "word": word,
                     "total": total, "correct": correct, "accuracy": acc})
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


# ── Model comparison chart ────────────────────────────────────────────────────

def plot_comparison(results, results_dir):
    names  = list(results.keys())
    top1   = [results[n]["top1"] for n in names]
    top5   = [results[n]["top5"] for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, [v*100 for v in top1], w,
                   label="Top-1", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x + w/2, [v*100 for v in top5], w,
                   label="Top-5", color="#4CAF50", edgecolor="white")

    for bar in bars1 + bars2:
        ax.annotate(f"{bar.get_height():.1f}%",
                    (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5),
                    ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Comparison — WLASL100 Test Set")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(out, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate WLASL models on test set")
    parser.add_argument("--data_dir",    default=DEFAULT_DATA)
    parser.add_argument("--models_dir",  default=DEFAULT_MODELS)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS)
    parser.add_argument("--batch_size",  type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test data
    X_te, y_te, label_map = load_test_split(args.data_dir)
    ds_test   = WLASLDataset(X_te, y_te, augment=False)
    loader    = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
    print(f"Test samples: {len(ds_test)}  |  Classes: {len(label_map)}\n")

    results    = {}
    all_dfs    = []

    print("=" * 60)
    for fname in ["bilstm_best.pt", "transformer_best.pt"]:
        ckpt_path = os.path.join(args.models_dir, fname)
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {fname} (not found)")
            continue

        model, name, train_metrics = load_model(ckpt_path, device)
        print(f"Model: {name}")
        if train_metrics:
            print(f"  Best val top-1 (training): {train_metrics.get('val_top1', 'n/a'):.3f}"
                  f"  epoch={train_metrics.get('epoch', 'n/a')}")

        logits, preds, labels = run_inference(model, loader, device)
        t1 = top1_acc(preds, labels)
        t5 = top5_acc(logits, labels)

        print(f"  Test top-1 : {t1:.3f}  ({t1*100:.1f}%)")
        print(f"  Test top-5 : {t5:.3f}  ({t5*100:.1f}%)")

        # Confusion matrix
        plot_confusion(preds, labels, label_map, name, args.results_dir)

        # Per-class accuracy
        df = per_class_accuracy(preds, labels, label_map)
        df["model"] = name
        all_dfs.append(df)

        results[name] = {"top1": t1, "top5": t5}
        print()

    # Save per-class CSV
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        csv_path = os.path.join(args.results_dir, "per_class_accuracy.csv")
        combined.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # Print top and bottom 5 classes for each model
        for name in results:
            sub = combined[combined["model"] == name].dropna(subset=["accuracy"])
            print(f"\n{name} — Top 5 classes:")
            print(sub.head(5)[["word", "total", "correct", "accuracy"]].to_string(index=False))
            print(f"\n{name} — Bottom 5 classes:")
            print(sub.tail(5)[["word", "total", "correct", "accuracy"]].to_string(index=False))

    # Model comparison chart
    if len(results) > 0:
        plot_comparison(results, args.results_dir)

    print("\n" + "=" * 60)
    print("  Evaluation complete. Results saved to", args.results_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
