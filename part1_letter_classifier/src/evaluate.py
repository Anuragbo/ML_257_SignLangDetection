"""
evaluate.py
-----------
Loads saved models and the held-out test set, then produces:
  - Per-model accuracy, precision, recall, F1
  - 26x26 confusion matrices (saved as PNG)
  - Side-by-side model comparison bar chart (saved as PNG)

Usage:
    python evaluate.py
    # Defaults: data/models/results under part1_letter_classifier/
"""

import os
import argparse
from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)

import torch

_PART1_ROOT = Path(__file__).resolve().parent.parent


def load_label_map(data_dir):
    return np.load(os.path.join(data_dir, "label_map.npy"), allow_pickle=True).item()


def evaluate_model(name, y_true, y_pred, label_names, output_dir):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.3, ax=ax
    )
    ax.set_title(f"{name} — Confusion Matrix\nAccuracy: {acc:.4f}", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"confusion_{name.replace(' ', '_').lower()}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {fname}")

    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_comparison(results, output_dir):
    names = [r["name"] for r in results]
    metrics = ["accuracy", "f1", "precision", "recall"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — ASL Letter Classification", fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nComparison chart saved -> {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(_PART1_ROOT / "data"))
    parser.add_argument("--models_dir", default=str(_PART1_ROOT / "models"))
    parser.add_argument("--output_dir", default=str(_PART1_ROOT / "results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    label_map = load_label_map(args.data_dir)
    label_names = [label_map[i] for i in range(len(label_map))]

    # Load landmark test split
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

    results = []

    # ── SVM ──────────────────────────────────────────────────────────────
    svm_path = os.path.join(args.models_dir, "svm.pkl")
    if os.path.exists(svm_path):
        svm = joblib.load(svm_path)
        y_pred = svm.predict(X_test)
        results.append(evaluate_model("SVM", y_test, y_pred, label_names, args.output_dir))

    # ── Random Forest ─────────────────────────────────────────────────────
    rf_path = os.path.join(args.models_dir, "rf.pkl")
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
        y_pred = rf.predict(X_test)
        results.append(evaluate_model("Random Forest", y_test, y_pred, label_names, args.output_dir))

    # ── MLP ───────────────────────────────────────────────────────────────
    mlp_path = os.path.join(args.models_dir, "mlp.pkl")
    if os.path.exists(mlp_path):
        mlp = joblib.load(mlp_path)
        y_pred = mlp.predict(X_test)
        results.append(evaluate_model("MLP", y_test, y_pred, label_names, args.output_dir))

    # ── CNN ───────────────────────────────────────────────────────────────
    cnn_path = os.path.join(args.models_dir, "cnn_best.pt")
    X_test_img_path = os.path.join(args.data_dir, "X_test_img.npy")
    y_test_img_path = os.path.join(args.data_dir, "y_test_img.npy")

    if os.path.exists(cnn_path) and os.path.exists(X_test_img_path):
        from train import ASL_CNN, predict_cnn
        X_test_img = np.load(X_test_img_path)
        y_test_img = np.load(y_test_img_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn = ASL_CNN(num_classes=len(label_map)).to(device)
        cnn.load_state_dict(torch.load(cnn_path, map_location=device))
        y_pred = predict_cnn(cnn, X_test_img)
        results.append(evaluate_model("CNN", y_test_img, y_pred, label_names, args.output_dir))

    if results:
        plot_comparison(results, args.output_dir)

        print("\n" + "="*50)
        print("FINAL RANKING (by F1 score)")
        print("="*50)
        for r in sorted(results, key=lambda x: x["f1"], reverse=True):
            print(f"  {r['name']:<20} Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}")
    else:
        print("No saved models found. Run train.py first.")


if __name__ == "__main__":
    main()
