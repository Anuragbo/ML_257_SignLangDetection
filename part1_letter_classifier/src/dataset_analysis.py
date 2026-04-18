"""
dataset_analysis.py
-------------------
Comprehensive dataset audit for the ASL landmark project.

Outputs:
  - Console summary table (exact counts, balance, duplicates, detection rate)
  - results/dataset_raw_distribution.png   (raw per-class image count)
  - results/dataset_detection_rate.png     (MediaPipe detection rate per class)
  - results/dataset_post_preprocessing.png (per-class count after landmark filtering)
  - results/dataset_feature_distribution.png (63-dim feature histogram)

Usage (from project root):
    python part1_letter_classifier/src/dataset_analysis.py
Or (from src/):
    python dataset_analysis.py
"""

import os
import sys
import hashlib
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "data", "asl_dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import preprocessing utilities
sys.path.insert(0, SRC_DIR)
from preprocessing import ensure_model, make_detector, extract_landmarks

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


# ── Helper ────────────────────────────────────────────────────────────────────

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def get_class_folders(data_dir):
    """Return sorted list of single-character alphanumeric class folder names."""
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and len(d) == 1 and d.isalnum()
    ])


def collect_image_paths(data_dir, classes):
    """Return dict {class_name: [image_paths]}."""
    class_paths = {}
    for cls in classes:
        folder = os.path.join(data_dir, cls)
        paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTS)
        ]
        class_paths[cls] = paths
    return class_paths


# ── Step 1: Raw dataset audit ─────────────────────────────────────────────────

def raw_audit(class_paths):
    print("\n" + "="*60)
    print("STEP 1 — RAW DATASET AUDIT")
    print("="*60)

    counts = {cls: len(paths) for cls, paths in class_paths.items()}
    total  = sum(counts.values())
    vals   = list(counts.values())

    print(f"  Total raw images : {total}")
    print(f"  Number of classes: {len(counts)}")
    print(f"  Min per class    : {min(vals)}  ({min(counts, key=counts.get)})")
    print(f"  Max per class    : {max(vals)}  ({max(counts, key=counts.get)})")
    print(f"  Mean per class   : {np.mean(vals):.1f}")
    print(f"  Std per class    : {np.std(vals):.1f}")

    print("\n  Per-class counts:")
    print(f"  {'Class':<8} {'Count':>6}")
    print(f"  {'-'*8} {'-'*6}")
    for cls in sorted(counts):
        print(f"  {cls:<8} {counts[cls]:>6}")

    return counts, total


# ── Step 2: Duplicate detection ───────────────────────────────────────────────

def duplicate_audit(class_paths):
    print("\n" + "="*60)
    print("STEP 2 — DUPLICATE DETECTION (MD5 hash)")
    print("="*60)

    hash_map = defaultdict(list)
    all_paths = [p for paths in class_paths.values() for p in paths]
    print(f"  Hashing {len(all_paths)} images...")
    for path in tqdm(all_paths, desc="  Hashing", leave=False):
        try:
            hash_map[md5(path)].append(path)
        except Exception:
            pass

    duplicate_groups = {h: ps for h, ps in hash_map.items() if len(ps) > 1}
    dup_count = sum(len(ps) - 1 for ps in duplicate_groups.values())

    print(f"  Duplicate groups : {len(duplicate_groups)}")
    print(f"  Duplicate images : {dup_count}  (extra copies beyond first)")
    if duplicate_groups:
        print("  Sample duplicates (first 3 groups):")
        for i, (h, ps) in enumerate(list(duplicate_groups.items())[:3]):
            print(f"    Group {i+1}: {len(ps)} copies — {[os.path.basename(p) for p in ps]}")

    return dup_count, len(duplicate_groups)


# ── Step 3: MediaPipe detection rate ──────────────────────────────────────────

def detection_rate_audit(class_paths):
    print("\n" + "="*60)
    print("STEP 3 — MEDIAPIPE LANDMARK DETECTION RATE")
    print("="*60)

    ensure_model()
    detector = make_detector()

    per_class_results = {}  # cls -> (detected, total)
    total_detected = 0
    total_images   = 0

    for cls, paths in sorted(class_paths.items()):
        detected = 0
        for path in tqdm(paths, desc=f"  [{cls}]", leave=False):
            img = cv2.imread(path)
            if img is None:
                continue
            features = extract_landmarks(img, detector)
            if features is not None:
                detected += 1
        per_class_results[cls] = (detected, len(paths))
        total_detected += detected
        total_images   += len(paths)

    detector.close()

    overall_rate = total_detected / total_images * 100 if total_images > 0 else 0
    failed_count = total_images - total_detected

    print(f"\n  Total images tested   : {total_images}")
    print(f"  Successful detections : {total_detected}")
    print(f"  Failed detections     : {failed_count}")
    print(f"  Overall detection rate: {overall_rate:.1f}%")

    print(f"\n  {'Class':<8} {'Detected':>9} {'Total':>7} {'Rate':>7}")
    print(f"  {'-'*8} {'-'*9} {'-'*7} {'-'*7}")
    for cls in sorted(per_class_results):
        det, tot = per_class_results[cls]
        rate = det / tot * 100 if tot > 0 else 0
        flag = " ⚠" if rate < 70 else (" ~" if rate < 90 else "")
        print(f"  {cls:<8} {det:>9} {tot:>7} {rate:>6.1f}%{flag}")

    low_classes = [cls for cls, (d, t) in per_class_results.items()
                   if t > 0 and d/t < 0.70]
    if low_classes:
        print(f"\n  Classes with <70% detection: {low_classes}")

    return per_class_results, overall_rate, failed_count


# ── Step 4: Post-preprocessing stats ─────────────────────────────────────────

def postprep_stats(output_dir):
    print("\n" + "="*60)
    print("STEP 4 — POST-PREPROCESSING STATS (from X.npy / y.npy)")
    print("="*60)

    X         = np.load(os.path.join(output_dir, "X.npy"))
    y         = np.load(os.path.join(output_dir, "y.npy"))
    label_map = np.load(os.path.join(output_dir, "label_map.npy"), allow_pickle=True).item()

    print(f"  X shape      : {X.shape}  → {X.shape[0]} usable samples, {X.shape[1]} features")
    print(f"  y shape      : {y.shape}")
    print(f"  Classes      : {len(label_map)}")
    print(f"  Feature range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  Feature mean : {X.mean():.4f}")
    print(f"  Feature std  : {X.std():.4f}")

    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Post-preprocessing per-class sample counts:")
    print(f"  {'Class':<8} {'Samples':>8}")
    print(f"  {'-'*8} {'-'*8}")
    for idx, cnt in zip(unique, counts):
        print(f"  {label_map[idx]:<8} {cnt:>8}")

    print(f"\n  Min samples/class: {counts.min()}  ({label_map[unique[counts.argmin()]]})")
    print(f"  Max samples/class: {counts.max()}  ({label_map[unique[counts.argmax()]]})")
    print(f"  Mean samples/class: {counts.mean():.1f}")

    return X, y, label_map, unique, counts


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_raw_distribution(raw_counts, results_dir):
    classes = sorted(raw_counts.keys())
    values  = [raw_counts[c] for c in classes]
    mean    = np.mean(values)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(classes, values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axhline(mean, color="crimson", linestyle="--", linewidth=1.5, label=f"Mean = {mean:.0f}")
    ax.set_title("Raw Dataset: Image Count per Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("ASL Class (letter / digit)")
    ax.set_ylabel("Number of Images")
    ax.legend()
    ax.set_ylim(0, max(values) * 1.15)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    path = os.path.join(results_dir, "dataset_raw_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved: {path}")


def plot_detection_rate(per_class_results, results_dir):
    classes = sorted(per_class_results.keys())
    rates   = [per_class_results[c][0] / per_class_results[c][1] * 100
                if per_class_results[c][1] > 0 else 0 for c in classes]
    colors  = ["#2ecc71" if r >= 90 else ("#f39c12" if r >= 70 else "#e74c3c")
               for r in rates]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(classes, rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(90, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.7, label="≥90% (good)")
    ax.axhline(70, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="<70% (poor)")
    ax.set_title("MediaPipe Hand Detection Rate per Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("ASL Class (letter / digit)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_ylim(0, 110)
    patches = [
        mpatches.Patch(color="#2ecc71", label="≥90% (good)"),
        mpatches.Patch(color="#f39c12", label="70–90% (moderate)"),
        mpatches.Patch(color="#e74c3c", label="<70% (poor)"),
    ]
    ax.legend(handles=patches, loc="lower right")
    plt.tight_layout()
    path = os.path.join(results_dir, "dataset_detection_rate.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_post_preprocessing(label_map, unique, counts, results_dir):
    classes = [label_map[i] for i in unique]
    mean    = counts.mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(classes, counts, color="teal", edgecolor="white", linewidth=0.5)
    ax.axhline(mean, color="crimson", linestyle="--", linewidth=1.5, label=f"Mean = {mean:.0f}")
    ax.set_title("Post-Preprocessing: Usable Samples per Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("ASL Class (letter / digit)")
    ax.set_ylabel("Number of Usable Samples")
    ax.legend()
    ax.set_ylim(0, counts.max() * 1.15)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    path = os.path.join(results_dir, "dataset_post_preprocessing.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_distribution(X, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(X.flatten(), bins=80, color="steelblue", edgecolor="none", alpha=0.85)
    axes[0].set_title("All 63 Features (flattened)")
    axes[0].set_xlabel("Feature value")
    axes[0].set_ylabel("Frequency")

    per_feat_mean = X.mean(axis=0)
    per_feat_std  = X.std(axis=0)
    feat_idx      = np.arange(63)
    axes[1].bar(feat_idx, per_feat_mean, color="teal", alpha=0.8, label="Mean")
    axes[1].fill_between(feat_idx,
                          per_feat_mean - per_feat_std,
                          per_feat_mean + per_feat_std,
                          alpha=0.3, color="teal", label="±1 Std")
    axes[1].set_title("Per-Feature Mean ± Std")
    axes[1].set_xlabel("Feature index (0–62)")
    axes[1].set_ylabel("Value")
    axes[1].legend(fontsize=8)

    axes[2].bar(feat_idx, per_feat_std, color="coral", alpha=0.85)
    axes[2].set_title("Per-Feature Standard Deviation")
    axes[2].set_xlabel("Feature index (0–62)")
    axes[2].set_ylabel("Std deviation")

    fig.suptitle("Feature Distribution Analysis (63-dim Landmark Vectors)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(results_dir, "dataset_feature_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("ASL DATASET COMPREHENSIVE ANALYSIS")
    print("="*60)
    print(f"  Data dir   : {DATA_DIR}")
    print(f"  Results dir: {RESULTS_DIR}")

    classes     = get_class_folders(DATA_DIR)
    class_paths = collect_image_paths(DATA_DIR, classes)

    # Step 1: Raw counts
    raw_counts, total_raw = raw_audit(class_paths)
    plot_raw_distribution(raw_counts, RESULTS_DIR)

    # Step 2: Duplicates
    dup_count, dup_groups = duplicate_audit(class_paths)

    # Step 3: Detection rate
    per_class_results, overall_rate, failed_count = detection_rate_audit(class_paths)
    plot_detection_rate(per_class_results, RESULTS_DIR)

    # Step 4: Post-preprocessing from saved npy
    X, y, label_map, unique, counts = postprep_stats(OUTPUT_DIR)
    plot_post_preprocessing(label_map, unique, counts, RESULTS_DIR)
    plot_feature_distribution(X, RESULTS_DIR)

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY (for progress report)")
    print("="*60)
    print(f"  Raw images (total)         : {total_raw}")
    print(f"  Classes                    : {len(classes)}")
    print(f"  Duplicate images           : {dup_count} across {dup_groups} groups")
    print(f"  MediaPipe detection rate   : {overall_rate:.1f}%")
    print(f"  Failed landmark extractions: {failed_count} ({failed_count/total_raw*100:.1f}%)")
    print(f"  Usable samples (post-prep) : {X.shape[0]}")
    print(f"  Feature dimensions         : {X.shape[1]}")
    print(f"  Feature range              : [{X.min():.4f}, {X.max():.4f}]")
    print(f"  Min samples/class (usable) : {counts.min()}")
    print(f"  Max samples/class (usable) : {counts.max()}")
    print(f"  Mean samples/class (usable): {counts.mean():.1f}")
    print("="*60)
    print("Done. All plots saved to results/")


if __name__ == "__main__":
    main()
