"""
eda_phase_b.py
--------------
Phase B EDA — analyses the preprocessed landmark sequences (X.npy, y.npy).
Saves all plots to results/ and prints a summary to stdout.

Run AFTER preprocessing.py.

Usage:
    python eda_phase_b.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ── paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.join(os.path.dirname(__file__), "..")
SEQ_DIR     = os.path.join(BASE, "data", "sequences")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

print("=" * 60)
print("  WLASL EDA — Phase B (preprocessed sequences)")
print("=" * 60)

# ── B1. Load arrays ───────────────────────────────────────────────────────────
X         = np.load(os.path.join(SEQ_DIR, "X.npy"))          # (N, 30, 225)
y         = np.load(os.path.join(SEQ_DIR, "y.npy"))          # (N,)
label_map = np.load(os.path.join(SEQ_DIR, "label_map.npy"),
                    allow_pickle=True).item()                  # {idx: word}
splits    = np.load(os.path.join(SEQ_DIR, "splits.npy"),
                    allow_pickle=True).item()                  # {video_id: split}
video_ids = np.load(os.path.join(SEQ_DIR, "video_ids.npy"),
                    allow_pickle=True)                         # (N,)

N, T, D = X.shape
print(f"\nLoaded X: {X.shape}  y: {y.shape}")
print(f"  Timesteps (T) : {T}")
print(f"  Feature dims  : {D}  (left_hand=63, right_hand=63, pose=99)")
print(f"  Classes       : {len(label_map)}")
print(f"  dtype         : {X.dtype}")

# ── B2. Class balance ─────────────────────────────────────────────────────────
class_counts = Counter(y.tolist())
counts_arr   = np.array([class_counts[i] for i in range(len(label_map))])
words        = [label_map[i] for i in range(len(label_map))]

print(f"\nSamples per class:")
print(f"  min  : {counts_arr.min()}  ({label_map[int(counts_arr.argmin())]})")
print(f"  max  : {counts_arr.max()}  ({label_map[int(counts_arr.argmax())]})")
print(f"  mean : {counts_arr.mean():.1f}")
print(f"  std  : {counts_arr.std():.1f}")

# Classes with very few samples
low = [(label_map[i], counts_arr[i]) for i in range(len(label_map)) if counts_arr[i] <= 3]
print(f"\n  Classes with <=3 samples ({len(low)}):")
for w, c in sorted(low, key=lambda x: x[1]):
    print(f"    {w:20s}: {c}")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Bar chart sorted by count
sort_idx = np.argsort(counts_arr)[::-1]
axes[0].bar(range(len(label_map)), counts_arr[sort_idx], color="steelblue", width=1.0)
axes[0].set_xlabel("Class rank")
axes[0].set_ylabel("Number of sequences")
axes[0].set_title("Samples per class (sorted)")
axes[0].axhline(counts_arr.mean(), color="red", linestyle="--", label=f"mean={counts_arr.mean():.1f}")
axes[0].legend()

axes[1].hist(counts_arr, bins=range(1, counts_arr.max() + 2), color="steelblue", edgecolor="white")
axes[1].set_xlabel("Number of sequences")
axes[1].set_ylabel("Number of classes")
axes[1].set_title("Distribution of samples per class")

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "eda_b2_class_balance.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {out}")

# ── B3. Split breakdown ───────────────────────────────────────────────────────
split_list = [splits.get(vid, "train") for vid in video_ids]
split_ser  = pd.Series(split_list)
split_counts = split_ser.value_counts()

print(f"\nSplit breakdown:")
print(split_counts.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
colors = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}
bar_colors = [colors.get(s, "gray") for s in split_counts.index]
split_counts.plot.bar(ax=ax, color=bar_colors, edgecolor="white")
ax.set_title("Train / Val / Test split (extracted sequences)")
ax.set_xlabel("Split")
ax.set_ylabel("Number of sequences")
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                ha="center", fontsize=11)
plt.xticks(rotation=0)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "eda_b3_split_breakdown.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── B4. Zero-frame analysis ───────────────────────────────────────────────────
zero_mask       = (X.sum(axis=2) == 0)          # (N, T) — True where frame is all-zero
zero_per_seq    = zero_mask.sum(axis=1)          # (N,) — count of zero frames per sequence
zero_rate_total = zero_mask.sum() / (N * 30)

print(f"\nZero-detection frames:")
print(f"  Total zero frames : {zero_mask.sum()} / {N*30} ({100*zero_rate_total:.1f}%)")
print(f"  Seqs with 0 zeros : {(zero_per_seq == 0).sum()}")
print(f"  Seqs with >5 zeros: {(zero_per_seq > 5).sum()}")
print(f"  Max zeros in 1 seq: {zero_per_seq.max()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(zero_per_seq, bins=range(0, int(zero_per_seq.max()) + 2),
             color="coral", edgecolor="white")
axes[0].set_xlabel("Zero frames per sequence")
axes[0].set_ylabel("Number of sequences")
axes[0].set_title("Zero-detection frames per sequence")

# Zero rate per timestep position
zero_by_pos = zero_mask.mean(axis=0)  # (30,)
axes[1].bar(range(30), zero_by_pos * 100, color="coral", edgecolor="white")
axes[1].set_xlabel("Frame position (0–29)")
axes[1].set_ylabel("Zero-detection rate (%)")
axes[1].set_title("Zero-detection rate by frame position")

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "eda_b4_zero_frames.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── B5. Feature statistics ────────────────────────────────────────────────────
# Only non-zero frames (zero = no detection)
X_flat = X.reshape(-1, D)
nonzero_rows = X_flat[X_flat.sum(axis=1) != 0]

print(f"\nFeature statistics (non-zero frames only, n={len(nonzero_rows)}):")

# Segment into LH / RH / Pose
lh   = nonzero_rows[:, :63]
rh   = nonzero_rows[:, 63:126]
pose = nonzero_rows[:, 126:]

for name, seg in [("Left hand (63D)", lh), ("Right hand (63D)", rh), ("Pose (99D)", pose)]:
    print(f"  {name}: mean={seg.mean():.4f}, std={seg.std():.4f}, "
          f"min={seg.min():.3f}, max={seg.max():.3f}")

# Per-feature std across dataset (which dims carry most variance)
feat_std = nonzero_rows.std(axis=0)
fig, axes = plt.subplots(3, 1, figsize=(14, 9))
for ax, (name, start, end) in zip(axes, [("Left hand", 0, 63),
                                          ("Right hand", 63, 126),
                                          ("Pose", 126, 225)]):
    ax.bar(range(end - start), feat_std[start:end], color="steelblue", width=1.0)
    ax.set_title(f"Per-feature std — {name}")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Std dev")

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "eda_b5_feature_variance.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── B6. Temporal dynamics ─────────────────────────────────────────────────────
# Mean absolute value per timestep (motion proxy)
X_nz = X.copy()
X_nz[zero_mask] = np.nan   # mask zero-detection frames
with np.errstate(all="ignore"):
    mean_abs_per_t = np.nanmean(np.abs(X_nz), axis=(0, 2))  # (30,)

# Per-class mean feature norm (which classes have more hand movement?)
class_norms = {}
for idx in range(len(label_map)):
    seqs = X[y == idx]          # (k, 30, 225)
    if len(seqs) == 0:
        continue
    norms = np.linalg.norm(seqs, axis=2)   # (k, 30)
    class_norms[label_map[idx]] = norms.mean()

norm_series = pd.Series(class_norms).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(range(30), mean_abs_per_t, marker="o", markersize=4, color="steelblue")
axes[0].set_xlabel("Frame position (0–29)")
axes[0].set_ylabel("Mean |feature value|")
axes[0].set_title("Mean feature magnitude across time")

top20 = norm_series.head(20)
axes[1].barh(range(len(top20)), top20.values[::-1], color="steelblue")
axes[1].set_yticks(range(len(top20)))
axes[1].set_yticklabels(top20.index[::-1], fontsize=8)
axes[1].set_xlabel("Mean feature norm")
axes[1].set_title("Top 20 classes by mean landmark activity")

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "eda_b6_temporal_dynamics.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── B7. Hand usage (left vs right) ────────────────────────────────────────────
lh_active  = (X[:, :, :63].sum(axis=(1,2)) != 0)    # (N,) True if LH used at all
rh_active  = (X[:, :, 63:126].sum(axis=(1,2)) != 0) # (N,)

both   = (lh_active & rh_active).sum()
rh_only = (~lh_active & rh_active).sum()
lh_only = (lh_active & ~rh_active).sum()
neither = (~lh_active & ~rh_active).sum()

print(f"\nHand usage across sequences:")
print(f"  Both hands : {both}  ({100*both/N:.1f}%)")
print(f"  Right only : {rh_only}  ({100*rh_only/N:.1f}%)")
print(f"  Left only  : {lh_only}  ({100*lh_only/N:.1f}%)")
print(f"  Neither    : {neither}  ({100*neither/N:.1f}%)")

fig, ax = plt.subplots(figsize=(7, 4))
labels  = ["Both hands", "Right only", "Left only", "Neither"]
values  = [both, rh_only, lh_only, neither]
colors_ = ["#2196F3", "#4CAF50", "#FF9800", "#9E9E9E"]
bars = ax.bar(labels, values, color=colors_, edgecolor="white")
for bar, val in zip(bars, values):
    ax.annotate(f"{val}\n({100*val/N:.0f}%)",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() + 1),
                ha="center", fontsize=10)
ax.set_title("Hand usage across sequences")
ax.set_ylabel("Number of sequences")
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "eda_b7_hand_usage.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

print("\n" + "=" * 60)
print("  Phase B EDA complete — charts saved to results/")
print("  Next: upload data/sequences/ to Colab and run train.py")
print("=" * 60)
