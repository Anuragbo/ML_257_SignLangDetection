"""
eda_phase_a.py
--------------
Phase A EDA — analyses the raw WLASL JSON and download results.
Saves all plots to results/ and prints a summary to stdout.

Run BEFORE preprocessing.py.

Usage:
    python eda_phase_a.py
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ── paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(BASE, "data")
WLASL_JSON  = os.path.join(DATA_DIR, "WLASL_v0.3.json")
AVAIL_JSON  = os.path.join(DATA_DIR, "available_videos.json")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

SUBSET = 100   # WLASL100

# ── A1. Load JSON ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  WLASL EDA — Phase A (raw dataset)")
print("=" * 60)

with open(WLASL_JSON) as f:
    wlasl_data = json.load(f)

print(f"\nTotal glosses in full WLASL: {len(wlasl_data)}")
print(f"Example gloss entry keys   : {list(wlasl_data[0].keys())}")
print(f"Example instance keys      : {list(wlasl_data[0]['instances'][0].keys())}")

rows = []
for entry in wlasl_data:
    gloss = entry["gloss"]
    for inst in entry.get("instances", []):
        rows.append({
            "gloss":        gloss,
            "video_id":     str(inst.get("video_id", "")).zfill(5),
            "split":        inst.get("split", "train"),
            "signer_id":    inst.get("signer_id", -1),
            "frame_start":  inst.get("frame_start", 1),
            "frame_end":    inst.get("frame_end", -1),
            "url":          inst.get("url", ""),
        })

df_full = pd.DataFrame(rows)
print(f"\nTotal instances in full WLASL: {len(df_full)}")

# ── A2. WLASL100 instance counts ──────────────────────────────────────────────
gloss_counts     = df_full.groupby("gloss").size().sort_values(ascending=False)
wlasl100_glosses = gloss_counts.head(SUBSET)
df_100           = df_full[df_full["gloss"].isin(wlasl100_glosses.index)].copy()

print(f"\nWLASL100 — {SUBSET} glosses, {len(df_100)} total instances")
print(f"  Instances per gloss — min: {wlasl100_glosses.min()}, "
      f"max: {wlasl100_glosses.max()}, mean: {wlasl100_glosses.mean():.1f}")
print(f"\nTop 10 glosses:\n{wlasl100_glosses.head(10).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].bar(range(SUBSET), wlasl100_glosses.values, color="steelblue", width=1.0)
axes[0].set_xlabel("Gloss rank (0 = most instances)")
axes[0].set_ylabel("Number of video instances")
axes[0].set_title("WLASL100: instances per gloss")

axes[1].hist(wlasl100_glosses.values, bins=20, color="steelblue", edgecolor="white")
axes[1].set_xlabel("Instance count")
axes[1].set_ylabel("Number of glosses")
axes[1].set_title("Distribution of instance counts (WLASL100)")

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "eda_a2_instance_counts.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {out}")

# ── A3. Split distribution ────────────────────────────────────────────────────
split_counts = df_100["split"].value_counts()
print(f"\nSplit distribution (WLASL100 full):\n{split_counts.to_string()}")

fig, ax = plt.subplots(figsize=(7, 4))
colors = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}
bar_colors = [colors.get(s, "gray") for s in split_counts.index]
split_counts.plot.bar(ax=ax, color=bar_colors, edgecolor="white")
ax.set_title("WLASL100: train/val/test instance counts")
ax.set_xlabel("Split")
ax.set_ylabel("Number of instances")
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2, p.get_height() + 2),
                ha="center", fontsize=11)
plt.xticks(rotation=0)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "eda_a3_split_distribution.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── A4. Signer diversity ──────────────────────────────────────────────────────
signers_per_gloss = df_100.groupby("gloss")["signer_id"].nunique()
total_signers     = df_100["signer_id"].nunique()
single_signer     = (signers_per_gloss == 1).sum()

print(f"\nSigner diversity:")
print(f"  Total unique signers        : {total_signers}")
print(f"  Glosses with only 1 signer  : {single_signer}")
print(f"  Signers/gloss — min: {signers_per_gloss.min()}, "
      f"max: {signers_per_gloss.max()}, mean: {signers_per_gloss.mean():.1f}")

fig, ax = plt.subplots(figsize=(7, 4))
signers_per_gloss.hist(
    bins=range(1, int(signers_per_gloss.max()) + 2), ax=ax,
    color="steelblue", edgecolor="white"
)
ax.set_title("Unique signers per gloss (WLASL100)")
ax.set_xlabel("Number of unique signers")
ax.set_ylabel("Number of glosses")
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "eda_a4_signer_diversity.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── A5. Video duration ────────────────────────────────────────────────────────
df_100["duration_frames"] = df_100["frame_end"] - df_100["frame_start"] + 1
df_dur = df_100[df_100["duration_frames"] > 0].copy()
df_dur["duration_sec"] = df_dur["duration_frames"] / 25.0

print(f"\nVideo duration (frames @ 25 fps):")
print(df_dur["duration_frames"].describe().to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df_dur["duration_frames"], bins=40, color="coral", edgecolor="white")
axes[0].set_title("Duration distribution (frames)")
axes[0].set_xlabel("Frames")
axes[0].set_ylabel("Count")

axes[1].hist(df_dur["duration_sec"], bins=40, color="coral", edgecolor="white")
axes[1].set_title("Duration distribution (seconds, assumed 25 fps)")
axes[1].set_xlabel("Seconds")
axes[1].set_ylabel("Count")

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "eda_a5_duration_distribution.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── A6. Download coverage ─────────────────────────────────────────────────────
if not os.path.exists(AVAIL_JSON):
    print("\n  available_videos.json not found — skipping download coverage analysis.")
else:
    with open(AVAIL_JSON) as f:
        available = json.load(f)

    avail_ids = {r["video_id"] for r in available}
    total_ids = set(df_100["video_id"].tolist())
    failed_n  = len(total_ids - avail_ids)

    print(f"\nDownload coverage:")
    print(f"  WLASL100 total   : {len(total_ids)}")
    print(f"  Downloaded OK    : {len(avail_ids)}  ({100*len(avail_ids)/len(total_ids):.1f}%)")
    print(f"  Failed / missing : {failed_n}  ({100*failed_n/len(total_ids):.1f}%)")

    df_avail = pd.DataFrame(available)
    coverage = (df_avail.groupby("gloss").size() /
                df_100.groupby("gloss").size()).fillna(0)

    print(f"\n  Coverage per gloss — min: {coverage.min():.1%}, "
          f"max: {coverage.max():.1%}, mean: {coverage.mean():.1%}")

    low = coverage[coverage < 0.5].sort_values()
    if len(low):
        print(f"\n  Glosses with <50% coverage ({len(low)}):")
        for gloss, cov in low.items():
            print(f"    {gloss:25s}  {cov:.0%}")

    # Downloaded split counts
    avail_splits = defaultdict(int)
    for r in available:
        avail_splits[r["split"]] += 1
    print(f"\n  Downloaded split breakdown:")
    for split, cnt in sorted(avail_splits.items()):
        print(f"    {split:8s}: {cnt}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(coverage.values * 100, bins=20, color="#4CAF50", edgecolor="white")
    ax.set_xlabel("Download coverage (%)")
    ax.set_ylabel("Number of glosses")
    ax.set_title("Per-gloss download coverage (WLASL100)")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "eda_a6_download_coverage.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

print("\n" + "=" * 60)
print("  Phase A EDA complete — charts saved to results/")
print("  Next: run preprocessing.py, then eda_phase_b.py")
print("=" * 60)
