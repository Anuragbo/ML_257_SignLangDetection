"""
download_dataset.py
-------------------
Downloads WLASL videos from the WLASL_v0.3.json metadata file.

Filters to the top N glosses (by instance count) to form WLASL100/300/etc.
Saves successfully downloaded MP4s to data/videos/{video_id}.mp4.
Logs failures to data/failed_downloads.txt.
Writes data/available_videos.json for use by preprocessing.py.

Usage:
    python download_dataset.py
    python download_dataset.py --json ../data/WLASL_v0.3.json --subset 100
    python download_dataset.py --json ../data/WLASL_v0.3.json --subset 300 --output_dir ../data/videos
"""

import os
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_JSON   = os.path.join(os.path.dirname(__file__), "..", "data", "WLASL_v0.3.json")
DEFAULT_OUT    = os.path.join(os.path.dirname(__file__), "..", "data", "videos")
FAILED_LOG     = os.path.join(os.path.dirname(__file__), "..", "data", "failed_downloads.txt")
AVAILABLE_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "available_videos.json")

TIMEOUT   = 10  # seconds per attempt (reduced from 20)
RETRIES   = 1   # single attempt — dead URLs waste time on retries
WORKERS   = 16  # parallel download threads


# ── helpers ──────────────────────────────────────────────────────────────────

def load_wlasl_json(json_path: str, subset_size: int = 100):
    """
    Parse WLASL JSON and return a flat list of video records.

    Each record:
        {gloss, label_idx, video_id, url, split, frame_start, frame_end, signer_id}

    Selects the top `subset_size` glosses ordered by total instance count
    (most-resourced classes first, to maximise available videos).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Sort by number of instances descending → take top subset_size
    data_sorted = sorted(data, key=lambda x: len(x.get("instances", [])), reverse=True)
    data_subset = data_sorted[:subset_size]

    records = []
    for label_idx, entry in enumerate(data_subset):
        gloss = entry["gloss"]
        for inst in entry.get("instances", []):
            records.append({
                "gloss":       gloss,
                "label_idx":   label_idx,
                "video_id":    str(inst.get("video_id", "")).zfill(5),
                "url":         inst.get("url", ""),
                "split":       inst.get("split", "train"),
                "frame_start": inst.get("frame_start", 1),
                "frame_end":   inst.get("frame_end", -1),
                "signer_id":   inst.get("signer_id", -1),
            })

    print(f"Loaded {len(data_subset)} glosses, {len(records)} total video records.")
    return records, data_subset


def download_video(url: str, video_id: str, output_dir: str) -> bool:
    """
    Download a single video. Returns True on success, False on failure.
    Skips if the file already exists with a reasonable size.
    """
    if not url:
        return False

    dest = os.path.join(output_dir, f"{video_id}.mp4")
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return True   # already downloaded

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                content = resp.read()
            if len(content) < 1024:
                return False
            with open(dest, "wb") as f:
                f.write(content)
            return True
        except Exception:
            pass

    return False


def _download_task(rec: dict, output_dir: str):
    """Wrapper for parallel execution — returns (rec, success)."""
    ok = download_video(rec["url"], rec["video_id"], output_dir)
    return rec, ok


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download WLASL videos")
    parser.add_argument("--json",       default=DEFAULT_JSON,  help="Path to WLASL_v0.3.json")
    parser.add_argument("--output_dir", default=DEFAULT_OUT,   help="Where to save MP4 files")
    parser.add_argument("--subset",     type=int, default=100, help="Number of glosses (100, 300, …)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(FAILED_LOG), exist_ok=True)

    # ── load metadata ────────────────────────────────────────────────────────
    records, gloss_data = load_wlasl_json(args.json, args.subset)

    # ── parallel download loop ────────────────────────────────────────────────
    successful = []
    failed_ids = []

    print(f"Downloading with {WORKERS} parallel workers (timeout={TIMEOUT}s each)...")
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_download_task, rec, args.output_dir): rec
            for rec in records
        }
        for future in tqdm(as_completed(futures), total=len(records),
                           desc="Downloading videos", unit="video"):
            rec, ok = future.result()
            if ok:
                successful.append(rec)
            else:
                failed_ids.append(rec["video_id"])

    # ── write failure log ─────────────────────────────────────────────────────
    with open(FAILED_LOG, "w") as f:
        for vid in failed_ids:
            f.write(vid + "\n")

    # ── write available_videos.json ───────────────────────────────────────────
    with open(AVAILABLE_JSON, "w") as f:
        json.dump(successful, f, indent=2)

    # ── summary ───────────────────────────────────────────────────────────────
    total   = len(records)
    n_ok    = len(successful)
    n_fail  = len(failed_ids)

    print("\n" + "=" * 50)
    print(f"  Total records   : {total}")
    print(f"  Downloaded OK   : {n_ok}  ({100*n_ok/total:.1f}%)")
    print(f"  Failed / skipped: {n_fail}  ({100*n_fail/total:.1f}%)")
    print(f"  Saved to        : {args.output_dir}")
    print(f"  Failure log     : {FAILED_LOG}")
    print(f"  Available JSON  : {AVAILABLE_JSON}")

    # Per-split breakdown
    split_counts = defaultdict(int)
    for rec in successful:
        split_counts[rec["split"]] += 1
    print("\n  Split breakdown (successful downloads):")
    for split, cnt in sorted(split_counts.items()):
        print(f"    {split:10s}: {cnt}")

    # Per-gloss coverage
    gloss_total   = defaultdict(int)
    gloss_success = defaultdict(int)
    for rec in records:
        gloss_total[rec["gloss"]] += 1
    for rec in successful:
        gloss_success[rec["gloss"]] += 1

    low_coverage = [
        (g, gloss_success[g], gloss_total[g])
        for g in gloss_total
        if gloss_success[g] < 3
    ]
    if low_coverage:
        print(f"\n  WARNING: {len(low_coverage)} glosses have fewer than 3 downloaded videos:")
        for g, ok, tot in sorted(low_coverage, key=lambda x: x[1]):
            print(f"    {g:20s}  {ok}/{tot}")

    print("=" * 50)


if __name__ == "__main__":
    main()
