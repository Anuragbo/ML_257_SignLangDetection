"""
Download the ASL image dataset from Kaggle if it is missing or empty.

Default dataset: ayuraj/asl-dataset (American Sign Language — class folders per letter).

Requires Kaggle API credentials:
  ~/.kaggle/kaggle.json   (Linux / macOS)
  %USERPROFILE%\\.kaggle\\kaggle.json   (Windows)

Create the file from https://www.kaggle.com/settings → API → "Create New Token".
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Same dataset referenced in part1_letter_classifier/notebooks/01_EDA.ipynb
KAGGLE_DATASET_SLUG = "ayuraj/asl-dataset"

# Inside a downloaded archive, require this many class folders to pick the real root
# (avoids matching random nested folders).
MIN_CLASS_FOLDERS_TO_DETECT_ROOT = 15

# Local folder is "ready" if there is at least one class subfolder (do not re-download).
MIN_CLASS_FOLDERS_TO_SKIP_DOWNLOAD = 1

# Unzip nested archives until none left (some Kaggle exports ship zips inside zips)
_MAX_UNZIP_ROUNDS = 8


def _count_class_children(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(
        1
        for c in d.iterdir()
        if c.is_dir() and len(c.name) == 1 and c.name.isalnum()
    )


def find_class_image_root(search_under: Path) -> Path | None:
    """
    Find a directory whose *direct* subfolders are mostly single-letter/digit class names.
    """
    best: Path | None = None
    best_n = 0
    for dirpath, _, _ in os.walk(search_under):
        p = Path(dirpath)
        n = _count_class_children(p)
        if n > best_n:
            best_n = n
            best = p
    if best is not None and best_n >= MIN_CLASS_FOLDERS_TO_DETECT_ROOT:
        return best
    return None


def dataset_is_ready(data_asl: Path) -> bool:
    if not data_asl.is_dir():
        return False
    return _count_class_children(data_asl) >= MIN_CLASS_FOLDERS_TO_SKIP_DOWNLOAD


def _kaggle_json_hint() -> str:
    loc = Path.home() / ".kaggle" / "kaggle.json"
    return (
        "Kaggle API credentials are required for automatic download.\n\n"
        "  1. Open https://www.kaggle.com/settings  →  API  →  Create New Token\n"
        "  2. Save the downloaded file as:\n"
        f"       {loc}\n\n"
        "  3. Accept the dataset terms on the dataset’s Kaggle page if prompted.\n"
        "  4. Run again.\n"
    )


def _ensure_kaggle_credentials_file() -> None:
    if not (Path.home() / ".kaggle" / "kaggle.json").is_file():
        print(_kaggle_json_hint(), file=sys.stderr)
        raise SystemExit(1)


def _run_kaggle_download(staging: Path, dataset: str) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(staging),
    ]
    print("\n[Dataset] Downloading from Kaggle (progress from the Kaggle CLI below)…\n", flush=True)
    print("Command:", " ".join(cmd), "\n", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(_kaggle_json_hint(), file=sys.stderr)
        raise RuntimeError(
            "kaggle datasets download failed. "
            "Verify credentials (see README), run: pip install kaggle, "
            "and open the dataset page on Kaggle to accept its rules if required."
        )


def _extract_all_zips_under(folder: Path) -> None:
    """Extract every .zip under folder (in-place), then delete the archive. Repeat until stable."""
    for _ in range(_MAX_UNZIP_ROUNDS):
        zips = sorted(folder.rglob("*.zip"))
        if not zips:
            return
        for zpath in zips:
            rel = zpath.relative_to(folder)
            print(f"[Dataset] Extracting archive: {rel} …", flush=True)
            with zipfile.ZipFile(zpath, "r") as zf:
                members = zf.infolist()
                total = len(members)
                for i, m in enumerate(members, 1):
                    zf.extract(m, zpath.parent)
                    if total > 200 and (i % 400 == 0 or i == total):
                        print(f"  … extracted {i}/{total} files", flush=True)
            try:
                zpath.unlink()
            except OSError:
                pass


def _relocate_to_asl_dataset(found_root: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(found_root), str(target))


def ensure_dataset(
    data_asl: Path,
    *,
    dataset_slug: str = KAGGLE_DATASET_SLUG,
    no_download: bool = False,
) -> None:
    """
    If ``data_asl`` already contains enough class folders, do nothing.
    Otherwise download from Kaggle, unpack, detect layout, and move to ``data_asl``.
    """
    if dataset_is_ready(data_asl):
        print(f"[Dataset] OK — using existing data at:\n  {data_asl}\n", flush=True)
        return

    if no_download:
        print(
            f"[Dataset] Missing or incomplete under:\n  {data_asl}\n",
            file=sys.stderr,
        )
        raise SystemExit(
            "Re-run without --no-download to fetch from Kaggle, "
            "or place the ASL images there manually."
        )

    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise SystemExit(
            "The 'kaggle' package is required for automatic download.\n"
            "  pip install kaggle\n"
        ) from None

    _ensure_kaggle_credentials_file()

    staging = data_asl.parent / "_kaggle_staging"
    if staging.exists():
        print(f"[Dataset] Removing old staging folder:\n  {staging}", flush=True)
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(
        f"[Dataset] Data not found or incomplete at:\n  {data_asl}\n\n"
        f"[Dataset] Downloading Kaggle dataset: {dataset_slug}\n",
        flush=True,
    )

    _run_kaggle_download(staging, dataset_slug)
    _extract_all_zips_under(staging)

    found = find_class_image_root(staging)
    if found is None:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            "Downloaded files did not contain a recognizable ASL layout "
            f"(expected at least {MIN_CLASS_FOLDERS_TO_DETECT_ROOT} single-character class folders). "
            "Unpack manually into part1_letter_classifier/data/asl_dataset/ or check the dataset on Kaggle."
        )

    print(f"[Dataset] Detected class image root:\n  {found}\n", flush=True)
    print(f"[Dataset] Installing to:\n  {data_asl}\n", flush=True)
    _relocate_to_asl_dataset(found, data_asl)

    # Remove staging unless it was renamed by move (found == staging root)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)

    if not dataset_is_ready(data_asl):
        raise RuntimeError("After download, dataset layout still looks wrong.")

    print("[Dataset] Download and install complete.\n", flush=True)


def main() -> None:
    from argparse import ArgumentParser

    repo = Path(__file__).resolve().parent
    default_asl = repo / "part1_letter_classifier" / "data" / "asl_dataset"

    p = ArgumentParser(description="Download ASL dataset from Kaggle if needed.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=default_asl,
        help="Target directory for class folders (default: part1_letter_classifier/data/asl_dataset)",
    )
    p.add_argument(
        "-d",
        "--dataset",
        default=KAGGLE_DATASET_SLUG,
        help=f"Kaggle dataset slug (default: {KAGGLE_DATASET_SLUG})",
    )
    args = p.parse_args()

    ensure_dataset(args.data_dir, dataset_slug=args.dataset, no_download=False)


if __name__ == "__main__":
    main()
