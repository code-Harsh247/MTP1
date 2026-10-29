"""Run ON THE DGX. Python standard library only (PIL optional), so any env works.

Prints the on-disk layout of every dataset under Ear_Datasets so we can write the
manifest rules (folder-per-person or id-in-file-name) and verify subject/image counts.

    python3 inspect_datasets.py "/dgx1/dsamantaai/santhosh/Ear_Datasets" > datasets_report.txt
    python3 inspect_datasets.py "/dgx1/.../Ear_Datasets" --only AMI EarVN AWE IITD
"""
import argparse
import random
import statistics
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_DIRS = {"UERC"}  # UERC is large; inspected separately when we reach stage S4


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def tree(d, depth=0, max_depth=3, max_entries=6):
    """Print folders (with image counts) and a few sample files, a few levels deep."""
    entries = sorted(d.iterdir())
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    pad = "    " * depth
    if files:
        print(f"{pad}[{len(files)} files] e.g. " + ", ".join(f.name for f in files[:4]))
    for sub in dirs[:max_entries]:
        n = sum(1 for f in sub.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        print(f"{pad}{sub.name}/  ({n} images)")
        if depth + 1 < max_depth:
            tree(sub, depth + 1, max_depth, max_entries)
    if len(dirs) > max_entries:
        print(f"{pad}... and {len(dirs) - max_entries} more folders")


TEXT_EXTS = {".txt", ".csv", ".json", ".xml", ".md", ".tsv", ".yaml", ".yml"}


def preview_text_files(ds, max_files=4, max_lines=12):
    """First lines of small annotation/readme files: they often say how identities are encoded."""
    texts = sorted(f for f in ds.rglob("*") if f.is_file() and f.suffix.lower() in TEXT_EXTS
                   and f.stat().st_size < 5_000_000)
    for f in texts[:max_files]:
        print(f"--- {f.relative_to(ds).as_posix()} ({human(f.stat().st_size)}), first lines:")
        with open(f, errors="replace") as fh:
            for _, line in zip(range(max_lines), fh):
                print("    " + line.rstrip()[:200])
    if len(texts) > max_files:
        print(f"--- ... and {len(texts) - max_files} more text files")


def inspect(ds):
    imgs = [f for f in ds.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    others = Counter(f.suffix.lower() for f in ds.rglob("*")
                     if f.is_file() and f.suffix.lower() not in IMAGE_EXTS)
    total = sum(f.stat().st_size for f in imgs)
    print("=" * 80)
    print(f"{ds.name}:  {len(imgs)} images, {human(total)}")
    print(f"image extensions: {dict(Counter(f.suffix.lower() for f in imgs))}")
    if others:
        print(f"other files: {dict(others.most_common(8))}")
        preview_text_files(ds)
    tree(ds)
    if imgs:
        print("random sample of image paths:")
        for f in random.Random(0).sample(imgs, min(12, len(imgs))):
            print("   ", f.relative_to(ds).as_posix())
        parents = Counter(f.parent for f in imgs)
        print(f"folders that directly contain images: {len(parents)}; "
              f"images per such folder: median {statistics.median(parents.values())}")
        try:
            from PIL import Image
            sizes, modes = [], Counter()
            for f in random.Random(1).sample(imgs, min(100, len(imgs))):
                with Image.open(f) as im:
                    sizes.append(im.size)
                    modes[im.mode] += 1
            print(f"image size (100 sampled): width median {statistics.median(w for w, _ in sizes)}, "
                  f"height median {statistics.median(h for _, h in sizes)}; modes {dict(modes)}")
        except ImportError:
            print("PIL not available: skipped image-size stats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--only", nargs="*", help="case-insensitive substrings of dataset folder names")
    args = ap.parse_args()
    root = Path(args.root)
    print("top level of", root)
    for e in sorted(root.iterdir()):
        print("   ", e.name + ("/" if e.is_dir() else ""))
    for ds in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(s.lower() in ds.name.lower() for s in SKIP_DIRS):
            continue
        if args.only and not any(s.lower() in ds.name.lower() for s in args.only):
            continue
        inspect(ds)


if __name__ == "__main__":
    main()
