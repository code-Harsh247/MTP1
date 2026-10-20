"""Run ON THE DGX (U0 in plan_uerc.md). Reads only; prints what we need to build UERC splits:
the package layout, any metadata files (to find how the 3 source datasets are encoded),
identity/image counts, file-name patterns and image sizes.

    PY=/home/dsamantaai/miniconda3/envs/harsh_nlear/bin/python
    $PY scripts/dgx/inspect_uerc.py "/dgx1/dsamantaai/santhosh/Ear_Datasets/UERC 2023 Competition Package with sequestered anonymized data" > uerc_report.txt 2>&1
"""
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
META_EXTS = {".csv", ".txt", ".json", ".tsv", ".md", ".yaml", ".yml", ".xml"}


def show_tree(d, depth=0, max_depth=2, max_entries=8):
    entries = sorted(d.iterdir())
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    pad = "    " * depth
    if files:
        print(f"{pad}[{len(files)} files] e.g. " + ", ".join(f.name for f in files[:5]))
    for sub in dirs[:max_entries]:
        print(f"{pad}{sub.name}/")
        if depth + 1 < max_depth:
            show_tree(sub, depth + 1, max_depth, max_entries)
    if len(dirs) > max_entries:
        print(f"{pad}... and {len(dirs) - max_entries} more folders")


def main():
    pkg = Path(sys.argv[1])
    print(f"== package root: {pkg}")
    show_tree(pkg, max_depth=3)

    print("\n== metadata-like files (outside public/ image folders), first lines")
    for f in sorted(pkg.rglob("*")):
        if f.is_file() and f.suffix.lower() in META_EXTS and "public" not in f.parts[-3:-1]:
            print(f"--- {f.relative_to(pkg)}  ({f.stat().st_size} bytes)")
            try:
                with open(f, errors="replace") as fh:
                    for _ in range(6):
                        line = fh.readline()
                        if not line:
                            break
                        print("    " + line.rstrip()[:200])
            except OSError as e:
                print(f"    (unreadable: {e})")

    pub = next((p for p in pkg.rglob("public") if p.is_dir()), None)
    if pub is None:
        print("\nno public/ folder found")
        return
    print(f"\n== public/ = {pub}")
    ids = sorted(p for p in pub.iterdir() if p.is_dir())
    per_id, exts, sizes_b, samples, other = Counter(), Counter(), [], [], Counter()
    for d in ids:
        for f in d.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() in IMAGE_EXTS:
                per_id[d.name] += 1
                exts[f.suffix.lower()] += 1
                if len(sizes_b) < 20000:
                    sizes_b.append(f.stat().st_size)
                samples.append(f)
            else:
                other[f.suffix.lower()] += 1
    counts = sorted(per_id.values())
    print(f"identities: {len(ids)}   images: {sum(counts)}   extensions: {dict(exts)}")
    print(f"non-image files inside public/: {dict(other)}")
    print(f"images/identity: min {counts[0]}  p10 {counts[len(counts)//10]}  median {statistics.median(counts)}"
          f"  p90 {counts[9*len(counts)//10]}  max {counts[-1]}")
    for k in (4, 6, 10, 20, 50):
        print(f"  identities with >= {k:2d} images: {sum(c >= k for c in counts)}")
    print(f"mean file size: {statistics.mean(sizes_b) / 1024:.1f} KB  (RAM cache at 128x96: "
          f"{sum(counts) * 128 * 96 * 3 / 1e9:.1f} GB)")
    print("\nidentity folder names (first 10, last 5):", [d.name for d in ids[:10]], [d.name for d in ids[-5:]])
    print("random image paths (look for a source tag in names/subfolders):")
    for f in random.Random(0).sample(samples, min(25, len(samples))):
        print("   ", f.relative_to(pub).as_posix())
    depth = Counter(len(f.relative_to(pub).parts) for f in samples)
    print(f"path depth (parts below public/): {dict(depth)}")
    try:
        from PIL import Image
        sz, modes = [], Counter()
        for f in random.Random(1).sample(samples, min(500, len(samples))):
            with Image.open(f) as im:
                sz.append(im.size)
                modes[im.mode] += 1
        ws, hs = sorted(w for w, _ in sz), sorted(h for _, h in sz)
        print(f"image size (500 sampled): width p5/med/p95 {ws[len(ws)//20]}/{statistics.median(ws)}/{ws[-len(ws)//20]}"
              f"  height {hs[len(hs)//20]}/{statistics.median(hs)}/{hs[-len(hs)//20]}  modes {dict(modes)}")
    except ImportError:
        print("PIL not available (run with the harsh_nlear python)")


if __name__ == "__main__":
    main()
