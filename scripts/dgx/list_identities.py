"""Run ON THE DGX with an env that has PIL. Lists every folder that directly holds images,
with its image count and median image size, so identity folders and suspicious names
(e.g. "father2dup", "son16.1") can be checked by hand.

    python list_identities.py /path/to/dataset
"""
import argparse
import statistics
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--sample", type=int, default=5, help="images per folder to measure")
    args = ap.parse_args()
    root = Path(args.root)
    folders = {}
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            folders.setdefault(f.parent, []).append(f)
    all_w, all_h = [], []
    print(f"{'folder':40s}{'images':>7s}{'median WxH':>14s}")
    for d in sorted(folders, key=lambda p: p.relative_to(root).as_posix().lower()):
        files = sorted(folders[d])
        sizes = []
        for f in files[:args.sample]:
            with Image.open(f) as im:
                sizes.append(im.size)
        w = statistics.median(s[0] for s in sizes)
        h = statistics.median(s[1] for s in sizes)
        all_w += [s[0] for s in sizes]
        all_h += [s[1] for s in sizes]
        print(f"{d.relative_to(root).as_posix():40s}{len(files):7d}{f'{w:.0f}x{h:.0f}':>14s}")
    print(f"{len(folders)} folders, {sum(map(len, folders.values()))} images; "
          f"overall median size {statistics.median(all_w):.0f}x{statistics.median(all_h):.0f}, "
          f"min {min(all_w)}x{min(all_h)}, max {max(all_w)}x{max(all_h)}")


if __name__ == "__main__":
    main()
