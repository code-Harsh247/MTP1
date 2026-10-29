"""Run ON THE DGX (standard library only). Finds byte-identical image files across the folders
of one dataset and prints which folders share them, e.g. a person folder copied under a
second name ("father2" and "father2dup").

    python3 find_dup_files.py /path/to/dataset
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    root = Path(ap.parse_args().root)
    by_hash = defaultdict(list)
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            by_hash[hashlib.md5(f.read_bytes()).hexdigest()].append(f)
    n_files = sum(map(len, by_hash.values()))
    folder_files = defaultdict(int)
    for files in by_hash.values():
        for f in files:
            folder_files[f.parent] += 1
    shared = defaultdict(int)          # (folder a, folder b) -> number of identical files
    within = defaultdict(int)          # folder -> identical files inside the same folder
    for files in by_hash.values():
        folders = sorted({f.parent for f in files}, key=lambda p: p.as_posix())
        if len(files) > len(folders):
            for d in folders:
                n = sum(1 for f in files if f.parent == d)
                if n > 1:
                    within[d] += n - 1
        for i, a in enumerate(folders):
            for b in folders[i + 1:]:
                shared[(a, b)] += 1
    print(f"{n_files} image files, {len(by_hash)} distinct contents")
    print("folder pairs sharing identical files (shared / size of each folder):")
    for (a, b), n in sorted(shared.items(), key=lambda kv: -kv[1]):
        print(f"  {a.relative_to(root).as_posix():30s} {b.relative_to(root).as_posix():30s} "
              f"{n:4d}  ({folder_files[a]} / {folder_files[b]})")
    if not shared:
        print("  none")
    for d, n in within.items():
        print(f"  duplicates inside {d.relative_to(root).as_posix()}: {n}")


if __name__ == "__main__":
    main()
