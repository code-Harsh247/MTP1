"""Build the two zips to upload as PRIVATE Kaggle datasets:

    python scripts/kaggle/package.py                    # code and data
    python scripts/kaggle/package.py --code-only        # after a code change

nlear_code.zip holds only what the runs need. nlear_data.zip holds, per dataset, the manifest, the
frozen split and task files of its benchmark seeds (their hashes are checked here first) and the
128x96 image cache, which still contains the ear images. Nothing is uploaded by this script.
"""
import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kaggle_upload"
CODE_DIRS = ("nlear", "scripts", "configs", "benchmark")


def package_code():
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", *CODE_DIRS],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    with zipfile.ZipFile(OUT / "nlear_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in listed:
            if (ROOT / name).is_file():
                z.write(ROOT / name, name)
    print(f"nlear_code.zip: {len(listed)} files")


def package_data(names):
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
    with zipfile.ZipFile(OUT / "nlear_data.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            folder = ROOT / "data" / name
            pinned = dict(reversed(ln.split()) for ln in
                          (ROOT / "benchmark" / f"hashes_{name}.txt").read_text().splitlines())
            files = [folder / "manifest.csv", folder / "cache_128x96.pt"]
            for seed in bench[name]["seeds"]:
                for kind in ("splits", "tasks"):
                    f = folder / f"{kind}_seed{seed}.csv"
                    if hashlib.md5(f.read_bytes()).hexdigest() != pinned[f.name]:
                        raise ValueError(f"{f} differs from its frozen hash")
                    files.append(f)
            for f in files:
                z.write(f, f"{name}/{f.name}")
            print(f"{name}: {len(files)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="ami,awe,iitd,earvn")
    ap.add_argument("--code-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    package_code()
    if not args.code_only:
        package_data(args.only.split(","))
    for f in sorted(OUT.glob("*.zip")):
        print(f"{f}: {f.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
