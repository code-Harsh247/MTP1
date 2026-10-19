"""Run ONCE on the DGX login node before submitting jobs (needs internet for step 2).

1. Builds the decoded-image cache next to the manifest, so array jobs don't all race to build it.
2. Downloads the ImageNet ResNet weights into the torch cache (compute nodes may be offline).
3. Checks that every split file references only images in the manifest.

    python scripts/dgx/prepare.py --data configs/earvn_dgx.yaml
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torchvision
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nlear.data.cache import load_image_cache  # noqa: E402
from nlear.data.manifest import load_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
args = ap.parse_args()
cfg = yaml.safe_load(open(args.data))
d = cfg["data"]
manifest = load_manifest(d["manifest"])
h, w = d["img_size"]
images, row_of = load_image_cache(manifest, d["root"], d["img_size"], Path(d["manifest"]).parent / f"cache_{h}x{w}.pt")
print(f"cache ok: {tuple(images.shape)}")
for seed in cfg["splits"]["seeds"]:
    df = pd.read_csv(d["splits"].format(seed=seed), dtype={"path": str})
    missing = (~df["path"].isin(row_of)).sum()
    assert missing == 0, f"seed {seed}: {missing} split rows not in the manifest"
print("splits ok")
for name, w_ in (("resnet50", "IMAGENET1K_V2"), ("resnet18", "IMAGENET1K_V1")):
    getattr(torchvision.models, name)(weights=w_)
print("pretrained weights cached")
