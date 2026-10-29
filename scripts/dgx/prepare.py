"""Run ONCE per dataset on the DGX login node before submitting jobs (needs internet for weights).

1. Builds the decoded-image cache next to the manifest, so array jobs don't all race to build it.
2. Checks that every split file references only images in the manifest.
3. --weights: downloads the study-v2 timm weights (benchmark/models.yaml) into the Hugging Face
   cache, because jobs run with HF_HUB_OFFLINE=1 (compute nodes may be offline).
   --v1-weights: the torchvision ResNet weights v1 used.

    python scripts/dgx/prepare.py --data configs/ami_dgx.yaml --weights
    python scripts/dgx/prepare.py --data configs/uerc.yaml --workers 8
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nlear.data.cache import load_image_cache  # noqa: E402
from nlear.data.manifest import load_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--workers", type=int, default=0, help="parallel decoding processes for the cache build")
ap.add_argument("--weights", action="store_true", help="download the benchmark/models.yaml timm weights")
ap.add_argument("--v1-weights", action="store_true", help="download v1's torchvision ResNet weights")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.data))
d = cfg["data"]
manifest = load_manifest(d["manifest"])
h, w = d["img_size"]
images, row_of = load_image_cache(manifest, d["root"], d["img_size"],
                                  Path(d["manifest"]).parent / f"cache_{h}x{w}.pt", workers=args.workers)
print(f"cache ok: {tuple(images.shape)}")
for seed in cfg["splits"]["seeds"]:
    df = pd.read_csv(d["splits"].format(seed=seed), dtype={"path": str})
    missing = (~df["path"].isin(row_of)).sum()
    assert missing == 0, f"seed {seed}: {missing} split rows not in the manifest"
print("splits ok")
if args.weights:
    import timm
    for tag in yaml.safe_load(open(ROOT / "benchmark" / "models.yaml"))["models"].values():
        timm.create_model(tag, pretrained=True, num_classes=0)
        print(f"weights cached: {tag}")
if args.v1_weights:
    import torchvision
    for name, w_ in (("resnet50", "IMAGENET1K_V2"), ("resnet18", "IMAGENET1K_V1")):
        getattr(torchvision.models, name)(weights=w_)
    print("v1 torchvision weights cached")
