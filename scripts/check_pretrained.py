"""R0: do the ImageNet-1k weights of every model family load at 128x96, and are they sane?

For each model: build it with pretrained weights at the ear input size (timm resamples
ViT position embeddings and Swin attention-bias tables), then embed every AMI image with
the frozen backbone and report leave-one-out rank-1 (cosine nearest neighbour, closed set).
Random weights give a few percent; working pretrained weights give far more.

    .venv/Scripts/python scripts/check_pretrained.py
"""
import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps

# All ImageNet-1k only, so no family gets extra pretraining data.
WEIGHTS = {
    "classic CNN": "resnet50.tv2_in1k",          # same weights as torchvision IMAGENET1K_V2 (v1)
    "modern CNN": "convnext_tiny.fb_in1k",
    "lightweight CNN": "mobilenetv3_large_100.ra_in1k",
    "plain transformer": "deit_small_patch16_224.fb_in1k",
    "hierarchical transformer": "swin_tiny_patch4_window7_224.ms_in1k",
}
NEEDS_IMG_SIZE = ("deit", "vit", "swin")


def load_images(root, size):
    paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg", ".png", ".bmp"})
    ids = [p.parent.name for p in paths]
    imgs = [np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                       .resize((size[1], size[0]), Image.BILINEAR), dtype=np.float32) / 255.0 for p in paths]
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2), np.array(ids)


@torch.no_grad()
def loo_rank1(feats, ids):
    f = torch.nn.functional.normalize(feats, dim=1)
    s = f @ f.T
    s.fill_diagonal_(-2)
    return float((ids[s.argmax(1).numpy()] == ids).mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ami")
    ap.add_argument("--size", default="128x96")
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, ids = load_images(args.data, size)
    print(f"{args.data}: {len(ids)} images, {len(set(ids))} ids, input {size[0]}x{size[1]}")
    print(f"{'family':26s}{'weights':40s}{'feat':>6s}{'LOO rank-1':>12s}{'random':>9s}")
    for family, tag in WEIGHTS.items():
        name = tag.split(".")[0]
        kw = {"img_size": size} if name.startswith(NEEDS_IMG_SIZE) else {}
        out = []
        for pretrained in (True, False):
            torch.manual_seed(0)
            net = timm.create_model(tag, pretrained=pretrained, num_classes=0, **kw).to(device).eval()
            cfg = net.pretrained_cfg
            mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
            std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
            feats = torch.cat([net(((xb - mean) / std).to(device)).float().cpu()
                               for xb in x.split(32)])
            assert torch.isfinite(feats).all(), f"{tag}: non-finite features"
            out.append(loo_rank1(feats, ids))
            del net
            if device == "cuda":
                torch.cuda.empty_cache()
        print(f"{family:26s}{tag:40s}{feats.shape[1]:6d}{out[0]:12.3f}{out[1]:9.3f}")


if __name__ == "__main__":
    main()
