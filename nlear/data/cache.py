"""Decode a whole (small) dataset into one uint8 tensor once, then augment on the GPU.

DataLoader worker processes were very slow on the Windows laptop (see notes/phase0_data.md),
and the small datasets fit in RAM easily (EarVN at 128x96 is ~0.7 GB), so we skip them.
"""
import functools
import math
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def _decode(path, root, h, w):
    """One image -> uint8 HxWx3 array (top-level so worker processes can pickle it)."""
    with Image.open(Path(root) / path) as im:
        return np.asarray(im.convert("RGB").resize((w, h), Image.BILINEAR)).copy()


def load_image_cache(manifest, root, img_size, cache_path, workers=0):
    """Return (images uint8 [N,3,H,W], path -> row index). Built once and saved to cache_path.

    The saved cache is loaded memory-mapped, so a large cache (UERC: 9.2 GB) is not copied
    into every job's RAM and jobs on one node share the OS page cache. workers > 0 decodes
    in parallel processes when building (UERC: ~250k images).
    """
    cache_path = Path(cache_path)
    paths = manifest["path"].tolist()
    if cache_path.exists():
        blob = torch.load(cache_path, mmap=True)
        if blob["paths"] == paths and tuple(blob["img_size"]) == tuple(img_size):
            return blob["images"], {p: i for i, p in enumerate(paths)}
        print(f"cache {cache_path} is stale (manifest or size changed); rebuilding")
    h, w = img_size
    images = torch.empty((len(paths), 3, h, w), dtype=torch.uint8)
    decode = functools.partial(_decode, root=str(root), h=h, w=w)
    desc = f"decoding {cache_path.parent.name}"
    if workers > 0:
        with multiprocessing.get_context("spawn" if os.name == "nt" else "fork").Pool(workers) as pool:
            it = pool.imap(decode, paths, chunksize=256)
            for i, arr in enumerate(tqdm(it, total=len(paths), desc=desc, mininterval=5)):
                images[i] = torch.from_numpy(arr).permute(2, 0, 1)
    else:
        for i, p in enumerate(tqdm(paths, desc=desc, mininterval=2)):
            images[i] = torch.from_numpy(decode(p)).permute(2, 0, 1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"paths": paths, "img_size": list(img_size), "images": images}, cache_path)
    return images, {p: i for i, p in enumerate(paths)}


class GPUAugment:
    """Random rotation/scale/translation + brightness/contrast jitter, batched on the GPU.

    Mirrors build_transform(train=True). No horizontal flip (left vs right ears).
    """

    def __init__(self, degrees=10.0, translate=0.05, scale=(0.9, 1.1), jitter=0.2):
        self.degrees, self.translate, self.scale, self.jitter = degrees, translate, scale, jitter

    def __call__(self, x):  # x: float [B,3,H,W] in [0,1]
        b, dev = x.shape[0], x.device
        ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(self.degrees)
        s = torch.empty(b, device=dev).uniform_(*self.scale)
        t = (torch.rand(b, 2, device=dev) * 2 - 1) * self.translate * 2
        cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
        theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1),
                             torch.stack([sin, cos, t[:, 1]], 1)], 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
        j = self.jitter
        bright = torch.empty(b, 1, 1, 1, device=dev).uniform_(1 - j, 1 + j)
        contrast = torch.empty(b, 1, 1, 1, device=dev).uniform_(1 - j, 1 + j)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mean) * contrast + mean) * bright
        return x.clamp(0, 1)


def normalize(x):
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def to_input(batch_u8, device, augment=None):
    """uint8 [B,3,H,W] on CPU -> normalized float on device (augmented if given)."""
    x = batch_u8.to(device, non_blocking=True).float().div_(255)
    if augment is not None:
        x = augment(x)
    return normalize(x)
