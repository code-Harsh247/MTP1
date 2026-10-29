"""R0: can every model family of plan.md train at a common ear-shaped input size?

Builds each timm backbone WITHOUT pretrained weights (only shapes matter here), runs a
forward + backward pass in fp32 at each candidate size, and prints the feature size,
parameter count, peak GPU memory and time per step. Laptop numbers are only relative.

    .venv/Scripts/python scripts/check_archs.py
    .venv/Scripts/python scripts/check_archs.py --sizes 128x96 224x160 --batch 32
"""
import argparse
import time

import timm
import torch

ARCHS = {
    "classic CNN": "resnet50",
    "modern CNN": "convnext_tiny",
    "lightweight CNN": "mobilenetv3_large_100",
    "plain transformer": "deit_small_patch16_224",
    "hierarchical transformer": "swin_tiny_patch4_window7_224",
}
# timm needs the input size at build time for models with position embeddings / windows.
NEEDS_IMG_SIZE = ("deit", "vit", "swin")


def build(name, size):
    kw = {"img_size": size} if name.startswith(NEEDS_IMG_SIZE) else {}
    return timm.create_model(name, pretrained=False, num_classes=0, **kw)


def step_stats(net, size, batch, device, reps=3):
    net.to(device).train()
    x = torch.randn(batch, 3, *size, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps + 1):
        t0 = time.perf_counter()
        out = net(x)
        out.float().pow(2).mean().backward()
        net.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
    return out.shape[1], sorted(times[1:])[len(times[1:]) // 2], mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["128x96", "224x160", "224x224"],
                    help="HxW candidates (ears are ~1.4x taller than wide)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes]
    print(f"device {device}, batch {args.batch}, fp32, timm {timm.__version__}")
    print(f"{'family':26s}{'model':32s}{'size':>9s}{'params':>9s}{'feat':>6s}{'s/step':>8s}{'GB':>6s}  status")
    for family, name in ARCHS.items():
        for size in sizes:
            tag = f"{size[0]}x{size[1]}"
            try:
                net = build(name, size)
                n_par = sum(p.numel() for p in net.parameters()) / 1e6
                feat, sec, mem = step_stats(net, size, args.batch, device)
                print(f"{family:26s}{name:32s}{tag:>9s}{n_par:8.1f}M{feat:6d}{sec:8.3f}{mem:6.2f}  ok")
            except Exception as e:  # report and keep going: incompatibility is the finding
                msg = str(e).splitlines()[0][:70]
                print(f"{family:26s}{name:32s}{tag:>9s}{'':9s}{'':6s}{'':8s}{'':6s}  FAIL: {type(e).__name__}: {msg}")
            finally:
                net = None
                if device == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
