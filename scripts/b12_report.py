"""Apply the Option-B decision rules (notes/b_vit_cms.md) for B1 (static) and B2 (continual).

    python scripts/b12_report.py runs_dgx/uerc
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = [1, 2, 3]


def static(root, tag, key="cosine_proto", metric="dir@far0.05"):
    return np.array([json.load(open(root / f"baseline_deit_small_patch16_224_{tag}/seed{s}/metrics.json"))
                     ["results"][key]["test"][metric] for s in SEEDS])


def resnet(root, key="cosine_proto", metric="dir@far0.05"):
    return np.array([json.load(open(root / f"baseline_resnet50_e4/seed{s}/metrics.json"))
                     ["results"][key]["test"][metric] for s in SEEDS])


def cont(root, folder, key):
    out = []
    for s in SEEDS:
        m = json.load(open(root / "continual" / folder / f"seed{s}" / "metrics.json"))["summary"]
        out.append(m["final_U_test"][key[2:]] if key.startswith("U:") else m[key])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    root = Path(ap.parse_args().root)

    print(f"B1 STATIC, test, seeds {SEEDS}, cosine_proto")
    for var, title in (("", "2 gallery"), ("@g5", "5 gallery")):
        print(f"  [{title}]")
        print(f"  {'arm':22s}" + "".join(f"{m:>15s}" for m in ("rank1", "auroc", "dir@far0.05", "eer")))
        for label, fn in (("ResNet-50 (U1, AMP)", lambda m: resnet(root, "cosine_proto" + var, m)),
                          ("ViT (fp32)", lambda m: static(root, "vit", "cosine_proto" + var, m)),
                          ("CMS-ViT (fp32)", lambda m: static(root, "vitcms", "cosine_proto" + var, m))):
            print(f"  {label:22s}" + "".join(f"{fn(m).mean():9.3f}±{fn(m).std():.3f}"
                                              for m in ("rank1", "auroc", "dir@far0.05", "eer")))
    d = static(root, "vitcms") - static(root, "vit")
    w = int((d > 0).sum())
    print(f"  DECISION B1: CMS-ViT vs ViT, test DIR@5%: better on {w}/3, mean {d.mean():+.4f}  per seed {np.round(d, 4).tolist()}")
    print(f"  VERDICT B1: {'GO' if w == 3 else 'NO-GO'}")

    vit, cms = "replay_k5_fp32_vit", "replay_k5_vitcms1-16-256_p0.5_vit"
    cols = [("final_avg_rank1", "seen rank-1", 1), ("forgetting_rank1", "forget rank-1", -1),
            ("final_avg_dir@far0.05", "seen DIR@5%", 1), ("forgetting_dir@far0.05", "forget DIR@5%", -1),
            ("U:auroc", "unseen AUROC", 1), ("U:dir@far0.05", "unseen DIR@5%", 1)]
    print(f"\nB2 CONTINUAL (UERC P1, 10 tasks, replay k=5), test, seeds {SEEDS}")
    print(f"  {'arm':16s}" + "".join(f"{n:>16s}" for _, n, _ in cols))
    for label, f in (("ViT + R", vit), ("CMS-ViT + R", cms)):
        print(f"  {label:16s}" + "".join(f"{cont(root, f, k).mean():9.3f}±{cont(root, f, k).std():.3f}" for k, _, _ in cols))
    for k, n, sign in cols:
        dd = (cont(root, cms, k) - cont(root, vit, k)) * sign
        print(f"    {n:15s} CMS-ViT better on {int((dd > 0).sum())}/3, mean {dd.mean():+.4f}")
    d2 = cont(root, cms, "final_avg_rank1") - cont(root, vit, "final_avg_rank1")
    w2 = int((d2 > 0).sum())
    print(f"  DECISION B2: final seen rank-1 better on {w2}/3, mean {d2.mean():+.4f}  per seed {np.round(d2, 4).tolist()}")
    print(f"  VERDICT B2: {'GO' if w2 == 3 else 'NO-GO'}")


if __name__ == "__main__":
    main()
