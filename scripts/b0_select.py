"""B0 selection (notes/b_vit_cms.md): pick the plain-ViT recipe on UERC seed-0 VALIDATION and write
configs/vit_selected.yaml (configs/vit.yaml with the chosen lr_body and epochs).

    python scripts/b0_select.py runs_dgx/uerc/tune
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GRID = [(lr, e) for lr in ("3e-5", "1e-4") for e in (4, 8)]


def main():
    tune = Path(sys.argv[1])
    val = {}
    for lr, e in GRID:
        f = tune / f"baseline_deit_small_patch16_224_vit_lr{lr}_e{e}/seed0/metrics.json"
        if not f.exists():
            print(f"lr {lr} e{e}: no metrics (failed/diverged) -> ineligible")
            continue
        r = json.load(open(f))["results"]["cosine_proto"]
        assert "test" not in r, "tuning output must not contain test metrics"
        val[(lr, e)] = r["val"]
        print(f"VAL seed0 lr_body {lr:5s} epochs {e}: DIR@5% {r['val']['dir@far0.05']:.4f}  "
              f"AUROC {r['val']['auroc']:.4f}  rank1 {r['val']['rank1']:.4f}")
    ranked = sorted(val, key=lambda k: (val[k]["dir@far0.05"], val[k]["auroc"]), reverse=True)
    top, nxt = ranked[0], ranked[1]
    tie = val[top]["dir@far0.05"] - val[nxt]["dir@far0.05"] <= 0.005
    lr, e = max((top, nxt), key=lambda k: val[k]["auroc"]) if tie else top
    cfg = yaml.safe_load(open(ROOT / "configs/vit.yaml"))
    cfg["train"]["lr_body"], cfg["train"]["epochs"] = float(lr), int(e)
    out = ROOT / "configs/vit_selected.yaml"
    with open(out, "w", newline="\n") as fh:
        fh.write(f"# Written by scripts/b0_select.py from UERC seed-0 validation (B0): lr_body {lr}, epochs {e}\n")
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f"SELECTED lr_body {lr}, epochs {e} (tie={tie}) -> {out}")


if __name__ == "__main__":
    main()
