"""Score saved embeddings with (new) scorers, without retraining the backbone.

Tuning (validation only, printed as mean +- std over the given seeds, nothing saved):
    python scripts/rescore.py runs/earvn/baseline_resnet50 --val-only \
        --scorer evm:gallery:5 evm:train:50

Final scoring (adds the scorers to each seed's metrics.json, incl. split variants like g5):
    python scripts/rescore.py runs/earvn/baseline_resnet50 --scorer evm evm_trainneg

A scorer is a registered name (nlear.eval.SCORERS) or evm:<gallery|train>:<tail>.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nlear.eval.evm import make_evm  # noqa: E402
from nlear.memory import AssocMemory, make_memory_scorer  # noqa: E402
from nlear.eval.openset import SCORERS  # noqa: E402
from nlear.eval.protocol import run_protocol, source_blobs, variant_blobs  # noqa: E402

KEYS = ["rank1", "auroc", "dir@far0.01", "dir@far0.05", "dir@far0.1", "eer"]


def resolve(spec):
    if spec.startswith("evm:"):
        _, negatives, tail = spec.split(":")
        return make_evm(int(tail), negatives)
    if spec.startswith("mem:"):
        # mem:<linear|mlp>:<hidden>:<steps>:<lr>:<momentum>:<forget>  (untrained memory)
        _, arch, hidden, steps, lr, mom, fgt = spec.split(":")
        mem = AssocMemory(512, arch=arch, hidden=int(hidden), steps=int(steps), lr=float(lr),
                          momentum=float(mom), forget=float(fgt))
        return make_memory_scorer(mem, device="cuda" if torch.cuda.is_available() else "cpu")
    return SCORERS[spec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method_dir")
    ap.add_argument("--scorer", nargs="+", required=True)
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--by-source", action="store_true",
                    help="also score each source separately (keys scorer#<source>[@variant])")
    args = ap.parse_args()

    d = Path(args.method_dir)  # a method dir (seed*/ inside) or a single seed dir
    runs = [d / "embeddings.pt"] if (d / "embeddings.pt").exists() else sorted(d.glob("seed*/embeddings.pt"))
    assert runs, f"no embeddings under {args.method_dir}"
    collected = {}  # (scorer key) -> list of metric dicts, one per seed
    for emb_path in runs:
        run_dir = emb_path.parent
        cfg = yaml.safe_load(open(run_dir / "config.yaml"))
        blob = torch.load(emb_path, weights_only=False)
        results = {}
        groups = {"": blob}
        if args.by_source:
            groups.update({f"#{s}": b for s, b in source_blobs(blob, cfg["data"], cfg["seed"]).items()})
        jobs = [(g + (f"@{v}" if v else ""), vb) for g, gb in groups.items()
                for v, vb in variant_blobs(gb, cfg["data"], cfg["seed"]).items()]
        for suffix, vblob in jobs:
            for spec in args.scorer:
                key = spec + suffix
                try:
                    r = run_protocol(vblob, resolve(spec))
                except ValueError as e:  # non-finite scores: the memory write diverged
                    if "NaN" not in str(e) and "infinity" not in str(e):
                        raise
                    print(f"  {key}: diverged (non-finite scores), skipped")
                    continue
                results[key] = {"val": r["val"]} if args.val_only else r
                collected.setdefault(key, []).append(r["val" if args.val_only else "test"])
        if not args.val_only:
            m = json.load(open(run_dir / "metrics.json"))
            m["results"].update(results)
            json.dump(m, open(run_dir / "metrics.json", "w"), indent=1)
        print(f"scored {run_dir}")

    split = "val" if args.val_only else "test"
    print(f"\n{args.method_dir}  ({len(runs)} seeds, split={split})")
    print(f"{'scorer':24s}" + "".join(f"{k:>17s}" for k in KEYS))
    for key, rs in collected.items():
        cells = [f"{np.mean([r[k] for r in rs]):8.3f} ±{np.std([r[k] for r in rs]):6.3f}" for k in KEYS]
        print(f"{key:24s}" + "".join(f"{c:>17s}" for c in cells))


if __name__ == "__main__":
    main()
