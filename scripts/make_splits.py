"""Build the manifest (if missing) and identity-disjoint open-set splits, one per seed.

    python scripts/make_splits.py --config configs/ami.yaml
    python scripts/make_splits.py --config configs/ami.yaml --rebuild-manifest
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nlear.data.manifest import build_manifest, load_manifest  # noqa: E402
from nlear.data.splits import make_splits, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rebuild-manifest", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d, s = cfg["data"], dict(cfg["splits"])
    seeds = s.pop("seeds")
    variants = s.pop("variants", {}) or {}

    manifest_path = Path(d["manifest"])
    if manifest_path.exists() and not args.rebuild_manifest:
        manifest = load_manifest(manifest_path)
    else:
        manifest = build_manifest(d["root"], source=d.get("source", cfg["name"]),
                                  identity_regex=d.get("identity_regex"),
                                  source_from=d.get("source_from"))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
    counts = manifest.groupby("identity").size()
    print(f"[{cfg['name']}] manifest: {len(manifest)} images, {len(counts)} identities, "
          f"images/identity min {counts.min()} median {counts.median():.0f} max {counts.max()}")

    for seed in seeds:
        df = make_splits(manifest, seed=seed, **s)
        out = Path(d["splits"].format(seed=seed))
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nseed {seed} -> {out}")
        print(summarize(df).to_string(index=False))
        # Evaluation-only variants (e.g. more gallery images): same identities in every
        # split, only the gallery/probe roles differ, so one trained model serves all.
        for name, override in variants.items():
            v = make_splits(manifest, seed=seed, **{**s, **override})
            if not v[["path", "split", "known"]].equals(df[["path", "split", "known"]]):
                raise ValueError(f"variant {name} changed identities; it may only change roles")
            vout = Path(variant_path(d["splits"], name).format(seed=seed))
            v.to_csv(vout, index=False)
            g = summarize(v).set_index("split")
            print(f"  variant {name} -> {vout}  (test gallery {g.loc['test', 'gallery_imgs']}, "
                  f"probes {g.loc['test', 'probe_imgs']})")


def variant_path(template, name):
    p = Path(template)
    return str(p.with_name(p.name.replace("splits_", f"splits_{name}_", 1)))


if __name__ == "__main__":
    main()
