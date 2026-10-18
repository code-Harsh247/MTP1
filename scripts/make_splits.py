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

    manifest_path = Path(d["manifest"])
    if manifest_path.exists() and not args.rebuild_manifest:
        manifest = load_manifest(manifest_path)
    else:
        manifest = build_manifest(d["root"], source=d.get("source", cfg["name"]),
                                  identity_regex=d.get("identity_regex"))
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


if __name__ == "__main__":
    main()
