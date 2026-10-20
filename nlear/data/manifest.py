"""A manifest lists every image once: path (relative to the image root), identity, source."""
import hashlib
import re
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_manifest(root, source="unknown", identity_regex=None, dedup=True, source_from=None):
    """Scan every image under root and return a manifest DataFrame.

    identity_regex=None: identity is the first folder under root (root/<identity>/**/<img>).
    identity_regex given: it is searched in the path relative to root, and group 1 is the
    identity. Used for datasets that keep all images in one folder with the subject id in
    the file name (e.g. r"^(\\d+)_" for "012_front_ear.jpg").

    source_from="extension": each identity's source is the majority file extension of its
    images (UERC: the 3 source datasets are not labelled, but JPG vs PNG tracks them; used
    only to stratify splits). Otherwise every row gets the fixed `source`.
    """
    root = Path(root)
    pattern = re.compile(identity_regex) if identity_regex else None
    rows, unmatched = [], []
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = f.relative_to(root).as_posix()
        if pattern is None:
            if "/" not in rel:
                unmatched.append(rel)
                continue
            ident = rel.split("/", 1)[0]
        else:
            m = pattern.search(rel)
            if m is None:
                unmatched.append(rel)
                continue
            ident = m.group(1)
        rows.append({"path": rel, "identity": ident, "source": source})
    if unmatched:
        print(f"warning: {len(unmatched)} images had no identity, skipped (e.g. {unmatched[:3]})")
    if not rows:
        raise FileNotFoundError(f"no images with an identity found under {root}")
    df = pd.DataFrame(rows)
    if source_from == "extension":
        ext = df["path"].str.rsplit(".", n=1).str[-1].str.lower()
        majority = ext.groupby(df["identity"]).agg(lambda e: e.value_counts().idxmax())
        df["source"] = df["identity"].map(majority)
    elif source_from is not None:
        raise ValueError(f"unknown source_from {source_from!r}")
    if dedup:
        df = drop_exact_duplicates(df, root)
    return df


def drop_exact_duplicates(df, root):
    """Remove byte-identical image files.

    A duplicate inside one identity could land in both gallery and probe and make matching
    trivially easy. Copies are dropped, keeping the shallowest path. A file shared by two
    identities is label noise, so every copy of it is dropped.
    """
    root = Path(root)
    df = df.copy()
    df["md5"] = [hashlib.md5((root / p).read_bytes()).hexdigest() for p in df["path"]]
    df["_depth"] = df["path"].str.count("/")
    df = df.sort_values(["_depth", "path"])
    ids_per_hash = df.groupby("md5")["identity"].nunique()
    conflict = df["md5"].isin(ids_per_hash[ids_per_hash > 1].index)
    dup = df.duplicated("md5", keep="first") & ~conflict
    if dup.any():
        print(f"dedup: dropped {int(dup.sum())} exact duplicate files "
              f"(e.g. {df.loc[dup, 'path'].head(3).tolist()})")
    if conflict.any():
        print(f"dedup: dropped {int(conflict.sum())} files shared by different identities "
              f"(e.g. {df.loc[conflict, 'path'].head(3).tolist()})")
    return df[~dup & ~conflict].drop(columns="_depth").sort_values("path").reset_index(drop=True)


def load_manifest(csv_path):
    df = pd.read_csv(csv_path, dtype={"identity": str, "path": str, "source": str})
    missing = {"path", "identity"} - set(df.columns)
    if missing:
        raise ValueError(f"manifest {csv_path} lacks columns {missing}")
    if "source" not in df.columns:
        df["source"] = "unknown"
    return df
