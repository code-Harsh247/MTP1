import numpy as np
import torch
import pandas as pd
import pytest
from PIL import Image

from nlear.data import EarDataset, build_manifest, build_transform, make_splits


@pytest.fixture
def fake_root(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(50):
        d = tmp_path / f"id{i:03d}"
        d.mkdir()
        for j in range(int(rng.integers(3, 12))):
            arr = rng.integers(0, 255, (60, 40, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{j}.png")
    return tmp_path


def test_manifest_identity_from_file_name(tmp_path):
    # AMI/IITD-style layout: one folder, subject id in the file name
    for sid in range(3):
        for view in ("front", "up", "down"):
            Image.new("RGB", (8, 8)).save(tmp_path / f"{sid:03d}_{view}_ear.jpg")
    m = build_manifest(tmp_path, identity_regex=r"(?:^|/)(\d+)_", dedup=False)  # blank images are identical
    assert sorted(m["identity"].unique()) == ["000", "001", "002"]
    assert (m.groupby("identity").size() == 3).all()


def test_manifest_merges_presplit_train_test_folders(tmp_path):
    # AMI/AWE/IITD on disk: <train|test>/<id>/<img>; identity must be <id>, not train/test
    for part, ids in (("train", ["001", "002"]), ("test", ["003"])):
        for sid in ids:
            d = tmp_path / part / sid
            d.mkdir(parents=True)
            for k in range(2):
                Image.new("RGB", (8, 8)).save(d / f"{sid}_{k}.bmp")
    m = build_manifest(tmp_path, identity_regex=r"([^/]+)/[^/]+$", dedup=False)
    assert sorted(m["identity"].unique()) == ["001", "002", "003"]


def test_manifest_drops_exact_duplicates(tmp_path):
    # AMI ships subject 104 a second time inside test/102/104/
    for d in ("102", "104", "102/104"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    imgs = {k: rng.integers(0, 255, (8, 8, 3), dtype=np.uint8) for k in ("a", "b", "c")}
    Image.fromarray(imgs["a"]).save(tmp_path / "102" / "102_a.png")
    Image.fromarray(imgs["b"]).save(tmp_path / "104" / "104_b.png")
    Image.fromarray(imgs["b"]).save(tmp_path / "102" / "104" / "104_b.png")   # duplicate copy
    Image.fromarray(imgs["c"]).save(tmp_path / "102" / "102_c.png")
    Image.fromarray(imgs["c"]).save(tmp_path / "104" / "104_c.png")           # same file, 2 ids
    m = build_manifest(tmp_path, identity_regex=r"([^/]+)/[^/]+$")
    assert m["path"].tolist() == ["102/102_a.png", "104/104_b.png"]


def test_splits_follow_open_set_protocol(fake_root):
    m = build_manifest(fake_root)
    df = make_splits(m, seed=0, min_images=4, n_gallery=2)

    assert (m.groupby("identity").size() >= 4).sum() == df["identity"].nunique()
    assert (df.groupby("identity")["split"].nunique() == 1).all()
    assert not ((df["known"] == 0) & (df["role"] == "gallery")).any()
    g = df[df["role"] == "gallery"].groupby("identity").size()
    assert (g == 2).all()
    for split in ("val", "test"):
        s = df[df["split"] == split]
        assert (s["known"] == 1).any() and (s["known"] == 0).any()


def test_splits_are_deterministic(fake_root):
    m = build_manifest(fake_root)
    a = make_splits(m, seed=3)
    b = make_splits(m, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_dataset_loads_images(fake_root):
    df = make_splits(build_manifest(fake_root), seed=0)
    ds = EarDataset(df, fake_root, "train", transform=build_transform((64, 64), train=True))
    img, label, known = ds[0]
    assert img.shape == (3, 64, 64)
    assert 0 <= label < ds.num_classes
    assert int(known) == -1


def test_source_from_extension_uses_majority_per_identity(tmp_path):
    rng = np.random.default_rng(0)
    for ident, exts in (("a", ["jpg"] * 3 + ["png"]), ("b", ["png"] * 3)):
        d = tmp_path / ident
        d.mkdir()
        for j, ext in enumerate(exts):
            Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(d / f"{j}.{ext}")
    m = build_manifest(tmp_path, source_from="extension")
    assert m.groupby("identity")["source"].unique().map(list).to_dict() == {"a": ["jpg"], "b": ["png"]}


def test_parallel_cache_build_matches_serial_and_reloads_mmapped(tmp_path, fake_root):
    from nlear.data.cache import load_image_cache
    m = build_manifest(fake_root)
    serial, idx_s = load_image_cache(m, fake_root, (16, 12), tmp_path / "s.pt", workers=0)
    parallel, idx_p = load_image_cache(m, fake_root, (16, 12), tmp_path / "p.pt", workers=2)
    assert idx_s == idx_p and torch.equal(serial, parallel)
    again, _ = load_image_cache(m, fake_root, (16, 12), tmp_path / "p.pt")
    assert torch.equal(again, parallel)
