import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nlear.data.cache import GPUAugment
from nlear.eval.openset import evaluate, score_cosine_max, score_cosine_proto, score_subspace
from nlear.eval.protocol import run_protocol
from nlear.models import ArcFace


def test_hand_computed_metrics():
    # 2 enrolled ids (A, B); probes: A (correct, 0.9), B (wrong argmax, 0.8), A (correct, 0.3),
    # unknown (0.5), unknown (0.2)
    S = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.3, 0.1], [0.5, 0.4], [0.2, 0.1]])
    m = evaluate(S, ["A", "B"], ["A", "B", "A", "X", "Y"], [1, 1, 1, 0, 0], fars=(0.5,))
    assert m["rank1"] == pytest.approx(2 / 3)
    # known max-scores 0.9, 0.8, 0.3 vs unknown 0.5, 0.2 -> 5 of 6 pairs ordered right (not 0.3 vs 0.5)
    assert m["auroc"] == pytest.approx(5 / 6)
    # tau = median of unknown scores (0.35): accepted & correct known probes = only the 0.9 one
    assert m["dir@far0.5"] == pytest.approx(1 / 3)


def test_calibrated_threshold_reports_actual_far():
    S = torch.tensor([[0.9, 0.1], [0.5, 0.4], [0.2, 0.1]])
    m = evaluate(S, ["A", "B"], ["A", "X", "Y"], [1, 0, 0], fars=(0.5,), calib_taus={0.5: 0.4})
    assert m["cal_far@far0.5"] == pytest.approx(0.5)   # 0.5 > 0.4 accepted, 0.2 rejected
    assert m["cal_dir@far0.5"] == pytest.approx(1.0)


def test_unknown_probe_must_not_be_enrolled():
    S = torch.tensor([[0.9, 0.1]])
    with pytest.raises(AssertionError):
        evaluate(S, ["A", "B"], ["A"], [0])


def _clusters(n_ids=10, d=64, per_id=6, noise=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(n_ids, d, generator=g), dim=1)
    emb = centers.repeat_interleave(per_id, 0) + noise * torch.randn(n_ids * per_id, d, generator=g)
    return emb, [f"id{i}" for i in range(n_ids) for _ in range(per_id)]


@pytest.mark.parametrize("fn", [score_cosine_proto, score_cosine_max, score_subspace])
def test_scorers_separate_clean_clusters(fn):
    emb, ids = _clusters()
    gal = [i for i in range(len(ids)) if i % 6 < 2 and int(ids[i][2:]) < 5]      # ids 0-4 enrolled
    prb = [i for i in range(len(ids)) if i % 6 >= 2]
    S, classes = fn(emb[gal], [ids[i] for i in gal], emb[prb])
    labels = [ids[i] for i in prb]
    m = evaluate(S, classes, labels, [int(l[2:]) < 5 for l in labels])
    assert m["rank1"] == 1.0 and m["auroc"] > 0.99 and m["eer"] < 0.01


def test_subspace_with_one_gallery_image_is_abs_cosine():
    g = torch.randn(3, 16)
    p = torch.randn(5, 16)
    S, _ = score_subspace(g, ["a", "b", "c"], p)
    C = F.normalize(p, dim=1) @ F.normalize(g, dim=1).T
    assert torch.allclose(S, C.abs(), atol=1e-5)


def test_run_protocol_on_blob():
    emb, ids = _clusters(n_ids=8, per_id=5)
    n = len(ids)
    split = np.array(["val" if int(i[2:]) < 4 else "test" for i in ids])
    known = np.array([1 if int(i[2:]) % 4 < 2 else 0 for i in ids])
    role = np.array(["gallery" if (k == 1 and j % 5 < 2) else "probe" for j, k in enumerate(known)])
    blob = {"emb": emb, "identity": np.array(ids), "split": split, "role": role, "known": known}
    r = run_protocol(blob, "cosine_proto", fars=(0.1,))
    assert r["test"]["rank1"] == 1.0 and "cal_dir@far0.1" in r["test"]


def test_weibull_fit_recovers_parameters():
    from scipy.stats import weibull_min
    from nlear.eval.evm import fit_weibull_low
    x = weibull_min.rvs(3.0, scale=0.4, size=4000, random_state=0)
    k, lam = fit_weibull_low(x)
    assert k == pytest.approx(3.0, rel=0.1) and lam == pytest.approx(0.4, rel=0.05)


@pytest.mark.parametrize("negatives", ["gallery", "train"])
def test_evm_separates_clean_clusters_and_uses_train_negatives(negatives):
    from nlear.eval.evm import make_evm
    # clean clusters (within-class distance << half the between-class distance): EVM's margin
    # assumption holds, so it must be perfect; with noisier clusters it legitimately degrades
    emb, ids = _clusters(n_ids=12, per_id=6, noise=0.05)
    split = np.array(["train" if int(i[2:]) >= 8 else "test" for i in ids])
    known = np.array([1 if int(i[2:]) < 4 else (0 if int(i[2:]) < 8 else -1) for i in ids])
    role = np.array(["train" if s == "train" else ("gallery" if (k == 1 and j % 6 < 2) else "probe")
                     for j, (s, k) in enumerate(zip(split, known))])
    blob = {"emb": emb, "identity": np.array(ids), "split": split, "role": role, "known": known}
    # make a val copy of test so the protocol can calibrate
    for k in ("identity", "role", "known"):
        blob[k] = np.concatenate([blob[k], blob[k][split == "test"]])
    blob["split"] = np.concatenate([split, np.full((split == "test").sum(), "val")])
    blob["emb"] = torch.cat([emb, emb[torch.as_tensor(split == "test")]])
    fn = make_evm(tail=5, negatives=negatives)
    called = {}
    orig = fn

    def spy(G, gl, P, **kw):
        called["neg"] = kw.get("neg")
        return orig(G, gl, P, **kw)
    spy.needs_train_negatives = orig.needs_train_negatives
    r = run_protocol(blob, spy, fars=(0.1,))
    assert (called["neg"] is not None) == (negatives == "train")
    if negatives == "train":
        assert called["neg"].shape[0] == (split == "train").sum()
    assert r["test"]["rank1"] == 1.0 and r["test"]["auroc"] > 0.95


def test_evm_score_is_monotonic_in_psi_and_never_overflows():
    from nlear.eval.evm import score_evm
    emb, ids = _clusters(n_ids=5, per_id=4)
    S, classes = score_evm(emb[::2], ids[::2], emb[1::2], tail=3)
    assert S.shape == (10, 5) and torch.isfinite(S).all()
    # far-away probes: Psi underflows to 0, but scores must stay finite and distinct
    S2, _ = score_evm(emb[::2], ids[::2], -emb[1::2], tail=3)
    assert torch.isfinite(S2).all() and (S2.max(1).values > S2.min(1).values).all()
    # a degenerate (identical) tail -> huge kappa: still finite
    g = F.normalize(torch.randn(2, 8), dim=1)
    S3, _ = score_evm(g, ["a", "b"], torch.randn(4, 8), neg=g.repeat(5, 1), tail=5)
    assert torch.isfinite(S3).all()


def test_evm_score_matches_psi_ordering():
    # s = kappa*log(lambda/d) must order probes exactly like Psi = exp(-(d/lambda)^kappa)
    d = torch.tensor([0.1, 0.2, 0.3, 0.5]).double()
    kappa, lam = 2.5, 0.25
    psi = torch.exp(-(d / lam) ** kappa)
    s = kappa * torch.log(lam / d)
    assert torch.equal(torch.argsort(psi), torch.argsort(s))


def test_arcface_margin_lowers_target_logit():
    torch.manual_seed(0)
    emb, y = torch.randn(8, 32), torch.randint(0, 5, (8,))
    plain, arc = ArcFace(32, 5, s=10, m=0.0), ArcFace(32, 5, s=10, m=0.3)
    arc.weight.data = plain.weight.data.clone()
    l0, _ = plain(emb, y)
    l1, _ = arc(emb, y)
    assert torch.isfinite(l1) and l1 > l0


def test_gpu_augment_keeps_shape_and_range():
    x = torch.rand(4, 3, 32, 24)
    y = GPUAugment()(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


def test_source_blobs_split_gallery_and_probes_by_source(tmp_path):
    import pandas as pd
    from nlear.eval.protocol import source_blobs
    emb, ids = _clusters(n_ids=8, per_id=5)
    n = len(ids)
    paths = np.array([f"{i}/{j}.jpg" for i, j in zip(ids, range(n))])
    split = np.array(["val" if int(i[2:]) % 2 == 0 else "test" for i in ids])
    known = np.array([1 if int(i[2:]) < 4 else 0 for i in ids])
    role = np.array(["gallery" if (k == 1 and j % 5 < 2) else "probe" for j, k in enumerate(known)])
    source = np.array(["jpg" if int(i[2:]) in (0, 1, 4, 5) else "png" for i in ids])
    pd.DataFrame({"path": paths, "identity": ids, "split": split, "role": role, "known": known,
                  "source": source}).to_csv(tmp_path / "splits_seed0.csv", index=False)
    blob = {"emb": emb, "identity": np.array(ids), "split": split, "role": role, "known": known, "path": paths}
    cfg = {"data": {"splits": str(tmp_path / "splits_seed{seed}.csv")}, "splits": {}}
    parts = source_blobs(blob, cfg, 0)
    assert set(parts) == {"jpg", "png"}
    for s, b in parts.items():
        assert len(b["emb"]) == (source == s).sum() == len(b["path"])
        assert set(np.unique(b["identity"])) == set(np.array(ids)[source == s])
        r = run_protocol(b, "cosine_proto", fars=(0.1,))
        assert r["test"]["rank1"] == 1.0
