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
