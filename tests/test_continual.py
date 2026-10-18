import numpy as np
import pandas as pd
import pytest
import torch

from nlear.continual import continual_summary, make_task_splits
from nlear.data.splits import make_splits
from nlear.models import ArcFace


def _split_df(n_ids=40, per_id=12, seed=0):
    m = pd.DataFrame({"path": [f"{i:03d}/{j}.jpg" for i in range(n_ids) for j in range(per_id)],
                      "identity": [f"{i:03d}" for i in range(n_ids) for _ in range(per_id)],
                      "source": "x"})
    return make_splits(m, seed=seed)


def test_task_splits_are_disjoint_complete_and_deterministic():
    df = _split_df()
    a = make_task_splits(df, n_tasks=4, holdout_frac=0.25, seed=1)
    b = make_task_splits(df, n_tasks=4, holdout_frac=0.25, seed=1)
    pd.testing.assert_frame_equal(a, b)
    tr = a[a["split"] == "train"]
    assert sorted(tr["task"].unique()) == [0, 1, 2, 3]
    assert (tr.groupby("identity")["task"].nunique() == 1).all()
    sizes = tr.groupby("task")["identity"].nunique()
    assert sizes.max() - sizes.min() <= 1
    # every train identity: 2 held-out gallery, >=1 held-out probe, rest training images
    per = tr.groupby(["identity", "cl_role"]).size().unstack(fill_value=0)
    assert (per["held_gallery"] == 2).all() and (per["held_probe"] >= 1).all() and (per["train"] >= 1).all()
    # val/test untouched
    assert (a.loc[a["split"] != "train", "cl_role"] == "eval").all()
    assert (a.loc[a["split"] != "train", "task"] == -1).all()


def test_too_few_images_to_hold_out_is_an_error():
    df = _split_df(per_id=4)
    with pytest.raises(ValueError):
        make_task_splits(df, n_tasks=2, holdout_frac=0.9, seed=0)


def test_forgetting_and_backward_transfer_by_hand():
    # 3 stages x 3 tasks; metric = value in the matrix
    M = [[0.8, 0.1, 0.1],
         [0.5, 0.9, 0.2],
         [0.4, 0.6, 0.7]]
    A = [[{"dir@far0.05": v} for v in row] for row in M]
    s, _ = continual_summary(A)
    assert s["final_avg_dir@far0.05"] == pytest.approx((0.4 + 0.6 + 0.7) / 3)
    # task0: best before last = max(0.8, 0.5) = 0.8 -> drop 0.4; task1: max(0.9) -> drop 0.3
    assert s["forgetting_dir@far0.05"] == pytest.approx((0.4 + 0.3) / 2)
    # bwt: (0.4 - 0.8 + 0.6 - 0.9) / 2
    assert s["bwt_dir@far0.05"] == pytest.approx((-0.4 - 0.3) / 2)


def test_single_stage_summary_has_no_forgetting():
    A = [[{"dir@far0.05": 0.5}, {"dir@far0.05": 0.7}]]
    s, _ = continual_summary(A)
    assert s == {"final_avg_dir@far0.05": pytest.approx(0.6)}


def test_arcface_active_mask_excludes_unseen_classes():
    torch.manual_seed(0)
    head = ArcFace(16, 6, s=10, m=0.3)
    emb, y = torch.randn(8, 16), torch.randint(0, 3, (8,))
    active = torch.tensor([True, True, True, False, False, False])
    loss, cos = head(emb, y, active)
    assert torch.isfinite(loss)
    assert (cos[:, 3:] == -2.0).all() and (cos.argmax(1) < 3).all()
    loss.backward()
    # unseen classes get no gradient through the softmax
    assert head.weight.grad[3:].abs().sum() == 0
