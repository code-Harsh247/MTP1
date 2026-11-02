import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_strength_selection_rule():
    m = _load("r2_select")

    def r(rank1, forgetting, lam=1.0):
        return dict(rank1=rank1, forgetting=forgetting, lam=lam)

    best = r(0.90, 0.30)
    assert m.pick([r(0.80, 0.01), best]) is best                 # rank-1 decides
    near = r(0.897, 0.10)
    assert m.pick([best, near]) is near                          # within 0.005: lower forgetting
    low = r(0.90, 0.30, lam=1.0)
    assert m.pick([r(0.90, 0.30, lam=10.0), low]) is low         # tie: lower strength
    assert m.pick([r(float("nan"), 0.1), best]) is best
    assert m.pick([]) is None


def test_forgetting_rule():
    m = _load("r2_report")

    def r(f, final=0.5, joint=0.9):
        return dict(forgetting=f, final=final, joint=joint)

    assert m.verdict([r(0.2), r(0.1)], 2, 0.10, 0.10) == "SHOWN"           # boundary counts
    assert m.verdict([r(0.2), r(0.09)], 2, 0.10, 0.10) == "not shown"      # every seed must pass
    assert m.verdict([r(0.2), r(0.2, final=0.85)], 2, 0.10, 0.10) == "not shown"   # joint gap
    assert m.verdict([r(0.2)], 2, 0.10, 0.10) == "INCOMPLETE"              # a missing run never passes
    assert m.verdict([r(float("nan")), r(0.2)], 2, 0.10, 0.10) == "INCOMPLETE"


def test_forgetting_matches_the_summary():
    from nlear.continual import continual_summary
    m = _load("r2_report")
    A = [[{"rank1": v} for v in row] for row in ([0.9, 0.1, 0.2], [0.7, 0.8, 0.2], [0.5, 0.6, 0.9])]
    sm, _ = continual_summary(A, "rank1")
    final, forgetting = m.forgetting_of(A, "rank1")
    assert abs(final - sm["final_avg_rank1"]) < 1e-9 and abs(forgetting - sm["forgetting_rank1"]) < 1e-9
