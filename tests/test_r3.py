import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r3_report", ROOT / "scripts" / "r3_report.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
RULE = yaml.safe_load(open(ROOT / "benchmark" / "r3_rule.yaml"))


def _row(base_f=0.4, cms_f=0.2, base_final=0.5, cms_final=0.6, joint=0.9):
    return dict(base_f=base_f, cms_f=cms_f, base_final=base_final, cms_final=cms_final, joint=joint)


def test_reduced_needs_a_real_drop_on_enough_seeds():
    assert m.compare([_row()] * 3, 3, RULE) == "REDUCED"
    assert m.compare([_row(cms_f=0.36)] * 3, 3, RULE) == "not reduced"              # mean drop too small
    assert m.compare([_row(), _row(), _row(cms_f=0.5)], 3, RULE) == "not reduced"   # 3 seeds: all must win
    assert m.compare([_row()] * 4 + [_row(cms_f=0.45)], 5, RULE) == "REDUCED"       # 5 seeds: one may lose
    assert m.compare([_row()] * 3 + [_row(cms_f=0.45)] * 2, 5, RULE) == "not reduced"
    assert m.compare([_row()] * 2, 3, RULE) == "INCOMPLETE"


def test_less_forgetting_by_learning_less_does_not_count():
    assert m.compare([_row(cms_final=0.47)] * 3, 3, RULE) == "not reduced"
    assert m.compare([_row(cms_final=0.49)] * 3, 3, RULE) == "REDUCED"


def test_fixed_needs_low_forgetting_and_a_final_near_joint():
    assert m.compare([_row(cms_f=0.05, cms_final=0.85)] * 3, 3, RULE) == "FIXED"
    assert m.compare([_row(cms_f=0.15, cms_final=0.85)] * 3, 3, RULE) == "REDUCED"
    assert m.compare([_row(cms_f=0.05, cms_final=0.7)] * 3, 3, RULE) == "REDUCED"


def test_rq2_answer():
    assert m.rq2(["REDUCED", "FIXED", "not reduced"]) == "YES"
    assert m.rq2(["REDUCED", "not reduced"]) == "PARTIAL"     # half is not more than half
    assert m.rq2(["not reduced", "not reduced"]) == "NO"
