# NL-Ear

Nested Learning memories for open-set ear recognition. MTP project; see [plan.md](plan.md).

```
plan.md               project plan and phases
notes/                equations (nl_equations.md), data-transfer steps (phase0_data.md)
papers/               Nested Learning and Titans PDFs
configs/              one per dataset: ami, earvn, awe (laptop), uerc (DGX)
nlear/data/           manifest, identity-disjoint open-set splits, dataset
scripts/make_splits.py
scripts/dgx/          scripts that run on the cluster
tests/
```

## Setup (Windows laptop)

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
```
