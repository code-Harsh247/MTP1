# NL-Ear

Catastrophic forgetting in ear recognition across model families and six ear datasets, and whether Nested Learning (NL) fixes it. MTP project; see [plan.md](plan.md).

Study v1 (NL for open-set ear recognition) is archived in [archive/v1/](archive/v1/) and tagged `v1-final`.

```
plan.md               study v2 plan: phases R0-R5, rules, datasets, models (one experiment per dataset)
notes/                NL equations (nl_equations.md), DGX runbook (dgx_runbook.md), new results notes
papers/               Nested Learning and Titans PDFs
reports/nl_primer/    plain-language study guide to Nested Learning
configs/              one per dataset and model recipe
nlear/                library: data, models, CMS, continual learning, evaluation
scripts/              training, evaluation and report scripts; scripts/dgx/ runs on the cluster
tests/
archive/v1/           plans, notes and reports from study v1
```

## Setup (Windows laptop)

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
```
