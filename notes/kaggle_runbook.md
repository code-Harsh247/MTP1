# Kaggle runbook (study v2)

*Written 2026-11-01. Tested in a simulated session on the laptop, not yet on Kaggle itself.*

Kaggle runs AMI, AWE, IITD and EarVN. UERC stays on the DGX: its cache is 9.2 GB and it is the most expensive set. Within a model × dataset cell, every arm runs on the same kind of machine.

Check each dataset's licence before uploading. The image cache still contains the ear images.

## Upload (once)

```bash
python scripts/kaggle/package.py
```

This writes `kaggle_upload/nlear_code.zip` and `kaggle_upload/nlear_data.zip`. On kaggle.com, create two datasets, both **Private**: `nlear-code` and `nlear-data`. After a code change, run with `--code-only` and upload a new version of `nlear-code`.

## Run a session

1. Create a notebook. In its settings: Accelerator **GPU T4 x2**, Internet **on** (needs a verified phone number).
2. Add both datasets as input. From the second session on, also add the previous version's output, which holds `ledger.tsv`.
3. Use this single cell:

```python
import os
code = next(r for r, _, f in os.walk('/kaggle/input', followlinks=True) if 'session.py' in f)
!python {code}/session.py --r1 tune --hours 11
```

4. **Save Version → Save & Run All**. It keeps running with the browser closed.
5. When it ends, download `results.tar.gz` and `ledger.tsv` from the Output tab. Unpack each session's archive into the project root.

If the log ends in "not finished", repeat from step 2. Finished lines are skipped, failed or stopped ones run again. After 5 failures in a row the session stops early, so read `logs_kaggle/` in the archive.

## Avoid running twice

The DGX queue holds all 150 tuning runs. Before its worker starts, take the Kaggle datasets out (in `nlear/` on the DGX):

```bash
mkdir -p queue/on_kaggle && grep -lE -- '--bench (ami|awe|iitd|earvn) ' queue/pending/*.task | xargs -I{} mv {} queue/on_kaggle/
```

If Kaggle doesn't work out, put them back:

```bash
mv queue/on_kaggle/*.task queue/pending/
```

## Session commands by phase

Use one of these as the cell's last line:

| Phase | Last line |
|---|---|
| R2, naive, replay and strength tuning | `!python {code}/session.py --r2 first --hours 11` |
| R2, EWC and LwF | `!python {code}/session.py --r2 second --hours 11` |
| R3, CMS (tunes, selects, then runs) | `!python {code}/session.py --r3 cms --hours 11` |
| R3, replay + CMS | `!python {code}/session.py --r3 replay --only earvn,awe,ami,iitd --hours 11` |
| R4, task-boundary CMS (35 runs) | `!python {code}/session.py --r4 --only earvn,awe,ami --hours 11` |

The order in `--only` is the order the runs happen in.

## Choosing recipes

```bash
python scripts/r1_select.py runs_kaggle/r1_tune runs_dgx/r1_tune
```

The final runs follow with `--r1 final` in the notebook cell, once `benchmark/recipes.yaml` is committed and in a new version of `nlear-code`.
