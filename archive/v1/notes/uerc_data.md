# UERC 2023 data (U0)

*Inspected 2026-10-20 on the DGX with `scripts/dgx/inspect_uerc.py`. The full report is in `data/uerc_report.txt` (laptop).*

**Package:** `/dgx1/dsamantaai/santhosh/Ear_Datasets/UERC 2023 Competition Package with sequestered anonymized data/`

```
README, config.ini, uerc23.py, ...
baseline_model/   train_baseline.py, uerc-resnet18-dropout0.2.pt   (official UERC baseline)
data/             public_annotations.csv (cid, gender, ethnicity), sequestered_anonymized_*.csv,
                  public/<cid>/<n>.jpg|png, sequestered_anonymized/0/   (unlabelled; not used)
```

## `public/` at a glance

| | |
|---|---|
| Identities | **1,310** (folders `0001` … `1310`) |
| Images | **248,655**: 234,651 JPG and 14,004 PNG; layout `public/<cid>/<n>.<ext>` |
| Images per identity | min **10**, 10th percentile 10, median 150, 90th percentile 428, max 645. Everyone has ≥ 10 |
| Image size (500 sampled) | width 18 / **31** / 78 px (5th percentile / median / 95th percentile); height 31 / **62** / 140 px. RGB (a few RGBA) |
| Mean file size | 9.9 KB |
| RAM cache at 128×96 | **9.2 GB** |

## Composition (from the package README, confirmed by the prepare job 34941)

| Source | Subjects | Images | Images/subject | File type |
|---|---|---|---|---|
| VGGFace2-Ear-derived | 660 | 234,651 | ~355 | **all JPG** |
| UERC 2019 training set | 166 | 2,304 | ~14 | PNG |
| Newly collected | 484 | 11,700 | ~24 | PNG |

- **The file-type proxy is exact for VGGFace2:** 660 JPG identities and exactly 234,651 JPG images. PNG covers both of the other sources, which we don't separate.
- **The prepare job found 28 exact duplicates** (e.g. `0530/017.png`, `0717/340.jpg`). The final manifest has 248,627 images.
- **Splits (every seed):** 917 train / 131 val (65 known + 66 unknown) / 262 test (131 + 131) identities, balanced across sources (e.g. test: 132 JPG, 130 PNG identities). Each seed has about 49k test probes and about 24k val probes.

**Image imbalance (important).** VGGFace2 is half the people but **94% of the images**, so any metric averaged over probe images is dominated by VGGFace2 ears. From U1 on, results are also reported **per source**: VGGFace2 (JPG) vs. the other two (PNG), each evaluated with its own gallery and probes (`--by-source`).

## Decisions

- **Source stratification.** The source is not in the annotations (they only have gender and ethnicity), so each identity's majority file type (JPG/PNG) is used, via `source_from: extension` in `configs/uerc.yaml`. This separates VGGFace2 from the rest exactly (see above).
- **Image size.** The images are tiny, much smaller than EarVN's (median 77×110). We keep **128×96** so the recipe stays comparable with EarVN, which means about 3–4× upsampling. A smaller input (e.g. 96×64) could be tried later on validation if training speed matters.
- **Splits.** Seeds 0–3 (0 for tuning on val only; 1–3 for the reported test runs). 70/10/20 gives about 917 / 131 / 262 identities per split. Val and test are half known, half unknown. There are 2 gallery images per identity, plus a g5 variant.
- **Preparation runs on a compute node** (`scripts/dgx/prepare_uerc.sbatch`): manifest with exact-duplicate removal (hashes about 2.4 GB), splits, a parallel cache build (14 processes), and the package README printed into the log. The cache is **memory-mapped** when loaded, so jobs don't each copy 9.2 GB into RAM. Training jobs still request `--mem=64G` to be safe.
- **Not used:** the sequestered set (unlabelled) and `uerc23_dataset.zip` (truncated). The official baseline model is kept in mind as an extra reference for U1.
