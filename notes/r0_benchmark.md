# R0: freezing the benchmark (in progress)

*Started 2026-10-29. Nothing here is frozen until the final R0 commit, which adds `benchmark/` with split-file hashes.*

## Status

| Item | Status |
|---|---|
| KinEar | **dropped** (2026-10-29, user decision): uncropped photos and copied folders; details below |
| Common input size for the five models | **done: 128×96** (shapes check below) |
| Pretrained weights load at 128×96 | **done** (all five, ImageNet-1k weights; check below) |
| Splits | **done locally**: seed 5 added for AMI, AWE, IITD (seeds 0–4 verified byte-identical to v1); EarVN already had 0–10; UERC keeps 0–3 on the DGX |
| Task lists and held-out rule | **done locally** for AMI, AWE, IITD, EarVN (`benchmark/`); UERC must be generated on the DGX |
| Generic backbone + CMS for all five families | **done** (`nlear/models.py`, `nlear/cms.py`, `tests/test_body_cms.py`) |
| Cost of one run per dataset | to do, on the DGX |

## KinEar inspection (dataset dropped; kept for the record)

The script is streamed over ssh, so nothing has to be copied to the DGX first. It also inspects the two folders we never opened (`Dataset_aligned`, `TS_Data`).

```
ssh dsamantaai@gpucluster.iitkgp.ac.in "python3 - /dgx1/dsamantaai/santhosh/Ear_Datasets --only kinear Dataset_aligned TS_Data" < scripts/dgx/inspect_datasets.py > data/kinear_report.txt
```

It prints:
- the folder layout and image counts;
- image sizes;
- the first lines of any annotation or readme files, which tell us how identities are encoded.

### First pass (2026-10-29, `data/kinear_report.txt`)

- **Size:** 1,477 JPG images, 2 GB, in 76 folders, one per person (`train/` 53, `test/` 23), median 19 images per folder.
- **Names are family roles:** `father1`, `mother3`, `daughter10`, `son22`, `ElderDaughter1`, …
- **Large files.** About 1.4 MB each, named like phone photos (`IMG2020…`, WhatsApp `IMG-…-WA…`). These may be **uncropped photos**, not ear crops. To check.
- **Suspicious folders:** `father2dup` and `son16.1`. They may be the same person as `father2` and `son16`, which would make one person count as two identities. To check.
### Second pass (folder list `data/kinear_folders.txt`, samples `data/kinear_samples/`)

- **Close-up ear photos, but not cropped.** Each picture shows the ear plus hair, cheek and neck, from different phones. Sizes range from 534×960 to 4000×4000 (median 2448×3264). In some pictures (e.g. `mother3`) the ear is only about a third of the frame. AMI is similar (close-up, uncropped); AWE, EarVN and UERC are tight crops.
- **`father2dup` is a byte-identical copy of `father2`** (checked on 2 files). The same is likely for `father5dup` / `father5`. Our manifest builder drops *every* copy of a file shared by two identities, so it would delete these people entirely. KinEar needs an alias rule instead: `<name>dup` → `<name>`, then normal deduplication.
- **Numbered variants:** `son13.1`/`son13.2`, `son16.1`/`son16.2`, `daughter15`/`daughter15.2`. These may be the same person (two sessions) or different people. To check.
- `train/mother 19` has a space in its name; harmless, but the identity regex must allow it.
- As with AMI, AWE and IITD, the dataset's own train/test folders are ignored.

### Considered and abandoned: cropping KinEar's ears

**The plan was** (not carried out, because KinEar was dropped the same day as too much trouble):
1. Copy KinEar (2 GB) to the laptop, like the other small datasets. Do the duplicate check and the numbered-folder check there.
2. Apply each photo's EXIF orientation. Phone photos are often stored sideways.
3. Detect the ear with a zero-shot open-vocabulary detector: OWLv2 (`google/owlv2-base-patch16-ensemble`), prompt "a human ear". Keep the top box and pad it by 15%.
4. Review **every** crop on contact sheets. Fix misses by hand in a small box file.
5. Save the crops to `data/kinear/` and commit the box list (`benchmark/kinear_boxes.csv`), so the crops can be rebuilt exactly.

**Why OWLv2:** it needs no training data. We have no ear boxes for full photos, and only 1,477 images have to be checked, which is feasible by eye. We make our own open-set splits.

## Model shapes check (`scripts/check_archs.py`, laptop RTX 3050, fp32, batch 32, random weights)

| Family | timm model | Params | Feature dim | s/step 128×96 | GB 128×96 | s/step 224×224 |
|---|---|---|---|---|---|---|
| Classic CNN | resnet50 | 23.5M | 2048 | 0.161 | 0.77 | 0.339 |
| Modern CNN | convnext_tiny | 27.8M | 768 | 0.356 | 0.95 | 0.796 |
| Lightweight CNN | mobilenetv3_large_100 | 4.2M | 1280 | 0.055 | 0.37 | 0.112 |
| Plain transformer | deit_small_patch16_224 | 21.6M | 384 | 0.090 | 0.56 | 0.380 |
| Hierarchical transformer | swin_tiny_patch4_window7_224 | 27.5M | 768 | 0.160 | 1.08 | 0.745 |

All five also run at 160×112 and 224×160.

**Decision (proposed): keep 128×96 for every model.**
1. All five models run at it.
2. It keeps the ear's aspect ratio.
3. EarVN images are tiny, so a larger size would only upsample them.
4. The v1 image caches are already at this size.

**Swin at 128×96.** timm adapts the windows automatically. The stage grids are 32×24, 16×12, 8×6 and 4×3. Windows shrink to 7×6 in stage 3 and 4×3 in stage 4, and shifted windows are switched off where the grid is not larger than the window (stage 4, and the width of stage 3). Because some windows changed size, the pretrained relative-position-bias tables must be resized when the weights are loaded. timm does that, but it still has to be checked with real weights.

**ResNet-50 weights.** v1 used torchvision's `IMAGENET1K_V2`. timm's `resnet50.tv2_in1k` holds the same weights, so all five models can come from timm through one code path.

## Pretrained weights at 128×96 (`scripts/check_pretrained.py`, 2026-10-29)

**Weights: ImageNet-1k only, for every family,** so no model gets extra pretraining data.

| Family | timm weights | Frozen-feature LOO rank-1 on AMI | Same, random weights |
|---|---|---|---|
| Classic CNN | `resnet50.tv2_in1k` (= torchvision V2, as in v1) | 0.798 | 0.266 |
| Modern CNN | `convnext_tiny.fb_in1k` | 0.888 | 0.252 |
| Lightweight CNN | `mobilenetv3_large_100.ra_in1k` | 0.867 | 0.303 |
| Plain transformer | `deit_small_patch16_224.fb_in1k` | 0.634 | 0.339 |
| Hierarchical transformer | `swin_tiny_patch4_window7_224.ms_in1k` | 0.805 | 0.324 |

Setup: all 707 AMI images, 100 people, closed-set nearest neighbour by cosine, no training.

**Reading:**
- All five load at 128×96 with no errors. timm resampled DeiT's position embeddings and Swin's attention-bias tables.
- Every pretrained model is far above its random-weight version, so the weights really are in use.
- Random weights already score 0.25–0.34, because AMI's photos of one person are very similar.
- These numbers only check that loading works. They are not a ranking of the models: none of them is trained yet.

**For the DGX:** it runs with `HF_HUB_OFFLINE=1`, so the four new weight sets (all except DeiT, which v1 already used there) must be copied into the DGX's Hugging Face cache before R1.

## Tasks and held-out images per dataset (`benchmark/tasks.yaml`)

Held-out images of each training person are never trained on. They measure forgetting of learned people. v1's rule was 2 gallery images plus at least 1 probe (20% of the person's images).

| Dataset | Train ids (70%) | Tasks | People per task | Held out per person | Left to train on |
|---|---|---|---|---|---|
| AMI | 70 | 5 | 14 | 2 gallery + 1 probe | 4 of 7 |
| AWE | 70 | 5 | 14 | 2 gallery + 1 probe | 7 of 10 |
| EarVN1.0 | 115 | 5 | 23 | 20% (v1 rule) | ~80% |
| IITD-I | ~155 | 5 | ~31 | **1 probe**; gallery = the person's training images | 2–5 of 3–6 |
| UERC | 917 | 10 | ~92 | 20% (v1 rule) | ~80% |

IITD needs a small code change. Its held-out gallery comes from training images, because there are too few images to keep a separate one.

### As built (2026-10-29, `scripts/make_task_lists.py`)

Each task file is written next to its split file (`data/<name>/tasks_seed{s}.csv`, not in git). `benchmark/hashes_<name>.txt` pins the md5 of every split file and task file, and `make_task_lists.py --check` verifies them.

Seed-0 summary (`benchmark/summary_<name>.txt` has every seed):

| Dataset | Tasks × people | Train images | Gallery images | Held-out probes |
|---|---|---|---|---|
| AMI | 5 × 14 | 280 | 140 (held out) | 70 |
| AWE | 5 × 14 | 490 | 140 (held out) | 70 |
| IITD | 5 × 31 | 400 | 400 (= the training images) | 155 |
| EarVN | 5 × 23 | 15,858 | 230 (held out) | 3,732 |

**The IITD rule (`held_gallery: 0`).** This is new code in `nlear/continual.py`. Each IITD person holds out only 1 probe and is enrolled with its own training images (new column `cl_gallery`). Seen-person accuracy is therefore a little optimistic on IITD, because the gallery images were trained on. Forgetting is still measured correctly: it is the drop of that same score over later tasks. The default rule (2 held-out gallery images) is unchanged, and the v1 task splits are reproduced exactly.

**Caution: AMI and AWE have tiny per-task tests.** Each task's seen-person test is only 14 people with 1 held-out probe each, so one probe moves rank-1 by 0.07. These datasets will need their extra seeds (4–5), and their per-task numbers should be read as noisy.

## Generic backbone and CMS for all five families (2026-10-29)

**Backbone.** `EmbeddingNet` takes any timm weight tag through one code path (`nlear/models.py: _timm`).
- Stochastic depth is set to 0 for every family. Swin's timm default is 0.1; the others are 0.
- MobileNetV3 keeps its 1×1 head conv, so its feature size is 1,280.
- All five use the same ImageNet normalization as our data pipeline.
- v1 configs (`resnet50`, bare ViT names) still build exactly what they built before.

**CMS (`body_cms_levels`).** `nlear/models.py: cms_targets` lists the layers that are replicated:

| Family | Layers replicated | Count |
|---|---|---|
| ResNet-50 | `conv1`, `conv3` of each bottleneck | 32 |
| ConvNeXt-T | `mlp` of each block | 18 |
| MobileNetV3 | `conv_pw` (+ `conv_pwl`) of each block | 29 |
| DeiT-S, Swin-T | `mlp` of each block | 12 |

`CMSScheduler.for_body` puts copy l of every replicated layer on update period l. Everything else in the body is updated every step.

**Tests** (`tests/test_body_cms.py`, all 74 tests of the suite pass). For every family:
- the CMS model computes exactly the plain model at initialization;
- slow copies change only on their period;
- with periods 1,1,1 the copies stay identical.

## DGX setup (2026-10-29)

- Code bundle `68384e0` and `r0_data.tar.gz` (manifests, splits and task files of AMI, AWE, IITD; EarVN task files) unpacked under `MTP_Harsh/`.
- **timm 1.0.30 on the DGX**, the same as the laptop.
- `prepare.py`:
  - built the AMI (700), AWE (1,000) and IITD (793) caches at 128×96;
  - checked the splits;
  - cached all five weight sets in the DGX Hugging Face cache.
- `make_task_lists.py --dgx --check`: **AMI, AWE, IITD and EarVN match their hash files on the DGX**. The cluster copies are byte-identical to the laptop's.
- UERC task lists built on the DGX: 917 train people, 10 tasks of 91–92. Seed 0 has 138,750 train images, 1,834 held-out gallery images and 33,015 held-out probes; seeds 1–3 are similar (`benchmark/summary_uerc.txt`).

**Shell notes:**
- In PowerShell, `$` inside a double-quoted ssh command is expanded *locally*. Use full paths instead of `$P` / `$PWD`.
- A one-shot `ssh host "cmd"` does not load the login profile, so `sbatch` is not on the PATH. Submit jobs from an interactive ssh session.

## Still to do in R0

1. **On the DGX:**
   - ~~bring UERC's hash and summary files back~~ (done: `benchmark/hashes_uerc.txt`, `summary_uerc.txt`);
   - the timing job (`make_r0_timing_tasks.sh`, 25 runs) is **submitted as job 35551**; waiting for results.
2. **The R0 freeze commit.**
