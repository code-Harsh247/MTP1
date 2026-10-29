# R0: freezing the benchmark (in progress)

*Started 2026-10-29. Nothing here is frozen until the final R0 commit, which adds `benchmark/` with split-file hashes.*

## Status

| Item | Status |
|---|---|
| KinEar | **dropped** (2026-10-29, user decision): uncropped photos and copied folders; details below |
| Common input size for the five models | **done: 128×96** (shapes check below) |
| Pretrained weights load at 128×96 | **done** (all five, ImageNet-1k weights; check below) |
| Splits for seeds 0–5, all five datasets | to do: AMI, AWE, EarVN, IITD and UERC have seeds 0–4 from v1; add seed 5 |
| Task count and held-out rule per dataset | draft below |
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

## Draft: tasks and held-out images per dataset

Held-out images of each training person are never trained on. They measure forgetting of learned people. v1's rule was 2 gallery images plus at least 1 probe (20% of the person's images).

| Dataset | Train ids (70%) | Tasks | People per task | Held out per person | Left to train on |
|---|---|---|---|---|---|
| AMI | 70 | 5 | 14 | 2 gallery + 1 probe | 4 of 7 |
| AWE | 70 | 5 | 14 | 2 gallery + 1 probe | 7 of 10 |
| EarVN1.0 | 115 | 5 | 23 | 20% (v1 rule) | ~80% |
| IITD-I | ~155 | 5 | ~31 | **1 probe**; gallery = the person's training images | 2–5 of 3–6 |
| UERC | 917 | 10 | ~92 | 20% (v1 rule) | ~80% |

IITD needs a small code change. Its held-out gallery comes from training images, because there are too few images to keep a separate one.
