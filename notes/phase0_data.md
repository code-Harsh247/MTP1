# Phase 0: getting the small datasets onto the laptop

We develop on the small datasets on the laptop, in the order **AMI → EarVN1.0 → AWE**. UERC stays on the DGX until stage S4 (see [plan.md](../plan.md) §4). The small sets are copied over **whole**.

The DGX login is **`dsamantaai@gpucluster.iitkgp.ac.in`** (if you are off campus, you may need the institute VPN on for this). The account is **shared**: keep everything under `/dgx1/dsamantaai/santhosh/MTP_Harsh/` and prefix jobs with `harsh_`.

## Step 1: inspect the layouts (on the DGX)

We don't yet know the exact folder names or how each dataset encodes the subject ID. The configs in `configs/` contain guesses marked `VERIFY`.

Run these in a terminal on the laptop, from `C:\MTP1`:

```bash
scp scripts/dgx/inspect_datasets.py dsamantaai@gpucluster.iitkgp.ac.in:/dgx1/dsamantaai/santhosh/MTP_Harsh/
```

```bash
ssh dsamantaai@gpucluster.iitkgp.ac.in "cd /dgx1/dsamantaai/santhosh/MTP_Harsh && python3 inspect_datasets.py /dgx1/dsamantaai/santhosh/Ear_Datasets > datasets_report.txt 2>&1"
```

```bash
scp dsamantaai@gpucluster.iitkgp.ac.in:/dgx1/dsamantaai/santhosh/MTP_Harsh/datasets_report.txt data/
```

The script only reads files and skips UERC, so it is quick. **Then tell me the report is there** (`data/datasets_report.txt`). From it I'll fix the configs and give you the exact pack commands. The report shows:

- the folder names
- the image counts and total size of each dataset
- whether identity comes from the folder or the file name
- the image sizes

## Step 2: pack and download

The report (`data/datasets_report.txt`, 2026-10-18) gave these layouts:

| Folder on the DGX | Unpacks to | Layout | Ids / images |
|---|---|---|---|
| `ami_dataset` | `data/ami` | `{train,test}/<id>/<id>_<view>_ear.jpg` | 101 / 707 |
| `EarVN1.0 dataset` | `data/earvn` | `Images/<NNN.Name>/*.jpg` | 164 / 28,412 |
| `awe_dataset` | `data/awe` | `{train,test}/<id>/NN.png` + `annotations.json` | 100 / 1,000 |
| `iitd1p_dataset` | `data/iitd` | `{train,test}/<id>/<id>_<k>.bmp` | 221 / 793 |

That is about 210 MB in total. The `--transform` option renames the folders to short names without spaces while packing; nothing on the DGX is changed.

**1. Pack on the DGX:**

```bash
ssh dsamantaai@gpucluster.iitkgp.ac.in "mkdir -p /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear_data && cd /dgx1/dsamantaai/santhosh/Ear_Datasets && tar czf /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear_data/small_ears.tar.gz --transform 's,^EarVN1.0 dataset,earvn,;s,^ami_dataset,ami,;s,^awe_dataset,awe,;s,^iitd1p_dataset,iitd,' 'EarVN1.0 dataset' ami_dataset awe_dataset iitd1p_dataset && ls -lh /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear_data/small_ears.tar.gz"
```

**2. Download:**

```bash
scp dsamantaai@gpucluster.iitkgp.ac.in:/dgx1/dsamantaai/santhosh/MTP_Harsh/nlear_data/small_ears.tar.gz data/
```

**3. Unpack:**

```bash
tar -xzf data/small_ears.tar.gz -C data
```

## Step 3: build splits (on the laptop)

```bash
.venv/Scripts/python scripts/make_splits.py --config configs/ami.yaml
```

This writes `data/ami/manifest.csv` and `splits_seed0..4.csv`, and prints subjects and images per split. Repeat with `earvn`, `awe` and `iitd`.

## Notes

- **The datasets' own train/test folders are ignored.** AMI, AWE and IITD come pre-split, but those splits are closed-set protocols from earlier papers. We merge them (identity = the folder holding the image) and make our own open-set splits.
- **IITD has only 3–6 images per person** (median 3), so its config uses `min_images: 3` and `n_gallery: 1`. It is used only as a cross-dataset test.
- **EarVN1.0 images are small** (about 4.6 KB per JPEG on average). Check the actual pixel sizes after download before fixing `img_size`. PIL isn't installed in the DGX `python3`, so the report couldn't measure them.
- Not inspected yet: `Dataset_aligned/`, `TS_Data/` and `kinear_dataset/`. It's worth asking what `Dataset_aligned` is.
- `data/` is git-ignored. Images never go into the repo.
- Expected counts after `make_splits`: AMI 101/707, EarVN1.0 164/28,412, AWE 100/1,000, IITD 221/793. A mismatch means a wrong regex or folder choice.
- These datasets carry their own licence terms (most are research-only). Keep them on your machines and don't redistribute.

## Phase 0 results (2026-10-18)

After `make_splits.py` with duplicate removal. Split counts are for seed 0; the other seeds have the same sizes but different people.

| Dataset | Ids | Images | Img/id | Train ids | Val (known/unknown) | Test (known/unknown) | Image size (median w×h) | Mode |
|---|---|---|---|---|---|---|---|---|
| AMI | 100 | 700 | 7 | 70 | 5 / 5 | 10 / 10 | 492×702 (all equal) | RGB |
| EarVN1.0 | 164 | 28,410 | 107–300 | 115 | 8 / 8 | 16 / 17 | 77×110 (p5 27×37) | RGB |
| AWE | 100 | 1,000 | 10 | 70 | 5 / 5 | 10 / 10 | 64×122 (very variable) | RGBA |
| IITD | 221 | 793 | 3–6 | 155 | 11 / 11 | 22 / 22 | 50×180 (all equal) | grayscale |

Findings:

- **AMI packaging error.** Subject 104 is duplicated inside `test/102/104/`; the 7 files are byte-identical. The literature's "100 subjects / 700 images" is correct.
- **EarVN1.0 had 2 exact duplicate files.** `build_manifest` now drops byte-identical copies for every dataset, and drops every copy of a file shared by two identities.
- **Image size is 128×96 (h×w) everywhere.** Ears are about 1.4× taller than wide, so square inputs would distort them. EarVN images are tiny, so going larger only upsamples.
- **Data loading on Windows.** With no workers it runs at about 280 img/s, so an EarVN epoch takes about 70 s. DataLoader worker processes were *much slower* here, apparently from per-file overhead on Windows. For Phase 2: decode each small dataset into RAM once (EarVN at 128×96 is about 0.7 GB as uint8), augment on the GPU, and use `num_workers=0`.
- **Splits are tiny for AMI and AWE** (10 known and 10 unknown test ids). This confirms the decision to report mean ± std over 5 seeds.
