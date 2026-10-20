#!/bin/bash
# U0: build the UERC manifest (with exact-duplicate removal), splits for seeds 0-3 (+g5),
# and the 128x96 RAM cache (9.2 GB, parallel decoding). Reads the dataset; writes only to
# /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear_data/uerc/.
#   sbatch scripts/dgx/prepare_uerc.sbatch
#SBATCH --job-name=harsh_prep_uerc
#SBATCH --partition=dgx_all
#SBATCH --account=dsamantaai
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear
PY=/home/dsamantaai/miniconda3/envs/harsh_nlear/bin/python
PKG="/dgx1/dsamantaai/santhosh/Ear_Datasets/UERC 2023 Competition Package with sequestered anonymized data"
mkdir -p /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear_data/uerc

echo "== package README (first 80 lines)"
head -80 "$PKG/README" || true

echo "== manifest + splits ($(date +%H:%M:%S))"
"$PY" scripts/make_splits.py --config configs/uerc.yaml

echo "== source proxy per identity"
"$PY" - <<'EOF'
import pandas as pd
m = pd.read_csv("/dgx1/dsamantaai/santhosh/MTP_Harsh/nlear_data/uerc/manifest.csv", dtype=str)
print(m.groupby("source")["identity"].nunique().to_string())
for s in range(4):
    d = pd.read_csv(f"/dgx1/dsamantaai/santhosh/MTP_Harsh/nlear_data/uerc/splits_seed{s}.csv", dtype=str)
    print(f"seed {s}:", d.groupby(["split", "source"])["identity"].nunique().unstack(fill_value=0).to_dict("index"))
EOF

echo "== cache ($(date +%H:%M:%S))"
"$PY" scripts/dgx/prepare.py --data configs/uerc.yaml --workers 14
echo "== done ($(date +%H:%M:%S))"
