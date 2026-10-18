from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(img_size, train):
    h, w = img_size
    if train:
        ops = [
            T.Resize((h, w)),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(0.2, 0.2, 0.2),
        ]
        # No horizontal flip by default: a flipped left ear looks like a right ear.
        # Whether flipping helps is an ablation for later, once we know UERC's left/right mix.
    else:
        ops = [T.Resize((h, w))]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class EarDataset(Dataset):
    """Rows of a splits CSV, filtered by split/role/known.

    Returns (image, label, known). `label` is a contiguous index over the identities
    present in this filtered set, so the train set can feed a classification head directly.
    """

    def __init__(self, splits_csv, root, split, roles=None, known=None, transform=None):
        df = splits_csv if isinstance(splits_csv, pd.DataFrame) else \
            pd.read_csv(splits_csv, dtype={"identity": str, "path": str})
        df = df[df["split"] == split]
        if roles is not None:
            df = df[df["role"].isin(roles)]
        if known is not None:
            df = df[df["known"] == known]
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.identities = sorted(self.df["identity"].unique())
        self.id_to_label = {i: n for n, i in enumerate(self.identities)}

    def __len__(self):
        return len(self.df)

    @property
    def num_classes(self):
        return len(self.identities)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.id_to_label[row["identity"]], torch.tensor(int(row["known"]))
