from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])


class UnpairedImageDataset(Dataset):
    """
    Unpaired dataset for CycleGAN: returns one MRI slice and one CT slice.
    """

    def __init__(
        self,
        mri_dir: Path,
        ct_dir: Path,
        transform: Callable,
    ) -> None:
        self.mri_paths = list_images(mri_dir)
        self.ct_paths = list_images(ct_dir)
        if not self.mri_paths:
            raise ValueError(f"No MRI images found in {mri_dir}")
        if not self.ct_paths:
            raise ValueError(f"No CT images found in {ct_dir}")
        self.transform = transform

    def __len__(self) -> int:
        return max(len(self.mri_paths), len(self.ct_paths))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mri_path = self.mri_paths[idx % len(self.mri_paths)]
        ct_path = random.choice(self.ct_paths)
        mri_img = Image.open(mri_path).convert("L")
        ct_img = Image.open(ct_path).convert("L")
        return self.transform(mri_img), self.transform(ct_img)
