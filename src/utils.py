from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torchvision.utils import save_image, make_grid


def get_device(force_cpu: bool = False) -> torch.device:
    # Modified: Return CUDA if available unless forced to use CPU
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_sample_grid(images: List[torch.Tensor], path: Path, nrow: int = 4) -> None:
    """
    Save a list of images (C,H,W tensors in [-1,1]) as a grid for quick inspection.
    """
    stacked = torch.stack(images, dim=0)
    grid = make_grid(stacked, nrow=nrow, normalize=True, value_range=(-1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)
