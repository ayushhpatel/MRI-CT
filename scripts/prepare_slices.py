"""
Convert downloaded MRI/CT volumes into 2D PNG slices for CycleGAN training.
Usage:
    python scripts/prepare_slices.py --raw data/raw --out data/processed --size 256 --max-slices 80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

MRI_KEYWORDS = ("ixi", "t1", "t2", "prostate")
CT_KEYWORDS = ("ct", "spleen", "copd")


def classify_modality(path: Path) -> str:
    name = path.name.lower()
    if any(k in name for k in MRI_KEYWORDS):
        return "mri"
    if any(k in name for k in CT_KEYWORDS):
        return "ct"
    raise ValueError(f"Could not classify modality for {path}")


def normalize_slice(slice_arr: np.ndarray) -> np.ndarray:
    # percentile clipping to limit outliers, then min-max to 0-255
    low, high = np.percentile(slice_arr, (1, 99))
    clipped = np.clip(slice_arr, low, high)
    clipped = clipped - clipped.min()
    denom = clipped.max() if clipped.max() > 0 else 1.0
    scaled = (clipped / denom) * 255.0
    return scaled.astype(np.uint8)


def load_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    if data.ndim == 4:  # drop channels if present
        data = data[..., 0]
    return np.asarray(data, dtype=np.float32)


def slice_volume(
    vol: np.ndarray,
    modality: str,
    source: Path,
    out_dir: Path,
    target_size: int,
    min_std: float,
    max_slices: int | None,
) -> List[Dict]:
    out_records: List[Dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    axes = (2 if vol.shape[2] >= vol.shape[0] else 0)  # heuristically prefer axial
    indices = list(range(vol.shape[axes]))
    if max_slices:
        step = max(1, len(indices) // max_slices)
        indices = indices[::step][:max_slices]

    for idx in indices:
        if axes == 2:
            slc = vol[:, :, idx]
        elif axes == 0:
            slc = vol[idx, :, :]
        else:
            slc = vol[:, idx, :]
        if np.std(slc) < min_std:
            continue
        norm = normalize_slice(slc)
        img = Image.fromarray(norm)
        img = img.convert("L")
        img = img.resize((target_size, target_size), Image.BILINEAR)
        fname = f"{source.stem}_z{idx:03d}.png"
        out_path = out_dir / fname
        img.save(out_path)
        out_records.append(
            {
                "path": str(out_path),
                "source": str(source),
                "slice_index": int(idx),
                "modality": modality,
                "mean": float(np.mean(norm)),
                "std": float(np.std(norm)),
            }
        )
    return out_records


def collect_volumes(raw_dir: Path) -> List[Path]:
    patterns = ("*.nii", "*.nii.gz")
    files: List[Path] = []
    for pat in patterns:
        files.extend(raw_dir.rglob(pat))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 2D slices from MRI/CT volumes.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=256, help="Output square image size.")
    parser.add_argument("--min-std", type=float, default=5.0, help="Discard slices with low variation.")
    parser.add_argument("--max-slices", type=int, default=80, help="Limit slices per volume (per axis).")
    args = parser.parse_args()

    volumes = collect_volumes(args.raw)
    if not volumes:
        raise SystemExit(f"No NIfTI files found in {args.raw}")

    manifest: Dict[str, List[Dict]] = {"mri": [], "ct": []}
    for vol_path in tqdm(volumes, desc="Volumes"):
        modality = classify_modality(vol_path)
        out_subdir = args.out / modality
        vol = load_volume(vol_path)
        records = slice_volume(
            vol=vol,
            modality=modality,
            source=vol_path,
            out_dir=out_subdir,
            target_size=args.size,
            min_std=args.min_std,
            max_slices=args.max_slices,
        )
        manifest[modality].extend(records)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest with {len(manifest['mri'])} MRI slices and {len(manifest['ct'])} CT slices to {manifest_path}")


if __name__ == "__main__":
    main()
