"""Prepare 2D PNG slices from CHAOS dataset volumes (NIfTI / DICOM).

Functionality:
- Input: data/raw/chaos/** (configurable)
- Output: data/chaos/mri_slices and data/chaos/ct_slices
- Modality detection by filename/folder (T1/T2/MR -> MRI, CT -> CT)
- NIfTI via nibabel, DICOM via pydicom (series stacking)
- Percentile clipping (1-99%), scaled to 0-255
- Skip slices with std < threshold
- Up to configurable max slices per scan
- Produces manifest.json at data/chaos/manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

import importlib
import subprocess
import sys

import pydicom   # force import, fail if missing

MRI_KEYWORDS = ("t1", "t2", "mr")
CT_KEYWORDS = ("ct",)


def detect_modality(path: Path) -> Optional[str]:
    s = "/".join(path.parts).lower()
    name = path.name.lower()
    if any(k in s or k in name for k in MRI_KEYWORDS):
        return "mri"
    if any(k in s or k in name for k in CT_KEYWORDS):
        return "ct"
    return None


def normalize_slice(slice_arr: np.ndarray) -> np.ndarray:
    low, high = np.percentile(slice_arr, (1.0, 99.0))
    clipped = np.clip(slice_arr, low, high)
    clipped = clipped - clipped.min()
    denom = clipped.max() if clipped.max() > 0 else 1.0
    scaled = (clipped / denom) * 255.0
    return scaled.astype(np.uint8)


def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    return np.asarray(data, dtype=np.float32)


def load_dicom_series(series_path: Path) -> np.ndarray:
    files = sorted(series_path.glob("*.dcm"))
    if not files:
        # maybe single .dcm file passed
        if series_path.suffix.lower() == ".dcm":
            files = [series_path]
        else:
            raise RuntimeError(f"No DICOM files found in {series_path}")
    slices = []
    meta = []
    for f in files:
        ds = pydicom.dcmread(str(f))
        arr = ds.pixel_array.astype(np.float32)
        slices.append(arr)
        meta.append(getattr(ds, "InstanceNumber", None))
    # sort by InstanceNumber if present
    if any(m is not None for m in meta):
        paired = sorted(zip(meta, slices), key=lambda x: (x[0] if x[0] is not None else 0))
        slices = [p[1] for p in paired]
    vol = np.stack(slices, axis=-1)
    return vol


def load_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".nii", ".gz") or path.name.lower().endswith(".nii.gz"):
        return load_nifti(path)
    if path.suffix.lower() == ".dcm":
        if pydicom is None:
            raise RuntimeError("pydicom is required to read DICOM files")
        return load_dicom_series(path)
    if path.is_dir():
        # Directory of DICOM files
        if pydicom is None:
            raise RuntimeError("pydicom is required to read DICOM directories")
        return load_dicom_series(path)
    # fallback try nifti
    return load_nifti(path)


def slice_volume(
    vol: np.ndarray,
    modality: str,
    subject: str,
    out_dir: Path,
    target_size: int,
    min_std: float,
    max_slices: Optional[int],
) -> List[Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # prefer last axis as slice axis if it's reasonably sized
    if vol.ndim == 3:
        z = vol.shape[2]
        axes = 2
    elif vol.ndim == 2:
        # single slice
        vol = vol[:, :, np.newaxis]
        z = 1
        axes = 2
    else:
        # try to find an axis with size > 1
        z = vol.shape[-1]
        axes = vol.ndim - 1

    indices = list(range(z))
    if max_slices and len(indices) > max_slices:
        step = max(1, len(indices) // max_slices)
        indices = indices[::step][:max_slices]

    records: List[Dict] = []
    for i in indices:
        slc = vol[:, :, i]
        if np.std(slc) < min_std:
            continue
        norm = normalize_slice(slc)
        img = Image.fromarray(norm).convert("L")
        img = img.resize((target_size, target_size), Image.BILINEAR)
        fname = f"chaos_{modality}_{subject}_slice_{i:03d}.png"
        out_path = out_dir / fname
        img.save(out_path)
        records.append({
            "path": str(out_path),
            "subject": subject,
            "slice": int(i),
        })
    return records


def collect_inputs(raw_root: Path) -> List[Path]:
    patterns = ("*.nii", "*.nii.gz", "*.dcm")
    found: List[Path] = []
    for pat in patterns:
        found.extend(raw_root.rglob(pat))
    # For DICOM directories, include folders that contain .dcm files
    for d in raw_root.rglob("*"):
        if d.is_dir() and any(d.glob("*.dcm")):
            found.append(d)
    # make unique and sort
    uniq = sorted(set(found), key=lambda p: str(p))
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CHAOS slices")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/chaos"))
    parser.add_argument("--mri_out", type=Path, default=Path("data/chaos/mri_slices"))
    parser.add_argument("--ct_out", type=Path, default=Path("data/chaos/ct_slices"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--min-std", type=float, default=5.0)
    parser.add_argument("--max-slices", type=int, default=120)
    args = parser.parse_args()

    inputs = collect_inputs(args.raw)
    if not inputs:
        raise SystemExit(f"No volumes found under {args.raw}")

    manifest: Dict[str, List[Dict]] = {"mri": [], "ct": []}
    for p in tqdm(inputs, desc="Processing volumes"):
        modality = detect_modality(p)
        if modality is None:
            # try to infer from parent folder
            modality = detect_modality(p.parent) or "mri"
        subject = p.stem if p.is_file() else p.name
        if modality == "mri":
            out_dir = args.mri_out
        else:
            out_dir = args.ct_out
        try:
            vol = load_volume(p)
        except Exception as e:
            print(f"Skipping {p}: could not load volume ({e})")
            continue
        records = slice_volume(
            vol=vol,
            modality=modality,
            subject=subject,
            out_dir=out_dir,
            target_size=args.size,
            min_std=args.min_std,
            max_slices=args.max_slices,
        )
        manifest[modality].extend(records)

    # ensure output folders exist
    args.mri_out.mkdir(parents=True, exist_ok=True)
    args.ct_out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("data/chaos/manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Wrote manifest: {manifest_path}")
    print(f"MRI slices: {len(manifest['mri'])}, CT slices: {len(manifest['ct'])}")


if __name__ == "__main__":
    main()
