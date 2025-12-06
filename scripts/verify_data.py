"""Verify CHAOS manifest and report counts and intensity statistics.

Loads manifest from data/chaos/manifest.json and prints counts and mean/std
for MRI and CT. Emits warnings when dataset is small or heavily unbalanced.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def load_manifest(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def compute_stats(paths: List[Path]) -> Dict[str, float]:
    total_pixels = 0
    sum_pix = 0.0
    sum_sq = 0.0
    count = 0
    for p in paths:
        img = np.array(Image.open(p), dtype=np.float32)
        pixels = img.size
        total_pixels += pixels
        sum_pix += float(img.sum())
        sum_sq += float((img ** 2).sum())
        count += 1
    if total_pixels == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0}
    mean = sum_pix / total_pixels
    var = (sum_sq / total_pixels) - (mean ** 2)
    std = float(np.sqrt(var)) if var > 0 else 0.0
    return {"count": count, "mean": float(mean), "std": float(std)}


def main() -> None:
    manifest_path = Path("data/chaos/manifest.json")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found at {manifest_path}. Run scripts/prepare_slices.py first.")
    manifest = load_manifest(manifest_path)
    mri_entries = manifest.get("mri", [])
    ct_entries = manifest.get("ct", [])
    mri_paths = [Path(r["path"]) for r in mri_entries]
    ct_paths = [Path(r["path"]) for r in ct_entries]

    mri_stats = compute_stats(mri_paths)
    ct_stats = compute_stats(ct_paths)

    print(f"MRI: {mri_stats['count']} slices, mean={mri_stats['mean']:.3f}, std={mri_stats['std']:.3f}")
    print(f"CT:  {ct_stats['count']} slices, mean={ct_stats['mean']:.3f}, std={ct_stats['std']:.3f}")

    # Warnings
    if mri_stats["count"] < 500 or ct_stats["count"] < 500:
        print("Warning: fewer than 500 slices for MRI or CT — dataset may be too small.")
    # check balance (10% tolerance)
    if mri_stats["count"] > 0 and ct_stats["count"] > 0:
        ratio = mri_stats["count"] / max(1, ct_stats["count"])
        if ratio < 0.9 or ratio > 1.1:
            print("Warning: dataset appears unbalanced (MRI vs CT counts differ by >10%).")


if __name__ == "__main__":
    main()
