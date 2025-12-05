"""
Verify processed MRI/CT slices and report basic statistics.
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


def summarize_images(paths: List[Path]) -> Dict[str, float]:
    stats = []
    for p in paths:
        img = np.array(Image.open(p), dtype=np.float32)
        stats.append(
            {
                "mean": float(img.mean()),
                "std": float(img.std()),
                "min": float(img.min()),
                "max": float(img.max()),
            }
        )
    means = [s["mean"] for s in stats]
    stds = [s["std"] for s in stats]
    return {
        "count": len(paths),
        "mean_mean": float(np.mean(means)),
        "std_mean": float(np.std(means)),
        "mean_std": float(np.mean(stds)),
        "std_std": float(np.std(stds)),
    }


def main() -> None:
    manifest_path = Path("data/processed/manifest.json")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found at {manifest_path}. Run scripts/prepare_slices.py first.")
    manifest = load_manifest(manifest_path)
    mri_paths = [Path(r["path"]) for r in manifest.get("mri", [])]
    ct_paths = [Path(r["path"]) for r in manifest.get("ct", [])]
    if not mri_paths or not ct_paths:
        raise SystemExit("Manifest missing MRI or CT entries.")

    mri_summary = summarize_images(mri_paths)
    ct_summary = summarize_images(ct_paths)

    print("MRI slices:", mri_summary)
    print("CT slices:", ct_summary)
    # Simple sanity check: intensity spread should not be degenerate
    if mri_summary["mean_std"] < 5 or ct_summary["mean_std"] < 5:
        print("Warning: low variance detected; check preprocessing thresholds.")


if __name__ == "__main__":
    main()
