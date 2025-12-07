"""Verify CHAOS dataset and report counts and intensity statistics.

Supports both manifest-based (data/chaos/) and direct directory (data/chaos_3k/) datasets.
Emits warnings when dataset is small or heavily unbalanced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_manifest(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def compute_stats(paths: List[Path], sample_size: int = None) -> Dict[str, float]:
    """Compute statistics for a list of image paths, optionally sampling for speed."""
    if sample_size and len(paths) > sample_size:
        # Sample evenly across the dataset
        step = len(paths) // sample_size
        sampled_paths = paths[::step][:sample_size]
        print(f"  Sampling {len(sampled_paths)} images from {len(paths)} total...")
    else:
        sampled_paths = paths
    
    total_pixels = 0
    sum_pix = 0.0
    sum_sq = 0.0
    count = 0
    
    for p in tqdm(sampled_paths, desc="Computing stats", leave=False):
        try:
            img = np.array(Image.open(p), dtype=np.float32)
            pixels = img.size
            total_pixels += pixels
            sum_pix += float(img.sum())
            sum_sq += float((img ** 2).sum())
            count += 1
        except Exception as e:
            print(f"Warning: Could not load {p}: {e}")
            continue
    
    if total_pixels == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    mean = sum_pix / total_pixels
    var = (sum_sq / total_pixels) - (mean ** 2)
    std = float(np.sqrt(var)) if var > 0 else 0.0
    
    # Compute min/max from a smaller sample for efficiency
    sample_for_minmax = sampled_paths[:50] if len(sampled_paths) > 50 else sampled_paths
    all_values = []
    for p in sample_for_minmax:
        try:
            img = np.array(Image.open(p), dtype=np.float32)
            all_values.extend(img.flatten())
        except:
            continue
    
    if all_values:
        min_val = float(np.min(all_values))
        max_val = float(np.max(all_values))
    else:
        min_val = max_val = 0.0
    
    return {
        "count": len(paths),  # Return full count, not just processed
        "processed": count,   # Actually processed files
        "mean": float(mean), 
        "std": float(std),
        "min": min_val,
        "max": max_val
    }


def verify_manifest_dataset(manifest_path: Path, sample_size: int = None) -> None:
    """Verify dataset using manifest.json structure."""
    print(f"📄 Using manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    mri_entries = manifest.get("mri", [])
    ct_entries = manifest.get("ct", [])
    mri_paths = [Path(r["path"]) for r in mri_entries]
    ct_paths = [Path(r["path"]) for r in ct_entries]
    
    print(f"🧠 Processing {len(mri_paths)} MRI images...")
    mri_stats = compute_stats(mri_paths, sample_size)
    print(f"🏥 Processing {len(ct_paths)} CT images...")
    ct_stats = compute_stats(ct_paths, sample_size)
    
    print_results("Manifest-based Dataset", mri_stats, ct_stats)


def verify_directory_dataset(dataset_dir: Path, sample_size: int = None) -> None:
    """Verify dataset using direct directory structure (mri/, ct/)."""
    mri_dir = dataset_dir / "mri"
    ct_dir = dataset_dir / "ct"
    
    if not mri_dir.exists() or not ct_dir.exists():
        raise SystemExit(f"Expected directories not found: {mri_dir} and {ct_dir}")
    
    print(f"📁 Using directory: {dataset_dir}")
    
    # Find image files
    mri_patterns = ["*.png", "*.jpg", "*.jpeg"]
    ct_patterns = ["*.png", "*.jpg", "*.jpeg"]
    
    mri_paths = []
    for pattern in mri_patterns:
        mri_paths.extend(mri_dir.glob(pattern))
    
    ct_paths = []
    for pattern in ct_patterns:
        ct_paths.extend(ct_dir.glob(pattern))
    
    print(f"🧠 Processing {len(mri_paths)} MRI images...")
    mri_stats = compute_stats(mri_paths, sample_size)
    print(f"🏥 Processing {len(ct_paths)} CT images...")
    ct_stats = compute_stats(ct_paths, sample_size)
    
    dataset_name = f"Directory Dataset ({dataset_dir.name})"
    print_results(dataset_name, mri_stats, ct_stats)


def print_results(dataset_name: str, mri_stats: Dict, ct_stats: Dict) -> None:
    """Print formatted results and warnings."""
    print(f"\n{'='*60}")
    print(f"📊 {dataset_name.upper()} VERIFICATION RESULTS")
    print(f"{'='*60}")
    
    print(f"🧠 MRI: {mri_stats['count']} slices")
    print(f"   Mean: {mri_stats['mean']:.3f}, Std: {mri_stats['std']:.3f}")
    print(f"   Range: [{mri_stats['min']:.1f}, {mri_stats['max']:.1f}]")
    if mri_stats['processed'] != mri_stats['count']:
        print(f"   (Processed: {mri_stats['processed']}/{mri_stats['count']})")
    
    print(f"🏥 CT: {ct_stats['count']} slices")
    print(f"   Mean: {ct_stats['mean']:.3f}, Std: {ct_stats['std']:.3f}")
    print(f"   Range: [{ct_stats['min']:.1f}, {ct_stats['max']:.1f}]")
    if ct_stats['processed'] != ct_stats['count']:
        print(f"   (Processed: {ct_stats['processed']}/{ct_stats['count']})")
    
    print(f"\n📈 DATASET QUALITY ASSESSMENT")
    print(f"{'-'*40}")
    
    # Size warnings
    if mri_stats["count"] < 500 or ct_stats["count"] < 500:
        print("⚠️  Small dataset: fewer than 500 slices for MRI or CT")
    else:
        print("✅ Large dataset: sufficient samples for deep learning")
    
    # Balance check (10% tolerance)
    if mri_stats["count"] > 0 and ct_stats["count"] > 0:
        ratio = mri_stats["count"] / max(1, ct_stats["count"])
        if 0.9 <= ratio <= 1.1:
            print("✅ Balanced dataset: MRI and CT counts are similar")
        else:
            print(f"⚠️  Imbalanced dataset: MRI/CT ratio = {ratio:.2f}")
        
        # Perfect balance check
        if mri_stats["count"] == ct_stats["count"]:
            print("🎯 Perfect balance: identical MRI and CT counts")
    
    # Intensity range checks
    total_samples = mri_stats["count"] + ct_stats["count"]
    print(f"📊 Total samples: {total_samples:,}")
    
    if mri_stats["std"] > 0 and ct_stats["std"] > 0:
        print("✅ Good variation: both modalities show intensity diversity")
    
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CHAOS dataset")
    parser.add_argument("--dataset", type=str, 
                       choices=["chaos", "chaos_3k"], 
                       default="chaos_3k",
                       help="Dataset to verify")
    parser.add_argument("--sample-size", type=int, default=500,
                       help="Sample size for statistics (None = use all)")
    args = parser.parse_args()
    
    if args.dataset == "chaos":
        # Original manifest-based dataset
        manifest_path = Path("data/chaos/manifest.json")
        if not manifest_path.exists():
            raise SystemExit(f"Manifest not found at {manifest_path}. Run scripts/prepare_slices.py first.")
        verify_manifest_dataset(manifest_path, args.sample_size)
    
    elif args.dataset == "chaos_3k":
        # Enhanced directory-based dataset
        dataset_dir = Path("data/chaos_3k")
        if not dataset_dir.exists():
            raise SystemExit(f"Dataset directory not found: {dataset_dir}")
        verify_directory_dataset(dataset_dir, args.sample_size)


if __name__ == "__main__":
    main()
