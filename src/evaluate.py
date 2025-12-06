"""Evaluation pipeline for CycleGAN MRI <-> CT translation.

This script evaluates cycle consistency using SSIM and PSNR metrics on unpaired data.
Performs MRI → fake CT → reconstructed MRI and CT → fake MRI → reconstructed CT cycles.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from .models import ResnetGenerator
from .datasets import list_images


class EvaluationDataset(Dataset):
    """Dataset for evaluation - loads individual MRI and CT images."""
    
    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, path.stem


def load_generators(checkpoint_path: Path, device: torch.device, n_blocks: int = 9) -> Tuple[ResnetGenerator, ResnetGenerator]:
    """Load trained CycleGAN generators from checkpoint."""
    G_AB = ResnetGenerator(n_blocks=n_blocks)  # MRI -> CT
    G_BA = ResnetGenerator(n_blocks=n_blocks)  # CT -> MRI
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    G_AB.load_state_dict(checkpoint["G_AB"])
    G_BA.load_state_dict(checkpoint["G_BA"])
    
    G_AB.to(device).eval()
    G_BA.to(device).eval()
    
    return G_AB, G_BA


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for SSIM/PSNR computation."""
    # Denormalize from [-1, 1] to [0, 1]
    img = (tensor.cpu().squeeze() + 1.0) / 2.0
    return img.clamp(0, 1).numpy()


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Tuple[float, float]:
    """Compute SSIM and PSNR between original and reconstructed images."""
    orig_np = tensor_to_numpy(original)
    recon_np = tensor_to_numpy(reconstructed)
    
    # SSIM with data range [0, 1]
    ssim_val = ssim(orig_np, recon_np, data_range=1.0)
    
    # PSNR with data range [0, 1]
    psnr_val = psnr(orig_np, recon_np, data_range=1.0)
    
    return float(ssim_val), float(psnr_val)


def create_visualization_grid(
    images: List[torch.Tensor], 
    titles: List[str], 
    save_path: Path,
    ncols: int = 6
) -> None:
    """Create and save visualization grid."""
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row, col = i // ncols, i % ncols
        img_np = tensor_to_numpy(img)
        axes[row, col].imshow(img_np, cmap='gray')
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(images), nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_cycle_consistency(
    checkpoint_path: Path,
    mri_dir: Path,
    ct_dir: Path,
    output_dir: Path,
    n_samples: int = 50,
    n_blocks: int = 9,
    device: torch.device = None
) -> None:
    """Main evaluation function."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Load generators
    G_AB, G_BA = load_generators(checkpoint_path, device, n_blocks)
    
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mri_paths = list_images(mri_dir)
    ct_paths = list_images(ct_dir)
    
    print(f"Found {len(mri_paths)} MRI images, {len(ct_paths)} CT images")
    
    if len(mri_paths) < n_samples or len(ct_paths) < n_samples:
        print(f"Warning: Requested {n_samples} samples but only have {len(mri_paths)} MRI and {len(ct_paths)} CT images")
        n_samples = min(n_samples, len(mri_paths), len(ct_paths))
    
    # Randomly sample images
    selected_mri = random.sample(mri_paths, n_samples)
    selected_ct = random.sample(ct_paths, n_samples)
    
    mri_dataset = EvaluationDataset(selected_mri, transform)
    ct_dataset = EvaluationDataset(selected_ct, transform)
    
    mri_loader = DataLoader(mri_dataset, batch_size=1, shuffle=False)
    ct_loader = DataLoader(ct_dataset, batch_size=1, shuffle=False)
    
    # Results storage
    results = []
    vis_images = []
    vis_titles = []
    
    print(f"\\nEvaluating cycle consistency on {n_samples} samples...")
    
    # Evaluate MRI -> CT -> MRI cycle
    print("Processing MRI -> fake CT -> reconstructed MRI...")
    mri_ssims, mri_psnrs = [], []
    
    with torch.no_grad():
        for i, (mri_img, mri_name) in enumerate(mri_loader):
            if i >= n_samples:
                break
                
            mri_img = mri_img.to(device)
            
            # Forward cycle: MRI -> fake CT -> reconstructed MRI
            fake_ct = G_AB(mri_img)
            recon_mri = G_BA(fake_ct)
            
            # Compute metrics
            ssim_val, psnr_val = compute_metrics(mri_img, recon_mri)
            mri_ssims.append(ssim_val)
            mri_psnrs.append(psnr_val)
            
            results.append({
                'sample_id': f'mri_{i:03d}',
                'filename': mri_name[0],
                'direction': 'MRI->CT->MRI',
                'ssim': ssim_val,
                'psnr': psnr_val
            })
            
            # Collect images for visualization (first 6 samples)
            if i < 6:
                vis_images.extend([
                    mri_img[0], fake_ct[0], recon_mri[0]
                ])
                vis_titles.extend([
                    f'Real MRI {i+1}', f'Fake CT {i+1}', f'Recon MRI {i+1}'
                ])
    
    # Evaluate CT -> MRI -> CT cycle
    print("Processing CT -> fake MRI -> reconstructed CT...")
    ct_ssims, ct_psnrs = [], []
    
    with torch.no_grad():
        for i, (ct_img, ct_name) in enumerate(ct_loader):
            if i >= n_samples:
                break
                
            ct_img = ct_img.to(device)
            
            # Reverse cycle: CT -> fake MRI -> reconstructed CT
            fake_mri = G_BA(ct_img)
            recon_ct = G_AB(fake_mri)
            
            # Compute metrics
            ssim_val, psnr_val = compute_metrics(ct_img, recon_ct)
            ct_ssims.append(ssim_val)
            ct_psnrs.append(psnr_val)
            
            results.append({
                'sample_id': f'ct_{i:03d}',
                'filename': ct_name[0],
                'direction': 'CT->MRI->CT',
                'ssim': ssim_val,
                'psnr': psnr_val
            })
            
            # Collect images for visualization (first 6 samples)
            if i < 6:
                vis_images.extend([
                    ct_img[0], fake_mri[0], recon_ct[0]
                ])
                vis_titles.extend([
                    f'Real CT {i+1}', f'Fake MRI {i+1}', f'Recon CT {i+1}'
                ])
    
    # Save results to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cycle_consistency_results.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'filename', 'direction', 'ssim', 'psnr'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved detailed results to {csv_path}")
    
    # Create visualization grid
    vis_path = output_dir / "cycle_consistency_visualization.png"
    create_visualization_grid(vis_images, vis_titles, vis_path, ncols=9)
    print(f"Saved visualization to {vis_path}")
    
    # Print summary statistics
    print(f"\\n{'='*60}")
    print("CYCLE CONSISTENCY EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Samples evaluated: {n_samples} per direction")
    print(f"")
    print(f"MRI → fake CT → reconstructed MRI:")
    print(f"  Mean SSIM: {np.mean(mri_ssims):.4f} ± {np.std(mri_ssims):.4f}")
    print(f"  Mean PSNR: {np.mean(mri_psnrs):.2f} ± {np.std(mri_psnrs):.2f} dB")
    print(f"")
    print(f"CT → fake MRI → reconstructed CT:")
    print(f"  Mean SSIM: {np.mean(ct_ssims):.4f} ± {np.std(ct_ssims):.4f}")
    print(f"  Mean PSNR: {np.mean(ct_psnrs):.2f} ± {np.std(ct_psnrs):.2f} dB")
    print(f"")
    print(f"Overall cycle consistency:")
    all_ssims = mri_ssims + ct_ssims
    all_psnrs = mri_psnrs + ct_psnrs
    print(f"  Mean SSIM: {np.mean(all_ssims):.4f} ± {np.std(all_ssims):.4f}")
    print(f"  Mean PSNR: {np.mean(all_psnrs):.2f} ± {np.std(all_psnrs):.2f} dB")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN cycle consistency")
    parser.add_argument("--checkpoint", type=Path, required=True,
                       help="Path to trained CycleGAN checkpoint")
    parser.add_argument("--mri-dir", type=Path, default=Path("data/chaos/mri_slices"),
                       help="Directory containing MRI images")
    parser.add_argument("--ct-dir", type=Path, default=Path("data/chaos/ct_slices"), 
                       help="Directory containing CT images")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"),
                       help="Directory to save evaluation results")
    parser.add_argument("--n-samples", type=int, default=50,
                       help="Number of samples to evaluate per direction")
    parser.add_argument("--n-blocks", type=int, default=9,
                       help="Number of ResNet blocks in generator")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu). Auto-detect if not specified")
    
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    if not args.mri_dir.exists():
        raise FileNotFoundError(f"MRI directory not found: {args.mri_dir}")
        
    if not args.ct_dir.exists():
        raise FileNotFoundError(f"CT directory not found: {args.ct_dir}")
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    evaluate_cycle_consistency(
        checkpoint_path=args.checkpoint,
        mri_dir=args.mri_dir,
        ct_dir=args.ct_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        n_blocks=args.n_blocks,
        device=device
    )


if __name__ == "__main__":
    main()