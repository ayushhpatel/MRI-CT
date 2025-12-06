"""Generate high-resolution fake CT and reconstructed MRI images.

This script loads a trained CycleGAN model and generates high-quality fake images
with the MRI -> fake CT -> reconstructed MRI pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image

from .models import ResnetGenerator
from .datasets import list_images, EvaluationDataset


def load_generators(checkpoint_path: Path, device: torch.device, n_blocks: int = 6, ngf: int = 64) -> tuple[ResnetGenerator, ResnetGenerator]:
    """Load trained CycleGAN generators from checkpoint."""
    G_AB = ResnetGenerator(n_blocks=n_blocks, ngf=ngf)  # MRI -> CT
    G_BA = ResnetGenerator(n_blocks=n_blocks, ngf=ngf)  # CT -> MRI
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    G_AB.load_state_dict(checkpoint["G_AB"])
    G_BA.load_state_dict(checkpoint["G_BA"])
    
    G_AB.to(device).eval()
    G_BA.to(device).eval()
    
    return G_AB, G_BA


def save_high_res_image(tensor: torch.Tensor, path: Path, denormalize: bool = True) -> None:
    """Save tensor as high-quality image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if denormalize:
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1.0) / 2.0
    
    # Clamp and convert to PIL for high-quality saving
    img_np = tensor.cpu().squeeze().clamp(0, 1).numpy()
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8), mode='L')
    img_pil.save(path, quality=95, optimize=True)


def generate_high_res_cycle(
    checkpoint_path: Path,
    mri_dir: Path,
    output_dir: Path,
    n_samples: int = 20,
    image_size: int = 384,
    n_blocks: int = 6,
    ngf: int = 64,
    device: Optional[torch.device] = None
) -> None:
    """Generate high-resolution MRI -> fake CT -> reconstructed MRI sequences."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Target resolution: {image_size}x{image_size}")
    
    # Load generators
    G_AB, G_BA = load_generators(checkpoint_path, device, n_blocks, ngf)
    
    # Prepare high-resolution transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load MRI images
    mri_paths = list_images(mri_dir)
    if len(mri_paths) < n_samples:
        print(f"Only {len(mri_paths)} MRI images available, generating all")
        n_samples = len(mri_paths)
    
    selected_mri = random.sample(mri_paths, n_samples)
    print(f"Generating high-res cycles for {n_samples} MRI samples")
    
    # Create output directories
    original_dir = output_dir / "original_mri"
    fake_ct_dir = output_dir / "fake_ct_high_res"
    reconstructed_dir = output_dir / "reconstructed_mri"
    
    with torch.no_grad():
        for i, mri_path in enumerate(selected_mri):
            print(f"Processing {i+1}/{n_samples}: {mri_path.name}")
            
            # Load and transform MRI
            mri_img = Image.open(mri_path).convert("L")
            mri_tensor = transform(mri_img).unsqueeze(0).to(device)
            
            # Generate fake CT (high resolution)
            fake_ct = G_AB(mri_tensor)
            
            # Reconstruct MRI from fake CT
            reconstructed_mri = G_BA(fake_ct)
            
            # Save all images with descriptive names
            base_name = f"{i:03d}_{mri_path.stem}"
            
            save_high_res_image(
                mri_tensor[0], 
                original_dir / f"{base_name}_original.png"
            )
            save_high_res_image(
                fake_ct[0], 
                fake_ct_dir / f"{base_name}_fake_ct.png"
            )
            save_high_res_image(
                reconstructed_mri[0], 
                reconstructed_dir / f"{base_name}_reconstructed.png"
            )
            
            # Save comparison triplet
            comparison_dir = output_dir / "comparisons"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # Create side-by-side comparison
            triplet = torch.cat([
                (mri_tensor[0] + 1) / 2,
                (fake_ct[0] + 1) / 2, 
                (reconstructed_mri[0] + 1) / 2
            ], dim=2)
            
            save_image(
                triplet,
                comparison_dir / f"{base_name}_triplet.png",
                normalize=False
            )
    
    print(f"\\nHigh-resolution generation complete!")
    print(f"Output saved to: {output_dir}")
    print(f"- Original MRI: {original_dir}")
    print(f"- Fake CT (high-res): {fake_ct_dir}")
    print(f"- Reconstructed MRI: {reconstructed_dir}")
    print(f"- Comparisons: {comparison_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate high-resolution fake CT and reconstructed MRI")
    parser.add_argument("--checkpoint", type=Path, required=True,
                       help="Path to trained CycleGAN checkpoint")
    parser.add_argument("--mri-dir", type=Path, default=Path("data/chaos/mri_slices"),
                       help="Directory containing MRI images")
    parser.add_argument("--output-dir", type=Path, default=Path("high_res_generation"),
                       help="Directory to save generated images")
    parser.add_argument("--n-samples", type=int, default=20,
                       help="Number of MRI samples to process")
    parser.add_argument("--image-size", type=int, default=384,
                       help="Target image resolution (e.g., 384 for 384x384)")
    parser.add_argument("--n-blocks", type=int, default=6,
                       help="Number of ResNet blocks in generator")
    parser.add_argument("--ngf", type=int, default=64,
                       help="Number of generator features")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    if not args.mri_dir.exists():
        raise FileNotFoundError(f"MRI directory not found: {args.mri_dir}")
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    generate_high_res_cycle(
        checkpoint_path=args.checkpoint,
        mri_dir=args.mri_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        image_size=args.image_size,
        n_blocks=args.n_blocks,
        ngf=args.ngf,
        device=device
    )


if __name__ == "__main__":
    main()