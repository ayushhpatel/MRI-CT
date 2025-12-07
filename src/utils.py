from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image
import nibabel as nib

from .models import ResnetGenerator


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


def preprocess_image(image: Image.Image, target_size: int = 256) -> torch.Tensor:
    """
    Preprocess PIL Image to tensor for CycleGAN inference.
    
    Args:
        image: PIL Image (grayscale or RGB)
        target_size: Target resolution (default 256)
        
    Returns:
        Preprocessed tensor in range [-1, 1] with shape (1, 1, H, W)
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to target size
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def load_nifti_slice(nifti_path: Path, slice_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load a 2D slice from a NIfTI file.
    
    Args:
        nifti_path: Path to .nii or .nii.gz file
        slice_idx: Specific slice index (if None, uses middle slice)
        
    Returns:
        Tuple of (2D numpy array, slice_index_used)
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    
    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = data.shape[2] // 2
    
    # Extract 2D slice
    slice_2d = data[:, :, slice_idx]
    
    # Normalize to 0-255 range for PIL compatibility
    slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255
    slice_2d = slice_2d.astype(np.uint8)
    
    return slice_2d, slice_idx


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor (1, 1, H, W) in range [-1, 1] to PIL Image.
    """
    # Remove batch and channel dimensions
    tensor = tensor.squeeze().cpu()
    
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    # Convert to numpy and scale to [0, 255]
    array = (tensor.numpy() * 255).astype(np.uint8)
    
    return Image.fromarray(array, mode='L')


def run_cyclegan_inference(
    mri_tensor: torch.Tensor, 
    checkpoint_path: Path,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run CycleGAN inference to generate synthetic CT and cycle-back MRI.
    
    Args:
        mri_tensor: Input MRI tensor (1, 1, H, W) in range [-1, 1]
        checkpoint_path: Path to trained checkpoint (.pt file)
        device: Device to run inference on (auto-detected if None)
        
    Returns:
        Tuple of (fake_ct_tensor, cycle_back_mri_tensor) both in range [-1, 1]
    """
    if device is None:
        device = get_device()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize generators (matching training configuration)
    G_AB = ResnetGenerator(input_nc=1, output_nc=1, ngf=64, n_blocks=9)  # MRI -> CT
    G_BA = ResnetGenerator(input_nc=1, output_nc=1, ngf=64, n_blocks=9)  # CT -> MRI
    
    # Load trained weights
    G_AB.load_state_dict(checkpoint['G_AB'])
    G_BA.load_state_dict(checkpoint['G_BA'])
    
    # Move to device and set to eval mode
    G_AB.to(device)
    G_BA.to(device)
    G_AB.eval()
    G_BA.eval()
    
    # Move input to device
    mri_tensor = mri_tensor.to(device)
    
    # Run inference without gradients
    with torch.no_grad():
        # MRI -> synthetic CT
        fake_ct = G_AB(mri_tensor)
        
        # Synthetic CT -> reconstructed MRI (cycle consistency)
        cycle_back_mri = G_BA(fake_ct)
    
    return fake_ct, cycle_back_mri


def save_inference_results(
    input_mri: torch.Tensor,
    fake_ct: torch.Tensor, 
    cycle_mri: torch.Tensor,
    output_dir: Path
) -> None:
    """
    Save inference results to individual PNG files.
    
    Args:
        input_mri: Original MRI tensor
        fake_ct: Generated CT tensor
        cycle_mri: Cycle-reconstructed MRI tensor
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to PIL Images and save
    input_pil = tensor_to_pil(input_mri)
    fake_ct_pil = tensor_to_pil(fake_ct)
    cycle_mri_pil = tensor_to_pil(cycle_mri)
    
    input_pil.save(output_dir / "input_mri.png")
    fake_ct_pil.save(output_dir / "fake_ct.png")
    cycle_mri_pil.save(output_dir / "cycle_mri.png")
