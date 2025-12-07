from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import numpy as np

import torch
from torchvision import transforms
from PIL import Image
import nibabel as nib

from .models import ResnetGenerator


def preprocess_image(image: Image.Image, target_size: int = 256) -> torch.Tensor:
    """
    Unified preprocessing for CycleGAN inference (matches training pipeline).
    
    Args:
        image: PIL Image (grayscale or RGB)
        target_size: Target resolution (default 256)
        
    Returns:
        Preprocessed tensor in range [-1, 1] with shape (1, 1, H, W)
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Deterministic resize using BICUBIC interpolation
    image = image.resize((target_size, target_size), Image.BICUBIC)
    
    # Convert to tensor and normalize to [-1, 1]
    # Using (x - 0.5) / 0.5 transformation (matches training)
    tensor = transforms.ToTensor()(image)
    tensor = (tensor - 0.5) / 0.5
    
    # Add batch dimension: (1, 1, H, W)
    tensor = tensor.unsqueeze(0)
    return tensor


def postprocess_tensor(tensor: torch.Tensor) -> Image.Image:
    """
    Unified postprocessing: convert model output tensor [-1,1] to viewable PIL image.
    
    Args:
        tensor: Model output tensor in range [-1, 1]
        
    Returns:
        PIL Image (grayscale, 8-bit)
    """
    # Remove batch and channel dimensions, move to CPU
    tensor = tensor.squeeze().detach().cpu().clamp(-1.0, 1.0)
    
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) * 0.5
    
    # Convert to numpy and scale to [0, 255]
    array = (tensor.numpy() * 255.0).astype(np.uint8)
    
    return Image.fromarray(array, mode='L')


class CachedCycleGANInference:
    """Cached CycleGAN model for efficient slice-by-slice inference."""
    
    def __init__(self, checkpoint_path: Path, device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.checkpoint_path = checkpoint_path
        
        # Initialize generators with exact training architecture
        self.G_AB = ResnetGenerator(input_nc=1, output_nc=1, ngf=64, n_blocks=9).to(self.device).eval()
        self.G_BA = ResnetGenerator(input_nc=1, output_nc=1, ngf=64, n_blocks=9).to(self.device).eval()
        
        # Load checkpoint weights
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint with robust key handling for different checkpoint formats."""
        try:
            state = torch.load(self.checkpoint_path, map_location=self.device)
            
            if isinstance(state, dict):
                # Try different checkpoint key formats
                if "G_AB" in state and "G_BA" in state:
                    # Format 1: G_AB/G_BA
                    self.G_AB.load_state_dict(state["G_AB"])
                    self.G_BA.load_state_dict(state["G_BA"])
                elif "G_AtoB" in state and "G_BtoA" in state:
                    # Format 2: G_AtoB/G_BtoA
                    self.G_AB.load_state_dict(state["G_AtoB"])
                    self.G_BA.load_state_dict(state["G_BtoA"])
                elif len(state.keys()) == 1:
                    # Single generator checkpoint
                    key = list(state.keys())[0]
                    self.G_AB.load_state_dict(state[key])
                else:
                    # Assume flat checkpoint is for G_AB
                    self.G_AB.load_state_dict(state)
            else:
                # Direct state dict for G_AB
                self.G_AB.load_state_dict(state)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {self.checkpoint_path}: {e}")
    
    @torch.no_grad()
    def infer_slice(self, mri_tensor: torch.Tensor, do_cycle: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run inference on a single MRI slice tensor.
        
        Args:
            mri_tensor: Input MRI tensor (1, 1, H, W) in range [-1, 1]
            do_cycle: Whether to compute cycle reconstruction
            
        Returns:
            Tuple of (fake_ct_tensor, cycle_mri_tensor)
        """
        mri_tensor = mri_tensor.to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            # MRI → CT
            fake_ct = self.G_AB(mri_tensor)
            
            # CT → MRI (cycle)
            cycle_mri = self.G_BA(fake_ct) if do_cycle else None
        
        return fake_ct.cpu(), cycle_mri.cpu() if cycle_mri is not None else None


def _load_nifti_volume_with_metadata(nifti_path: Path) -> Tuple[np.ndarray, dict]:
    """Load NIfTI volume preserving medical metadata for proper saving."""
    try:
        img = nib.load(str(nifti_path))
        data = img.get_fdata().astype(np.float32)
        
        # Handle 4D volumes by taking first timepoint/channel
        if data.ndim == 4:
            data = data[..., 0]
        
        metadata = {
            "affine": img.affine.copy(),
            "header": img.header.copy(),
            "original_shape": data.shape,
            "zooms": img.header.get_zooms()[:3],  # spatial only
        }
        
        return data, metadata
        
    except Exception as e:
        raise RuntimeError(f"Failed to load NIfTI file {nifti_path}: {e}")


def _save_nifti_volume_preserving_metadata(volume: np.ndarray, metadata: dict, out_path: Path) -> Path:
    """Save volume preserving original NIfTI metadata and affine matrix."""
    try:
        # Use original affine matrix
        affine = metadata.get("affine", np.eye(4))
        
        # Create new NIfTI with preserved metadata
        nii = nib.Nifti1Image(volume.astype(np.float32), affine)
        
        # Preserve header information where possible
        if "header" in metadata:
            try:
                original_header = metadata["header"]
                new_header = nii.header
                new_header.set_zooms(original_header.get_zooms()[:3])
            except Exception:
                pass  # Use default header if copy fails
        
        # Ensure output directory exists
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save NIfTI
        nib.save(nii, str(out_path))
        return out_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to save NIfTI file {out_path}: {e}")


def _resize_slice_to_original(processed_slice: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    """Resize processed 256×256 slice back to original dimensions."""
    from PIL import Image as PILImage
    
    # Convert to PIL, resize back, convert to numpy
    pil_img = PILImage.fromarray((processed_slice * 255).astype(np.uint8), mode='L')
    pil_resized = pil_img.resize(original_shape[::-1], PILImage.BICUBIC)  # PIL uses (W, H)
    return np.asarray(pil_resized, dtype=np.float32) / 255.0


def extract_nifti_slice_as_png(nifti_path: Path, slice_idx: int, axis: int = 2) -> Image.Image:
    """
    Robust extraction of NIfTI slice as PNG with proper normalization.
    
    Args:
        nifti_path: Path to NIfTI file
        slice_idx: Index of slice to extract
        axis: Axis along which to extract (0=sagittal, 1=coronal, 2=axial)
    
    Returns:
        PIL Image (grayscale, properly normalized)
    """
    try:
        # Load NIfTI volume
        volume, _ = _load_nifti_volume_with_metadata(nifti_path)
        
        # Extract slice along specified axis
        if axis == 0:  # Sagittal
            slice_data = volume[slice_idx, :, :]
        elif axis == 1:  # Coronal  
            slice_data = volume[:, slice_idx, :]
        else:  # Axial (default)
            slice_data = volume[:, :, slice_idx]
        
        # Robust normalization
        slice_clean = np.nan_to_num(slice_data.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        
        # Percentile-based normalization (handles outliers better)
        p2, p98 = np.percentile(slice_clean, [2, 98])
        
        if p98 - p2 < 1e-6:
            # Handle constant slices
            normalized = np.zeros_like(slice_clean, dtype=np.uint8)
        else:
            # Normalize to [0, 255]
            normalized = np.clip((slice_clean - p2) / (p98 - p2), 0, 1)
            normalized = (normalized * 255).astype(np.uint8)
        
        return Image.fromarray(normalized, mode='L')
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract slice {slice_idx} from {nifti_path}: {e}")


@torch.no_grad()
def run_dual_mode_inference(
    input_obj,  # PIL.Image for 2D, or Path/str for NIfTI
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
    do_cycle: bool = True,
    progress_callback = None,  # Optional progress callback for UI
) -> dict:
    """
    Unified dual-mode CycleGAN inference for 2D slices and 3D volumes.
    
    Args:
        input_obj: PIL.Image for 2D or Path/str for NIfTI
        checkpoint_path: Path to trained checkpoint
        device: Device for inference
        do_cycle: Whether to compute cycle reconstruction
        progress_callback: Optional callback(current, total) for progress
        
    Returns:
        For 2D: {mode: "2D", input: tensor, fake_ct: tensor, cycle_mri: tensor}
        For 3D: {mode: "3D", fake_volume: ndarray, output_path: str, original_shape: tuple}
    """
    # Initialize cached model (loads checkpoint once)
    model = CachedCycleGANInference(checkpoint_path, device)
    
    # Detect input type
    from PIL import Image as PILImage
    is_2d = isinstance(input_obj, PILImage.Image)
    is_path = isinstance(input_obj, (str, Path))
    
    if is_2d:
        # 2D slice processing
        x = preprocess_image(input_obj, target_size=256)
        fake_ct, cycle_mri = model.infer_slice(x, do_cycle)
        
        return {
            "mode": "2D",
            "input": x,
            "fake_ct": fake_ct,
            "cycle_mri": cycle_mri if cycle_mri is not None else torch.zeros_like(x),
        }
    
    elif is_path:
        # 3D NIfTI processing
        volume, metadata = _load_nifti_volume_with_metadata(Path(input_obj))
        
        H, W, D = volume.shape
        original_slice_shape = (H, W)
        
        # Global normalization for medical consistency
        vol_min, vol_max = float(volume.min()), float(volume.max())
        vol_range = vol_max - vol_min
        
        if vol_range < 1e-8:
            # Handle constant volumes
            fake_volume = np.zeros_like(volume)
        else:
            # Process slices with global normalization
            fake_slices = []
            
            for d in range(D):
                # Update progress if callback provided
                if progress_callback:
                    progress_callback(d + 1, D)
                
                # Extract and normalize slice using global statistics
                slice_2d = volume[:, :, d]
                normalized_slice = (slice_2d - vol_min) / vol_range
                
                # Convert to PIL for preprocessing
                slice_uint8 = (normalized_slice * 255.0).astype(np.uint8)
                pil_slice = Image.fromarray(slice_uint8, mode='L')
                
                # Preprocess to 256×256 and run inference
                x = preprocess_image(pil_slice, target_size=256)
                fake_ct_tensor, _ = model.infer_slice(x, do_cycle=False)
                
                # Convert back to numpy and resize to original dimensions
                fake_ct_pil = postprocess_tensor(fake_ct_tensor)
                fake_ct_256 = np.asarray(fake_ct_pil, dtype=np.float32) / 255.0
                
                # Resize back to original slice dimensions
                fake_ct_original = _resize_slice_to_original(fake_ct_256, original_slice_shape)
                fake_slices.append(fake_ct_original)
            
            # Stack into volume with original dimensions
            fake_volume = np.stack(fake_slices, axis=2)  # (H, W, D)
        
        # Save with preserved metadata
        out_path = Path("synthetic_ct.nii.gz")
        saved_path = _save_nifti_volume_preserving_metadata(fake_volume, metadata, out_path)
        
        return {
            "mode": "3D",
            "fake_volume": fake_volume,
            "output_path": str(saved_path),
            "original_shape": metadata["original_shape"],
        }
    
    else:
        raise ValueError("Unsupported input type. Use PIL.Image for 2D or Path for NIfTI 3D.")
