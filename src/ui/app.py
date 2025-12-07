"""
Streamlit UI for MRI→CT CycleGAN Inference

This application provides a web interface for running MRI to CT synthesis
using a trained CycleGAN model. Users can upload MRI images and generate
synthetic CT scans with cycle-consistency visualization.

Features:
- File upload (PNG, JPG, JPEG, NIfTI formats)
- Real-time inference with trained CycleGAN model
- Cycle consistency visualization (MRI → CT → MRI)
- Results download and export
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import torch
from PIL import Image
import numpy as np
import io
from typing import Optional, Tuple
import time

# Project imports
from src.utils import (
    preprocess_image, 
    load_nifti_slice, 
    tensor_to_pil, 
    run_cyclegan_inference,
    save_inference_results,
    get_device
)


# Configuration
CHECKPOINT_PATH = project_root / "runs" / "hd_fast" / "checkpoints" / "step_008000.pt"
RESULTS_DIR = project_root / "inference_results"
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg']
SUPPORTED_NIFTI_FORMATS = ['.nii', '.nii.gz']
TARGET_SIZE = 256


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="MRI→CT CycleGAN Inference",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 MRI→CT CycleGAN Inference")
    st.markdown("""
    **Generate synthetic CT scans from MRI images using trained CycleGAN**
    
    Upload an MRI image to generate a corresponding synthetic CT scan. 
    The model also shows cycle consistency by reconstructing the original MRI.
    """)


def check_requirements() -> bool:
    """Check if all requirements are met for inference."""
    issues = []
    
    # Check checkpoint availability
    if not CHECKPOINT_PATH.exists():
        issues.append(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
    
    # Check device availability
    device = get_device()
    if device.type == 'cuda':
        st.sidebar.success(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    else:
        st.sidebar.info("ℹ️ Using CPU (slower inference)")
    
    if issues:
        st.error("**Setup Issues:**")
        for issue in issues:
            st.error(issue)
        return False
    
    st.sidebar.success("✅ All requirements met")
    return True


def process_uploaded_file(uploaded_file) -> Optional[Tuple[Image.Image, str, Optional[int]]]:
    """
    Process uploaded file and return PIL Image, file info, and slice index (if NIfTI).
    
    Returns:
        Tuple of (PIL_Image, file_info_string, slice_index_if_nifti)
    """
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension in SUPPORTED_IMAGE_FORMATS:
            # Handle regular image files
            image = Image.open(uploaded_file)
            file_info = f"**File:** {uploaded_file.name} ({image.size[0]}×{image.size[1]})"
            return image, file_info, None
            
        elif any(uploaded_file.name.lower().endswith(ext) for ext in SUPPORTED_NIFTI_FORMATS):
            # Handle NIfTI files
            # Save uploaded file temporarily
            temp_path = RESULTS_DIR / "temp_nifti.nii.gz"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Extract middle slice
            slice_array, slice_idx = load_nifti_slice(temp_path)
            image = Image.fromarray(slice_array, mode='L')
            
            # Cleanup temp file
            temp_path.unlink()
            
            file_info = f"**File:** {uploaded_file.name} (NIfTI slice {slice_idx})"
            return image, file_info, slice_idx
            
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None


def display_inference_results(
    input_mri: torch.Tensor, 
    fake_ct: torch.Tensor, 
    cycle_mri: torch.Tensor
):
    """Display inference results in a 3-column layout."""
    
    # Convert tensors to PIL Images for display
    input_pil = tensor_to_pil(input_mri)
    fake_ct_pil = tensor_to_pil(fake_ct)
    cycle_mri_pil = tensor_to_pil(cycle_mri)
    
    st.markdown("### 🔬 Inference Results")
    
    # Three-column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📱 Input MRI**")
        st.image(input_pil, caption="Original MRI", use_column_width=True)
        
    with col2:
        st.markdown("**🦴 Generated CT**")
        st.image(fake_ct_pil, caption="Synthetic CT", use_column_width=True)
        
    with col3:
        st.markdown("**🔄 Reconstructed MRI**")
        st.image(cycle_mri_pil, caption="Cycle-back MRI", use_column_width=True)
    
    return input_pil, fake_ct_pil, cycle_mri_pil


def create_download_section(input_pil: Image.Image, fake_ct_pil: Image.Image, cycle_mri_pil: Image.Image):
    """Create download links for all generated images."""
    
    st.markdown("### 💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    # Helper function to convert PIL to bytes
    def pil_to_bytes(pil_image: Image.Image, format: str = 'PNG') -> bytes:
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return buffer.getvalue()
    
    with col1:
        input_bytes = pil_to_bytes(input_pil)
        st.download_button(
            label="📱 Download Input MRI",
            data=input_bytes,
            file_name="input_mri.png",
            mime="image/png"
        )
    
    with col2:
        fake_ct_bytes = pil_to_bytes(fake_ct_pil)
        st.download_button(
            label="🦴 Download Synthetic CT", 
            data=fake_ct_bytes,
            file_name="fake_ct.png",
            mime="image/png"
        )
    
    with col3:
        cycle_mri_bytes = pil_to_bytes(cycle_mri_pil)
        st.download_button(
            label="🔄 Download Reconstructed MRI",
            data=cycle_mri_bytes,
            file_name="cycle_mri.png", 
            mime="image/png"
        )


def run_inference_pipeline(image: Image.Image) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run the complete inference pipeline.
    
    Returns:
        Tuple of (input_tensor, fake_ct_tensor, cycle_mri_tensor) or None if error
    """
    try:
        # Preprocess image
        with st.spinner("🔄 Preprocessing image..."):
            input_tensor = preprocess_image(image, TARGET_SIZE)
        
        # Run inference
        with st.spinner("🧠 Running CycleGAN inference..."):
            start_time = time.time()
            fake_ct, cycle_mri = run_cyclegan_inference(
                input_tensor, 
                CHECKPOINT_PATH,
                device=get_device()
            )
            inference_time = time.time() - start_time
        
        st.success(f"✅ Inference completed in {inference_time:.2f} seconds")
        
        # Save results to disk
        with st.spinner("💾 Saving results..."):
            save_inference_results(input_tensor, fake_ct, cycle_mri, RESULTS_DIR)
            st.info(f"Results saved to: `{RESULTS_DIR}`")
        
        return input_tensor, fake_ct, cycle_mri
        
    except Exception as e:
        st.error(f"❌ Inference failed: {str(e)}")
        st.exception(e)  # Show full traceback in debug mode
        return None


def main():
    """Main application logic."""
    setup_page()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System status
        st.subheader("System Status")
        if not check_requirements():
            st.stop()
        
        # Model info
        st.subheader("Model Information")
        st.info(f"""
        **Checkpoint:** step_008000.pt  
        **Architecture:** ResNet Generator (9 blocks)  
        **Input/Output:** 256×256 grayscale  
        **Training:** CycleGAN on CHAOS dataset
        """)
        
        # Advanced options
        st.subheader("Advanced Options")
        
        force_cpu = st.checkbox("Force CPU inference", value=False)
        if force_cpu:
            st.warning("⚠️ CPU inference will be significantly slower")
    
    # Main interface
    st.markdown("## 📁 Section A: Upload MRI Image")
    
    uploaded_file = st.file_uploader(
        "Choose an MRI image file",
        type=['png', 'jpg', 'jpeg', 'nii', 'gz'],
        help="Supported formats: PNG, JPG, JPEG, NIfTI (.nii, .nii.gz)"
    )
    
    if uploaded_file is not None:
        # Process uploaded file
        result = process_uploaded_file(uploaded_file)
        if result is None:
            st.stop()
            
        image, file_info, slice_idx = result
        
        # Display file info and preview
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(file_info)
            if slice_idx is not None:
                st.info(f"Using middle slice: {slice_idx}")
            
        with col2:
            st.image(image, caption="Uploaded MRI Preview", width=300)
        
        # Section B: Run inference
        st.markdown("## 🚀 Section B: Generate Synthetic CT")
        
        if st.button("🧠 Generate CT Scan", type="primary", use_container_width=True):
            
            # Run inference pipeline
            results = run_inference_pipeline(image)
            
            if results is not None:
                input_tensor, fake_ct, cycle_mri = results
                
                # Section C: Display results
                st.markdown("## 📊 Section C: Results Visualization")
                pil_images = display_inference_results(input_tensor, fake_ct, cycle_mri)
                
                # Section D: Download options
                st.markdown("## 📥 Section D: Download Results")
                create_download_section(*pil_images)
                
                # Additional metrics/info
                st.markdown("### 📈 Additional Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Input Resolution", f"{TARGET_SIZE}×{TARGET_SIZE}")
                
                with col2:
                    device = get_device(force_cpu)
                    st.metric("Inference Device", device.type.upper())
                
                with col3:
                    st.metric("Model Type", "CycleGAN")
    
    else:
        # Show example/instructions when no file uploaded
        st.info("""
        👆 **Upload an MRI image to get started!**
        
        **Supported formats:**
        - Standard images: PNG, JPG, JPEG
        - Medical images: NIfTI (.nii, .nii.gz)
        
        **What happens next:**
        1. Your MRI will be preprocessed to 256×256 resolution
        2. The CycleGAN model generates a synthetic CT scan
        3. Cycle consistency shows reconstruction quality
        4. Download all results as PNG files
        """)
        
        # Show model architecture info
        with st.expander("🏗️ Model Architecture Details"):
            st.markdown("""
            **CycleGAN Architecture:**
            - **Generator:** ResNet-based with 9 residual blocks
            - **Discriminator:** PatchGAN (70×70 receptive field)
            - **Training:** Unpaired MRI↔CT translation
            - **Losses:** Adversarial + Cycle consistency + Identity
            
            **Key Features:**
            - No paired training data required
            - Preserves anatomical structure
            - Bidirectional translation capability
            - Instance normalization for stable training
            """)


if __name__ == "__main__":
    main()