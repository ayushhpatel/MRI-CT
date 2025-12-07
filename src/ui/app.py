import sys
from pathlib import Path
import io
import tempfile
import glob

import streamlit as st
import torch
from PIL import Image
import numpy as np
import nibabel as nib

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import everything from utils.py - no preprocessing duplicates
from src.utils import (
    preprocess_image,
    postprocess_tensor, 
    run_dual_mode_inference,
    extract_nifti_slice_as_png
)


# Configuration
CHECKPOINT_PATH = PROJECT_ROOT / "runs" / "chaos_3k_enhanced" / "checkpoints" / "step_024000.pt"
SUPPORTED_IMAGE_EXTS = [".png", ".jpg", ".jpeg"]
SUPPORTED_NIFTI_EXTS = [".nii", ".nii.gz"]


def _bytes_from_pil(pil_img: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL image to bytes for download."""
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()


def _resolve_checkpoint(default_path: Path) -> Path:
    """Resolve checkpoint path, fallback to latest if default missing."""
    if default_path.exists():
        return default_path
    
    # Try to find latest checkpoint in directory
    try:
        ckpt_dir = default_path.parent
        candidates = sorted(glob.glob(str(ckpt_dir / "*.pt")))
        if candidates:
            latest_path = Path(candidates[-1])
            st.warning(f"Default checkpoint missing. Using latest: {latest_path.name}")
            return latest_path
    except Exception as e:
        st.error(f"Checkpoint discovery failed: {e}")
    
    st.error(f"No checkpoints found in {default_path.parent}")
    return default_path  # Return original for error handling


def _create_volume_previews(volume: np.ndarray) -> tuple:
    """Create axial, coronal, sagittal preview slices from 3D volume."""
    H, W, D = volume.shape
    z_mid, y_mid, x_mid = D // 2, H // 2, W // 2
    
    # Robust normalization to handle different NIfTI data ranges
    def normalize_slice_robust(slice_data):
        # Convert to float and handle NaN/inf values
        slice_clean = np.nan_to_num(slice_data.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get percentile-based normalization (more robust than min/max)
        p2, p98 = np.percentile(slice_clean, [2, 98])
        
        if p98 - p2 < 1e-6:
            # Handle constant slices
            return np.zeros_like(slice_clean, dtype=np.uint8)
        
        # Clip and normalize to [0, 255]
        normalized = np.clip((slice_clean - p2) / (p98 - p2), 0, 1)
        return (normalized * 255).astype(np.uint8)
    
    # Extract and normalize middle slices
    axial_slice = normalize_slice_robust(volume[:, :, z_mid])
    coronal_slice = normalize_slice_robust(volume[:, y_mid, :])
    sagittal_slice = normalize_slice_robust(volume[x_mid, :, :])
    
    # Convert to PIL images
    axial_img = Image.fromarray(axial_slice, mode="L")
    coronal_img = Image.fromarray(coronal_slice, mode="L")
    sagittal_img = Image.fromarray(sagittal_slice, mode="L")
    
    return axial_img, coronal_img, sagittal_img, (z_mid, y_mid, x_mid)


def _setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="MRI→CT CycleGAN Inference", 
        page_icon="🧠", 
        layout="wide"
    )
    st.title("🧠 MRI→CT CycleGAN Inference (2D + 3D)")
    st.markdown("""
    **Unified dual-mode inference for medical image translation**
    
    📷 **2D Mode**: Upload PNG/JPG → Generate single CT slice  
    🧠 **3D Mode**: Upload NIfTI → Generate full CT volume (slice-by-slice)
    """)


def _render_sidebar(checkpoint_path: Path):
    """Render sidebar with system information."""
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Checkpoint status
        st.subheader("Checkpoint")
        resolved_path = _resolve_checkpoint(checkpoint_path)
        st.write(f"📁 {resolved_path.name}")
        exists = resolved_path.exists()
        if exists:
            st.success("✅ Checkpoint loaded")
        else:
            st.error("❌ Checkpoint missing")
            st.stop()
        
        # Device information
        st.subheader("Device")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cuda":
            st.success(f"🚀 GPU: {torch.cuda.get_device_name()}")
        else:
            st.info("💻 CPU (slower)")
        
        # Options
        st.subheader("Options")
        do_cycle = st.checkbox("Enable cycle reconstruction (2D only)", value=True)
        
        return resolved_path, do_cycle


def _handle_2d_inference(pil_img: Image.Image, checkpoint_path: Path, do_cycle: bool):
    """Handle 2D slice inference workflow."""
    st.image(pil_img, caption=f"Input MRI ({pil_img.size[0]}×{pil_img.size[1]})", width=320)
    
    if st.button("🚀 Run 2D Inference", type="primary", use_container_width=True):
        try:
            with st.spinner("Running MRI→CT inference..."):
                result = run_dual_mode_inference(pil_img, checkpoint_path, do_cycle=do_cycle)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(pil_img, caption="Input MRI", width=200)
            
            with col2:
                fake_ct_img = postprocess_tensor(result["fake_ct"])
                st.image(fake_ct_img, caption="Synthetic CT", width=200)
            
            with col3:
                if result.get("cycle_mri") is not None:
                    cycle_img = postprocess_tensor(result["cycle_mri"])
                    st.image(cycle_img, caption="Cycle MRI", width=200)
                else:
                    st.write("No cycle reconstruction")
            
            # Download synthetic CT
            st.download_button(
                label="📥 Download Synthetic CT (PNG)",
                data=_bytes_from_pil(fake_ct_img),
                file_name="synthetic_ct.png",
                mime="image/png",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"2D Inference failed: {str(e)}")


def _handle_3d_inference(tmp_path: Path, checkpoint_path: Path):
    """Handle 3D NIfTI volume inference workflow."""
    try:
        # Load and preview NIfTI
        img = nib.load(str(tmp_path))
        volume = img.get_fdata().astype(np.float32)
        
        # Handle 4D by taking first volume
        if volume.ndim == 4:
            volume = volume[..., 0]
            st.info("4D volume detected. Using first timepoint.")
        
        shape = volume.shape
        st.write(f"📊 **Volume shape**: {shape[0]} × {shape[1]} × {shape[2]}")
        
        # Create previews
        axial_img, coronal_img, sagittal_img, (z_mid, y_mid, x_mid) = _create_volume_previews(volume)
        
        # Display previews
        st.markdown("### 🔍 Volume Previews")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(axial_img, caption=f"Axial (slice {z_mid})", width=200)
        with col2:
            st.image(coronal_img, caption=f"Coronal (slice {y_mid})", width=200)
        with col3:
            st.image(sagittal_img, caption=f"Sagittal (slice {x_mid})", width=200)
        
        # NIfTI to PNG conversion
        st.markdown("### 📤 Export Single Slice")
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("Export Middle Axial Slice", key="export_middle_axial_btn"):
                st.image(axial_img, caption="Exported Axial PNG", width=250)
                st.download_button(
                    label="📥 Download Axial PNG",
                    data=_bytes_from_pil(axial_img),
                    file_name="axial_middle.png",
                    mime="image/png",
                    key="download_middle_axial_btn"
                )
        
        with col_export2:
            slice_idx = st.number_input(
                "Select slice index", 
                min_value=0, 
                max_value=shape[2]-1, 
                value=z_mid, 
                step=1,
                key="export_slice_idx_input"
            )
            if st.button("Export Selected Slice", key="export_selected_slice_btn"):
                try:
                    # Use robust slice extraction
                    selected_img = extract_nifti_slice_as_png(tmp_path, int(slice_idx), axis=2)
                    st.image(selected_img, caption=f"Slice {slice_idx}", width=250)
                    st.download_button(
                        label=f"📥 Download Slice {slice_idx}",
                        data=_bytes_from_pil(selected_img),
                        file_name=f"axial_{slice_idx:03d}.png",
                        mime="image/png",
                        key="download_selected_slice_btn"
                    )
                    
                    # Store selected slice for potential sync with CT explorer
                    st.session_state.selected_export_slice = int(slice_idx)
                    
                except Exception as e:
                    st.error(f"Failed to extract slice {slice_idx}: {e}")
        
        # 3D volume inference
        st.markdown("### 🧠 3D Volume Inference")
        
        # Initialize session state for inference results
        if "inference_result" not in st.session_state:
            st.session_state.inference_result = None
        if "original_previews" not in st.session_state:
            st.session_state.original_previews = None
        
        # Inference controls
        col_inference1, col_inference2 = st.columns([2, 1])
        
        with col_inference1:
            run_inference = st.button("🚀 Run 3D Volume Inference", type="primary", use_container_width=True)
        
        with col_inference2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.inference_result = None
                st.session_state.original_previews = None
                st.rerun()
        
        if run_inference:
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing slice {current}/{total} ({progress:.1%})")
                
                with st.spinner("Running slice-by-slice MRI→CT inference..."):
                    result = run_dual_mode_inference(
                        tmp_path, 
                        checkpoint_path, 
                        do_cycle=False,
                        progress_callback=progress_callback
                    )
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Store results in session state
                st.session_state.inference_result = result
                st.session_state.original_previews = (axial_img, coronal_img, sagittal_img, (z_mid, y_mid, x_mid))
                
                st.success("✅ 3D inference completed!")
                
            except Exception as e:
                st.error(f"3D Inference failed: {str(e)}")
        
        # Display results if available (persists across interactions)
        if st.session_state.inference_result is not None:
            result = st.session_state.inference_result
            axial_img, coronal_img, sagittal_img, (z_mid, y_mid, x_mid) = st.session_state.original_previews
            
            st.write(f"📊 Output shape: {result['original_shape']}")
            st.write(f"💾 Saved to: {result['output_path']}")
            
            # Display synthetic CT results with slice navigation
            st.markdown("### 🔬 Synthetic CT Results")
            
            # Load the generated synthetic CT volume for visualization
            try:
                synthetic_ct_path = Path(result['output_path'])
                
                # Create preview slices from synthetic CT
                synthetic_volume = result['fake_volume']
                syn_axial, syn_coronal, syn_sagittal, (syn_z, syn_y, syn_x) = _create_volume_previews(synthetic_volume)
                
                # Show comparison: Original MRI vs Synthetic CT
                st.markdown("#### 📊 Volume Comparison (Middle Slices)")
                col_orig, col_synth = st.columns(2)
                
                with col_orig:
                    st.markdown("**Original MRI**")
                    st.image(axial_img, caption=f"MRI Axial (slice {z_mid})", width=250)
                
                with col_synth:
                    st.markdown("**Synthetic CT**")
                    st.image(syn_axial, caption=f"CT Axial (slice {syn_z})", width=250)
                
                # Interactive slice explorer for synthetic CT
                st.markdown("#### 🎛️ Synthetic CT Slice Explorer")
                explore_col1, explore_col2 = st.columns([1, 2])
                
                with explore_col1:
                    # Slice selection controls (with session state keys for persistence)
                    
                    # Check if there's a selected export slice to sync with
                    if hasattr(st.session_state, 'selected_export_slice'):
                        export_slice = st.session_state.selected_export_slice
                        st.info(f"💡 Last exported MRI slice: {export_slice}")
                        if st.button("🔗 Jump CT Explorer to Same Slice", help=f"Set CT explorer to slice {export_slice}", key="sync_to_export_btn"):
                            # Force update the slider by setting its session state value
                            st.session_state.ct_slice_slider = min(export_slice, synthetic_volume.shape[2] - 1)
                            st.rerun()
                    
                    ct_slice_idx = st.slider(
                        "Select CT slice to view",
                        min_value=0,
                        max_value=synthetic_volume.shape[2] - 1,
                        value=syn_z,
                        help="Navigate through synthetic CT volume",
                        key="ct_slice_slider"
                    )
                    
                    # View orientation (with session state key)
                    ct_orientation = st.selectbox(
                        "View orientation",
                        ["Axial", "Coronal", "Sagittal"],
                        index=0,
                        key="ct_orientation_select"
                    )
                    
                    # Export current slice
                    if st.button("📤 Export Current Slice", use_container_width=True, key="export_slice_btn"):
                        try:
                            # Extract slice based on orientation
                            if ct_orientation == "Axial":
                                axis = 2
                            elif ct_orientation == "Coronal":
                                axis = 1
                            else:  # Sagittal
                                axis = 0
                            
                            # Use robust extraction from synthetic CT
                            current_slice_img = extract_nifti_slice_as_png(
                                synthetic_ct_path, ct_slice_idx, axis
                            )
                            
                            st.download_button(
                                label=f"📥 Download {ct_orientation} Slice {ct_slice_idx}",
                                data=_bytes_from_pil(current_slice_img),
                                file_name=f"synthetic_ct_{ct_orientation.lower()}_{ct_slice_idx:03d}.png",
                                mime="image/png",
                                use_container_width=True,
                                key="download_slice_btn"
                            )
                        except Exception as e:
                            st.error(f"Failed to export slice: {e}")
                
                with explore_col2:
                    # Display selected slice
                    try:
                        if ct_orientation == "Axial":
                            axis = 2
                        elif ct_orientation == "Coronal":
                            axis = 1
                        else:  # Sagittal
                            axis = 0
                        
                        # Extract and display current slice
                        current_slice_img = extract_nifti_slice_as_png(
                            synthetic_ct_path, ct_slice_idx, axis
                        )
                        
                        st.image(
                            current_slice_img,
                            caption=f"Synthetic CT - {ct_orientation} View (Slice {ct_slice_idx})",
                            width=400
                        )
                        
                        # Show slice statistics
                        slice_data = np.array(current_slice_img)
                        st.write(f"📈 Intensity range: {slice_data.min()}-{slice_data.max()}")
                        st.write(f"📐 Dimensions: {slice_data.shape[0]}×{slice_data.shape[1]}")
                        
                    except Exception as e:
                        st.error(f"Failed to display slice {ct_slice_idx}: {e}")
            
            except Exception as e:
                st.error(f"Failed to load synthetic CT for visualization: {e}")
            
            # Download synthetic CT volume
            with open(result['output_path'], 'rb') as f:
                st.download_button(
                    label="📥 Download Complete Synthetic CT Volume (NIfTI)",
                    data=f.read(),
                    file_name="synthetic_ct_volume.nii.gz",
                    mime="application/octet-stream",
                    use_container_width=True,
                    key="download_volume_btn"
                )
                
    except Exception as e:
        st.error(f"Failed to process NIfTI file: {str(e)}")


def main():
    """Main application."""
    _setup_page()
    
    # Sidebar
    checkpoint_path, do_cycle = _render_sidebar(CHECKPOINT_PATH)
    
    # File uploader
    st.markdown("## 📁 Upload Medical Image")
    uploaded = st.file_uploader(
        "Choose MRI file",
        type=["png", "jpg", "jpeg", "nii", "gz"],
        help="PNG/JPG for 2D slices, NIfTI (.nii/.nii.gz) for 3D volumes"
    )
    
    if uploaded is not None:
        file_ext = Path(uploaded.name).suffix.lower()
        
        if file_ext in SUPPORTED_IMAGE_EXTS:
            # 2D workflow
            st.markdown("## 📷 2D Slice Processing")
            try:
                pil_img = Image.open(uploaded).convert("L")
                _handle_2d_inference(pil_img, checkpoint_path, do_cycle)
            except Exception as e:
                st.error(f"Failed to process 2D image: {str(e)}")
        
        elif uploaded.name.lower().endswith((".nii", ".nii.gz")):
            # 3D workflow
            st.markdown("## 🧠 3D Volume Processing")
            try:
                # Save uploaded NIfTI to temp file
                tmp_path = Path(tempfile.gettempdir()) / f"upload_{uploaded.name.replace(' ', '_')}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.read())
                
                _handle_3d_inference(tmp_path, checkpoint_path)
                
            except Exception as e:
                st.error(f"Failed to process 3D volume: {str(e)}")
        
        else:
            st.error("❌ Unsupported file format. Use PNG/JPG for 2D or NIfTI for 3D.")
    
    else:
        # Instructions
        st.info("""
        👆 **Upload a medical image to get started**
        
        **Supported formats:**
        - 🖼️ **2D images**: PNG, JPG, JPEG (single MRI slices)
        - 🧠 **3D volumes**: NIfTI (.nii, .nii.gz) (full MRI volumes)
        
        **What this app does:**
        - Converts MRI images to synthetic CT images using CycleGAN
        - Preserves medical image metadata and spatial information
        - Provides download options for results
        """)


if __name__ == "__main__":
    main()