# MRI→CT CycleGAN Web Interface

A Streamlit-based web application for running MRI to CT synthesis using a trained CycleGAN model.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install Streamlit (if not already installed)
pip install streamlit

# Install additional dependencies for NIfTI support
pip install nibabel
```

### 2. Launch the UI
```bash
# Option 1: Use the launcher script
python run_ui.py

# Option 2: Run directly with Streamlit
streamlit run src/ui/app.py

# Option 3: Specify custom port
streamlit run src/ui/app.py --server.port 8502
```

### 3. Open in Browser
The application will automatically open at `http://localhost:8501`

## 📋 Features

### ✅ Supported Input Formats
- **Images**: PNG, JPG, JPEG
- **Medical**: NIfTI (.nii, .nii.gz) - automatically extracts middle slice

### ✅ Core Functionality  
- **File Upload**: Drag & drop or browse for MRI images
- **Preprocessing**: Automatic resize to 256×256 and normalization
- **CycleGAN Inference**: 
  - MRI → Synthetic CT generation
  - CT → MRI cycle-back reconstruction
- **Real-time Results**: Live visualization of all outputs
- **Download Options**: Save all results as PNG files

### ✅ Technical Features
- **GPU Acceleration**: Automatic CUDA detection and usage
- **Progress Tracking**: Real-time inference progress
- **Error Handling**: Comprehensive error reporting
- **Model Info**: Displays checkpoint and architecture details

## 🏗️ Architecture

The UI is built using:
- **Backend**: PyTorch CycleGAN implementation
- **Frontend**: Streamlit web framework  
- **Model**: ResNet Generator (9 blocks) + PatchGAN Discriminator
- **Checkpoint**: `runs/hd_fast/checkpoints/step_008000.pt`

## 📁 File Structure
```
src/ui/
├── app.py              # Main Streamlit application
├── __init__.py         # Package initialization
└── ...

src/
├── models.py           # CycleGAN model definitions
├── utils.py            # Inference and preprocessing utilities
├── train_cyclegan.py   # Training implementation
└── ...

runs/hd_fast/checkpoints/
└── step_008000.pt      # Pre-trained model weights

inference_results/      # Generated outputs saved here
├── input_mri.png
├── fake_ct.png
└── cycle_mri.png
```

## 🖼️ Usage Workflow

1. **Upload**: Select an MRI image (PNG/JPG/NIfTI)
2. **Preview**: View the uploaded image and file information  
3. **Generate**: Click "Generate CT Scan" button
4. **Results**: See 3-column layout:
   - Input MRI (left)
   - Generated CT (center) 
   - Reconstructed MRI (right)
5. **Download**: Save individual results as PNG files

## ⚙️ Configuration Options

### System Settings
- **Device Selection**: Auto-detects GPU, option to force CPU
- **Model Path**: Configurable checkpoint location
- **Output Directory**: Customizable results folder

### Advanced Options  
- **Force CPU**: Override GPU detection for debugging
- **Custom Resolution**: Modify target image size (default: 256×256)
- **Batch Processing**: Process multiple images (future feature)

## 🔧 Technical Requirements

### Software Dependencies
```
streamlit>=1.28.0
torch>=1.12.0
torchvision>=0.13.0
Pillow>=9.0.0
numpy>=1.21.0
nibabel>=4.0.0
```

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only inference
- **Recommended**: 16GB RAM, NVIDIA GPU with 4GB+ VRAM
- **Storage**: ~2GB for model checkpoints and cache

## 🐛 Troubleshooting

### Common Issues

**"Checkpoint not found" error:**
```bash
# Verify checkpoint exists
ls runs/hd_fast/checkpoints/step_008000.pt
```

**Import errors:**
```bash
# Check Python path and dependencies
python -c "from src.models import ResnetGenerator; print('OK')"
```

**GPU not detected:**
```bash
# Verify PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Streamlit port conflicts:**
```bash
# Use custom port
streamlit run src/ui/app.py --server.port 8502
```

### Performance Tips
- **GPU Inference**: ~2-5 seconds per image
- **CPU Inference**: ~15-30 seconds per image  
- **Memory Usage**: ~2-4GB during inference
- **Batch Size**: Single image processing for UI responsiveness

## 🎯 Model Information

**Training Details:**
- **Dataset**: CHAOS Challenge (MRI-CT pairs)
- **Architecture**: CycleGAN with ResNet generators
- **Training Steps**: 8,000 iterations
- **Image Resolution**: 256×256 grayscale
- **Normalization**: [-1, 1] range

**Performance Metrics:**
- **Cycle Consistency**: Maintains anatomical structure
- **Visual Quality**: High-fidelity CT synthesis
- **Inference Speed**: Real-time on modern GPUs

## 📝 Development Notes

### Code Organization
- `src/ui/app.py`: Main Streamlit interface
- `src/utils.py`: Inference pipeline and preprocessing
- `src/models.py`: CycleGAN model definitions

### Key Functions
- `run_cyclegan_inference()`: Core inference pipeline
- `preprocess_image()`: Input preprocessing
- `tensor_to_pil()`: Output postprocessing
- `load_nifti_slice()`: NIfTI file handling

### Future Enhancements
- [ ] Batch processing support
- [ ] Interactive parameter tuning
- [ ] Model comparison interface
- [ ] DICOM format support
- [ ] 3D volume visualization
- [ ] Quantitative metrics display

---

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly  
3. Ensure the checkpoint file exists and is accessible
4. Test with sample images first

**Happy generating! 🧠→🦴**