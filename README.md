# MRI-to-CT Cross-Modality Synthesis with CycleGAN

This repository implements a complete CycleGAN pipeline for synthesizing CT-like images from MRI scans. It includes both training from scratch and a web-based inference interface using the CHAOS dataset.

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10–3.12 (PyTorch wheels for 3.13 may lag)
- 16GB+ RAM recommended for training
- NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Installation
```bash
# Install PyTorch (choose CPU or GPU version)
# For GPU (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# For web UI (optional):
pip install streamlit nibabel
```

## 📊 Dataset Setup (CHAOS Challenge)

### Option 1: Full CHAOS Dataset (Recommended)
1. **Download CHAOS dataset**: Visit [CHAOS Challenge](https://chaos.grand-challenge.org/) and download the training data
2. **Extract to**: `data/raw/chaos/`
3. **Process slices**:
```bash
python scripts/prepare_slices.py --raw data/raw/chaos --out data/chaos --size 256 --max-slices 80
```

### Option 2: Small Test Dataset
```bash
# Download sample volumes (~230 MB):
python scripts/download_data.py

# Process into slices:
python scripts/prepare_slices.py --raw data/raw --out data/processed --size 256 --max-slices 80
```

### Data Verification
```bash
# Verify processed data stats:
python scripts/verify_data.py
```

## 🎯 Training CycleGAN from Scratch

### Step 1: Configure Training
Edit `configs/train_chaos_gpu.yaml` (or create your own config):
```yaml
data:
  mri_dir: "data/chaos/mri_slices"
  ct_dir: "data/chaos/ct_slices"
  resolution: 256
  batch_size: 4  # Adjust based on GPU memory
  
model:
  n_res_blocks: 9  # Higher for better quality
  
train:
  epochs: 200
  lr: 0.0002
  save_every_steps: 400
  mixed_precision: true  # Enable for faster training
  force_cpu: false  # Set to true for CPU-only training
```

### Step 2: Start Training
```bash
# GPU training (recommended):
python -m src.train_cyclegan --config configs/train_chaos_gpu.yaml

# CPU training (slower):
python -m src.train_cyclegan --config configs/cyclegan_cpu.yaml
```

### Step 3: Monitor Progress
- **Checkpoints**: Saved to `runs/your_experiment_name/checkpoints/`
- **Sample images**: Saved to `runs/your_experiment_name/samples/`
- **Training logs**: Printed to console with loss metrics

### Expected Training Time
- **GPU (RTX 3080)**: ~6-8 hours for 200 epochs
- **CPU**: ~24-48 hours for 200 epochs
- **Memory usage**: 4-8GB GPU VRAM, 8-16GB system RAM

## 🖥️ Web Interface for Inference

Once you have a trained model, launch the interactive web UI:

```bash
# Start Streamlit UI
python run_ui.py

# Or run directly:
streamlit run src/ui/app.py
```

**Features:**
- Upload MRI images (PNG, JPG, NIfTI formats)
- Real-time CycleGAN inference
- Cycle consistency visualization  
- Download generated results
- GPU/CPU automatic detection

**Access at**: http://localhost:8501

## 📁 Project Structure
```
MRI-CT/
├── src/
│   ├── models.py          # CycleGAN architecture
│   ├── train_cyclegan.py  # Training pipeline
│   ├── utils.py           # Utilities and inference
│   ├── datasets.py        # Data loading
│   └── ui/
│       └── app.py         # Streamlit web interface
├── configs/
│   ├── train_chaos_gpu.yaml    # GPU training config
│   └── cyclegan_cpu.yaml       # CPU training config
├── data/
│   ├── chaos/             # Processed CHAOS dataset
│   ├── raw/               # Raw downloaded data
│   └── processed/         # Processed sample data
├── runs/                  # Training outputs
│   └── hd_fast/
│       ├── checkpoints/   # Model weights (.pt files)
│       └── samples/       # Training samples
└── inference_results/     # UI inference outputs
```

## 🔧 Advanced Configuration

### Training Parameters
Key settings in your config file:
```yaml
model:
  n_res_blocks: 9        # Generator complexity (6-12)
  init_type: "normal"    # Weight initialization
  lsgan: true           # Use LSGAN loss (recommended)

train:
  epochs: 200           # Total training epochs
  batch_size: 4         # Adjust for GPU memory
  lr: 0.0002           # Learning rate
  lambda_cycle: 10.0    # Cycle consistency weight
  lambda_identity: 0.5  # Identity loss weight
  lr_decay_start: 100   # When to start LR decay
  mixed_precision: true # Enable AMP for speed
```

## 📊 Data Processing Details

### CHAOS Dataset Structure
- **Multi-modal**: T1/T2 weighted MRI + CT scans
- **Anatomical focus**: Abdominal organs (liver, kidneys, spleen)
- **Patients**: 40 training cases with paired MRI-CT
- **Resolution**: Original DICOM converted to 256×256 PNG slices

### Image Preprocessing Pipeline
```bash
# Automatic processing from raw CHAOS data:
python scripts/prepare_slices.py --raw data/raw/chaos --out data/chaos --size 256
```
- **Slice extraction**: Middle axial slices from 3D volumes
- **Intensity normalization**: Percentile clipping (1%-99%) + min-max scaling
- **Quality filtering**: Removes low-variance background slices
- **Format conversion**: DICOM → PNG for training efficiency
- **Manifest generation**: `data/chaos/manifest.json` for traceability

## 🎯 Training from Scratch Guide

### Phase 1: Setup and Validation (1-2 hours)
```bash
# 1. Verify data processing
python scripts/verify_data.py

# 2. Quick training test (10 epochs)
python -m src.train_cyclegan --config configs/cyclegan_cpu.yaml
```

### Phase 2: Full Training (6-48 hours)
```bash
# GPU training (recommended):
python -m src.train_cyclegan --config configs/train_chaos_gpu.yaml
```

**Monitor progress**:
- Loss curves should stabilize after ~50 epochs
- Sample images improve progressively  
- Cycle reconstruction should maintain anatomical structures

### Phase 3: Model Evaluation
```bash
# Test with web interface:
python run_ui.py

# Upload test images from: data/chaos/mri_slices/
# Expected outputs: Realistic CT-like images with preserved anatomy
```

## 🔍 Training Monitoring & Debugging

### Key Metrics to Track
- **Generator Loss**: Should decrease and stabilize (target: 1-5)
- **Discriminator Loss**: Should stay around 0.5-1.0  
- **Cycle Loss**: Most important - should decrease consistently
- **Identity Loss**: Helps preserve input characteristics

### Common Training Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Mode collapse | Identical outputs for different inputs | Reduce LR, increase cycle loss weight |
| Blurry results | Lack of fine details | Increase n_res_blocks, train longer |
| Training instability | Oscillating losses | Enable mixed precision, gradient clipping |
| Memory errors | GPU OOM | Reduce batch_size to 1-2 |
| Slow convergence | No improvement after 50+ epochs | Increase learning rate, check data quality |

## 🚀 Production Usage

### Batch Inference
```python
from src.utils import run_cyclegan_inference, preprocess_image
from PIL import Image
import torch
from pathlib import Path

# Process multiple images
checkpoint_path = Path("runs/hd_fast/checkpoints/step_008000.pt")
for img_path in Path("input_folder").glob("*.png"):
    # Load and preprocess image
    mri_tensor = preprocess_image(Image.open(img_path))
    
    # Run inference  
    fake_ct, cycle_mri = run_cyclegan_inference(mri_tensor, checkpoint_path)
    
    # Save results
    save_inference_results(mri_tensor, fake_ct, cycle_mri, 
                          Path("output_folder") / img_path.stem)
```

## 📚 References & Resources

- **CycleGAN Paper**: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- **CHAOS Challenge**: [Combined Healthy Abdominal Organ Segmentation](https://chaos.grand-challenge.org/)
- **Medical Image Translation**: Best practices for cross-modal synthesis
- **PyTorch Implementation**: Optimized for medical imaging workflows

## ⚠️ Clinical Usage Warning

**This is a research implementation. Generated medical images should:**
- Never be used for direct clinical diagnosis
- Always be validated by qualified medical professionals  
- Be clearly marked as AI-generated in any clinical context
- Undergo appropriate validation studies before clinical deployment
