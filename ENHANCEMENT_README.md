# CycleGAN MRI-CT Translation - Enhanced Version

## Task A: Training Upgrades

### Key Improvements Made:

1. **Architecture Enhancements:**
   - **9 ResNet blocks** (increased from 6) for better feature representation
   - **PatchGAN discriminator** with 70×70 receptive field
   - **Instance normalization** and reflect padding throughout
   - **No dropout** for stable high-quality generation

2. **Training Configuration:**
   - **256×256 image resolution** with smart data augmentation
   - **Mixed precision (AMP)** for faster training and lower memory usage
   - **Linear learning rate decay** starting at epoch 10
   - **Optimized for 30-minute training** on RTX 4070 (20 epochs, batch_size=2)

3. **Data Augmentation:**
   - Random crops from slightly larger images
   - Horizontal and vertical flips
   - Proper [-1, 1] normalization

4. **Training Stability:**
   - **Reduced identity loss weight** (1.0 vs 5.0) to focus on cycle consistency
   - **Adam optimizer** with β1=0.5, β2=0.999
   - **Replay buffer** for discriminator stability

### Why These Improvements Matter:

- **9 ResNet blocks**: More capacity for learning complex MRI↔CT mappings
- **Mixed precision**: ~40% speedup with minimal quality loss
- **Linear decay**: Prevents overfitting in later epochs
- **Better augmentation**: Improves generalization without artifacts
- **256×256 resolution**: Good balance between quality and training time

## Task B: Evaluation Pipeline

### Features:

1. **Cycle Consistency Metrics:**
   - SSIM (Structural Similarity Index)
   - PSNR (Peak Signal-to-Noise Ratio)
   - Evaluated on both MRI→CT→MRI and CT→MRI→CT cycles

2. **Visualization:**
   - Side-by-side comparison grids
   - Original → Fake → Reconstructed sequences
   - Saved as high-resolution PNG files

3. **Detailed Analysis:**
   - CSV output with per-sample metrics
   - Statistical summaries (mean ± std)
   - Configurable sample sizes

## Usage Instructions

### Training with Enhanced Config:
```bash
# Train with enhanced configuration
python -m src.train_cyclegan --config configs/train_chaos_gpu.yaml
```

### Evaluation:
```bash
# Basic evaluation on 50 random samples
python -m src.evaluate --checkpoint runs/chaos_enhanced/checkpoints/epoch_020.pt

# Custom evaluation with specific parameters
python -m src.evaluate \
    --checkpoint runs/chaos_enhanced/checkpoints/epoch_020.pt \
    --mri-dir data/chaos/mri_slices \
    --ct-dir data/chaos/ct_slices \
    --output-dir evaluation_results \
    --n-samples 100 \
    --n-blocks 9
```

### Output Files:
- `evaluation_results/cycle_consistency_results.csv` - Detailed per-sample metrics
- `evaluation_results/cycle_consistency_visualization.png` - Visual comparison grid

## Expected Performance:

With the enhanced configuration on RTX 4070:
- **Training time**: ~25-30 minutes (20 epochs)
- **Memory usage**: ~6-8GB VRAM (batch_size=2, AMP enabled)
- **Cycle consistency SSIM**: Expected >0.85 for good models
- **Cycle consistency PSNR**: Expected >25 dB for good models

## Configuration Details:

The enhanced config (`configs/train_chaos_gpu.yaml`) includes:
- 9 ResNet blocks for generators
- PatchGAN discriminators (70×70 patches)
- Mixed precision training (AMP)
- Linear learning rate decay
- Optimized batch size and learning schedule
- Enhanced data augmentation pipeline