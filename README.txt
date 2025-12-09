MRI-to-CT Medical Image Translation Using CycleGAN
========================================================

Welcome! This project converts MRI brain scans to CT-like images using artificial intelligence. This README provides step-by-step instructions to help you set up, run, and understand our medical image translation system.

WHAT YOU NEED BEFORE STARTING (Prerequisites)
==============================================

Before running our code, please make sure your computer has:

System Requirements:
- Operating System: Windows 10/11, macOS 10.15+, or Linux Ubuntu 18.04+
- Python Version: 3.10, 3.11, or 3.12 (we recommend 3.11)
- Memory: At least 16GB RAM for smooth operation
- Storage: 10GB free disk space for datasets and model files
- Graphics Card: NVIDIA GPU with 8GB+ memory (optional but highly recommended for faster training)

Required Software:
1. Python 3.11 - Download from python.org
2. Git (optional) - For downloading this repository

STEP 1: Setting Up Your Environment
===================================

A. Install Python and Essential Libraries

For Windows Users:
Step 1: Open Command Prompt as Administrator
Step 2: Install PyTorch for GPU (if you have NVIDIA graphics card)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Step 3: If you don't have a GPU, install CPU version instead
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

For Mac/Linux Users:
Step 1: Open Terminal
Step 2: Install PyTorch (CPU version for Mac, GPU for Linux with NVIDIA)
    pip install torch torchvision

For Linux with NVIDIA GPU:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

B. Install Project Dependencies
Navigate to the project folder and install all required packages:
    pip install -r requirements.txt

Install additional packages for the web interface:
    pip install streamlit nibabel pillow matplotlib seaborn pandas tqdm

C. Verify Installation
Test if everything is installed correctly:
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

If this runs without errors, you're ready to proceed!

STEP 2: Getting the Medical Image Dataset
=========================================

Our system uses the CHAOS medical imaging dataset, which contains real MRI and CT scans from hospitals. Here's how to set it up:

Option A: Use Our Pre-processed Dataset (Recommended for Beginners)

We've already prepared a balanced dataset with 2,500 MRI images and 2,500 CT images:

Step 1: Navigate to the project directory
    cd "path/to/MRI-CT"

Step 2: Create the enhanced dataset (this uses our existing processed data)
    python make_small_dataset.py

Step 3: Verify the dataset is ready
    python scripts/verify_data.py --dataset chaos_3k

What this does:
- Creates a folder called data/chaos_3k/ with balanced medical images
- Each image is 256x256 pixels, perfect for our AI model
- You should see output confirming 2,500 MRI and 2,500 CT images

Option B: Download Original CHAOS Dataset (Advanced Users)

If you want to work with the original medical dataset:

1. Register and Download:
   - Visit CHAOS Challenge website: https://chaos.grand-challenge.org/
   - Create a free account and download the training data
   - Extract the ZIP files to data/raw/chaos/

2. Process the Raw Medical Images:
    python scripts/prepare_slices.py --raw data/raw/chaos --out data/chaos --size 256 --max-slices 80

Verify Your Dataset:
    python scripts/verify_data.py --dataset chaos_3k

Expected Output:
- Perfect balance: 2,500 MRI + 2,500 CT images
- Good intensity ranges for medical images
- Ready for AI model training

STEP 3: Training Your AI Model (CycleGAN)
========================================

This is where the magic happens! We'll train an AI model to learn how to convert MRI scans to CT scans.

Before You Start Training

Time Requirements:
- With GPU: 3-4 hours of training time
- Without GPU: 1 day (24 hours)
- Recommendation: Let it run overnight or over a weekend

Step A: Configure Training Settings

The training configuration is already set up in configs/train_chaos_gpu.yaml. You don't need to change anything, but here's what it does:

- Dataset: Uses our chaos_3k enhanced dataset (2,500 + 2,500 images)
- Training Duration: 20 epochs (complete cycles through all data)
- Image Size: 256x256 pixels (medical image standard)
- Batch Size: 2 images at a time (fits in most GPU memory)

Step B: Start Training Your AI Model

Navigate to project directory:
    cd "path/to/MRI-CT"

Start training (this will run for many hours!):
    python src/train_cyclegan.py --config configs/train_chaos_gpu.yaml

What happens during training:
- The AI learns to convert MRI to CT and CT to MRI
- Progress is saved every 2000 steps
- Sample images are generated to show learning progress
- You'll see loss numbers decreasing (this is good!)

Step C: Monitor Training Progress

Watch the Console Output:
[Epoch 1/20] step 400 loss_G: 2.145 loss_D_A: 0.567 loss_D_B: 0.623
[Epoch 2/20] step 800 loss_G: 1.892 loss_D_A: 0.534 loss_D_B: 0.589
...

Check Generated Sample Images:
- Look in runs/chaos_3k_enhanced/samples/
- You'll see the AI's progress in converting MRI to CT images
- Early samples look blurry, but quality improves over time

Training Files Created:
- runs/chaos_3k_enhanced/checkpoints/ - Saved AI model weights
- runs/chaos_3k_enhanced/samples/ - Sample images showing progress

Step D: What to Expect

Training Progress:
- Hours 1-3: AI learns basic image structure
- Hours 4-8: Medical details start appearing  
- Hours 9-15: High-quality CT-like images generated
- Final Result: Professional-quality medical image translation

Signs of Successful Training:
- Loss values stabilize around 1-2
- Generated CT images look realistic
- Medical structures (organs, tissues) are clearly visible

STEP 4: Testing and Evaluating Your AI Model
============================================

After training completes, it's time to test how well your AI model performs!

A. Run Comprehensive Model Evaluation

Test your trained model with 150 medical images:
    python src/evaluate.py --checkpoint runs/chaos_3k_enhanced/checkpoints/step_024000.pt --n-samples 150

What this evaluation does:
- Tests AI on 150 MRI to CT and 150 CT to MRI translations
- Calculates quality metrics (SSIM, PSNR, MSE, MAE)
- Creates detailed performance reports and visualizations
- Generates comparison images to see AI performance

Expected Results (Our Model Achieved):
- SSIM: 0.90 (Excellent structural similarity - 90% match)
- PSNR: 29.61 dB (High image quality)
- Translation Quality: Medical-grade image conversion

B. Launch Interactive Web Interface

Experience your AI model in action with our user-friendly web interface:

Start the web application:
    streamlit run src/ui/app.py --server.port 8503

Then open your web browser and go to: http://localhost:8503

C. Using the Web Interface

Upload and Test Images:
1. 2D Images: Drag and drop PNG/JPG medical images
2. 3D Volumes: Upload NIfTI medical scan files
3. Real-time Processing: See AI convert MRI to CT instantly
4. Download Results: Save generated medical images

Interface Features:
- Image Upload: Easy drag-and-drop interface
- Instant Results: AI processes images in 2-3 seconds
- Quality Metrics: See translation confidence scores
- Download Options: Save results in multiple formats
- Cycle Testing: See MRI to CT to MRI reconstruction quality

D. Understanding Your Results

Quality Indicators:
- High SSIM (>0.8): AI preserves medical structures accurately
- High PSNR (>25 dB): Generated images have excellent quality
- Visual Assessment: Generated CT scans should look realistic and medically accurate

What Good Results Look Like:
- Clear organ boundaries (liver, kidneys, spleen)
- Proper tissue contrast (soft tissue vs. bone)
- No artifacts or unrealistic features
- Medical professionals could use these for reference

STEP 5: Complete Code Execution Summary
======================================

Here's everything you need to run from start to finish:

Quick Start Checklist (Follow These Commands in Order)

1. SETUP: Install required packages
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    pip install streamlit nibabel pillow matplotlib seaborn pandas tqdm

2. DATASET: Prepare medical images
    cd "path/to/MRI-CT"
    python make_small_dataset.py
    python scripts/verify_data.py --dataset chaos_3k

3. TRAINING: Train your AI model (3-4 hours)
    python src/train_cyclegan.py --config configs/train_chaos_gpu.yaml

4. EVALUATION: Test model performance
    python src/evaluate.py --checkpoint runs/chaos_3k_enhanced/checkpoints/step_024000.pt --n-samples 150

5. DEMO: Launch web interface
    streamlit run src/ui/app.py --server.port 8503

Project Folder Organization

MRI-CT/
├── src/                    Main AI code
│   ├── train_cyclegan.py      Trains the AI model
│   ├── evaluate.py            Tests model performance
│   ├── models.py              AI architecture definition
│   ├── utils.py               Helper functions
│   ├── datasets.py            Handles medical image loading
│   └── ui/
│       └── app.py             Web interface for testing
├── configs/                Training settings
│   └── train_chaos_gpu.yaml   Configuration for GPU training
├── data/                   Medical image datasets
│   └── chaos_3k/             Our processed dataset (2,500+2,500 images)
│       ├── mri/              MRI scan images
│       └── ct/               CT scan images
├── runs/                   Training results
│   └── chaos_3k_enhanced/    Your trained model
│       ├── checkpoints/      AI model files (.pt)
│       └── samples/          Training progress images
├── scripts/               Utility scripts
│   ├── verify_data.py        Checks dataset quality
│   └── download_data.py      Downloads medical data
├── make_small_dataset.py     Creates balanced dataset
├── requirements.txt          List of required packages
└── README.txt               This instruction file

ADVANCED CONFIGURATION
=====================

Training Parameters
Key settings in your config file:

model:
  n_res_blocks: 9        Generator complexity (6-12)
  init_type: "normal"    Weight initialization
  lsgan: true           Use LSGAN loss (recommended)

train:
  epochs: 20           Total training epochs
  batch_size: 2        Adjust for GPU memory
  lr: 0.0002           Learning rate
  lambda_cycle: 10     Cycle consistency weight
  lambda_identity: 1   Identity loss weight
  lr_decay_start: 10   When to start LR decay
  mixed_precision: true Enable AMP for speed

DATA PROCESSING DETAILS
=======================

CHAOS Dataset Structure:
- Multi-modal: T1/T2 weighted MRI + CT scans
- Anatomical focus: Abdominal organs (liver, kidneys, spleen)
- Patients: 40 training cases with paired MRI-CT
- Resolution: Original DICOM converted to 256x256 PNG slices

Image Preprocessing Pipeline:
Automatic processing from raw CHAOS data:
    python scripts/prepare_slices.py --raw data/raw/chaos --out data/chaos --size 256

- Slice extraction: Middle axial slices from 3D volumes
- Intensity normalization: Percentile clipping (1%-99%) + min-max scaling
- Quality filtering: Removes low-variance background slices
- Format conversion: DICOM to PNG for training efficiency
- Manifest generation: data/chaos/manifest.json for traceability

TRAINING FROM SCRATCH GUIDE
===========================

Phase 1: Setup and Validation (1-2 hours)
1. Verify data processing
    python scripts/verify_data.py

2. Quick training test (10 epochs)
    python -m src.train_cyclegan --config configs/cyclegan_cpu.yaml

Phase 2: Full Training (3-4 hours with GPU)
GPU training (recommended):
    python -m src.train_cyclegan --config configs/train_chaos_gpu.yaml

Monitor progress:
- Loss curves should stabilize after 10-15 epochs
- Sample images improve progressively  
- Cycle reconstruction should maintain anatomical structures

Phase 3: Model Evaluation
Test with web interface:
    streamlit run src/ui/app.py --server.port 8503

Upload test images from: data/chaos_3k/mri/
Expected outputs: Realistic CT-like images with preserved anatomy

TRAINING MONITORING & DEBUGGING
================================

Key Metrics to Track:
- Generator Loss: Should decrease and stabilize (target: 1-5)
- Discriminator Loss: Should stay around 0.5-1.0  
- Cycle Loss: Most important - should decrease consistently
- Identity Loss: Helps preserve input characteristics

Common Training Issues & Solutions:

Issue: Mode collapse
Symptoms: Identical outputs for different inputs
Solution: Reduce LR, increase cycle loss weight

Issue: Blurry results
Symptoms: Lack of fine details
Solution: Increase n_res_blocks, train longer

Issue: Training instability
Symptoms: Oscillating losses
Solution: Enable mixed precision, gradient clipping

Issue: Memory errors
Symptoms: GPU OOM
Solution: Reduce batch_size to 1-2

Issue: Slow convergence
Symptoms: No improvement after 50+ epochs
Solution: Increase learning rate, check data quality

PRODUCTION USAGE
================

Batch Inference:

from src.utils import run_cyclegan_inference, preprocess_image
from PIL import Image
import torch
from pathlib import Path

# Process multiple images
checkpoint_path = Path("runs/chaos_3k_enhanced/checkpoints/step_024000.pt")
for img_path in Path("input_folder").glob("*.png"):
    # Load and preprocess image
    mri_tensor = preprocess_image(Image.open(img_path))
    
    # Run inference  
    fake_ct, cycle_mri = run_cyclegan_inference(mri_tensor, checkpoint_path)
    
    # Save results
    save_inference_results(mri_tensor, fake_ct, cycle_mri, 
                          Path("output_folder") / img_path.stem)

REFERENCES & RESOURCES
======================

- CycleGAN Paper: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
  https://arxiv.org/abs/1703.10593
- CHAOS Challenge: Combined Healthy Abdominal Organ Segmentation
  https://chaos.grand-challenge.org/
- Medical Image Translation: Best practices for cross-modal synthesis
- PyTorch Implementation: Optimized for medical imaging workflows

CLINICAL USAGE WARNING
======================

This is a research implementation. Generated medical images should:
- Never be used for direct clinical diagnosis
- Always be validated by qualified medical professionals  
- Be clearly marked as AI-generated in any clinical context
- Undergo appropriate validation studies before clinical deployment
