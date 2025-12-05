# MRI-to-CT Cross-Modality Synthesis (CPU-friendly)

This repo contains a lightweight CycleGAN pipeline to synthesize CT-like slices from MRI slices with a CPU-only setup and a small, publicly available sample dataset (sourced from the MONAI extra test data release).

## Quickstart
- Create/activate a Python 3.10–3.12 environment (PyTorch wheels for 3.13 may lag).
- Install deps (CPU-only PyTorch on Windows):  
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`  
  `pip install -r requirements.txt`
- Download sample volumes (~230 MB total):  
  `python scripts/download_data.py`
- Slice volumes to PNGs (defaults: 256×256, up to 80 slices/volume):  
  `python scripts/prepare_slices.py --raw data/raw --out data/processed --size 256 --max-slices 80`
- Verify processed data stats:  
  `python scripts/verify_data.py`
- Train a small CycleGAN run (CPU, 2 epochs):  
  `python -m src.train_cyclegan --config configs/cyclegan_cpu.yaml`

Outputs land in `runs/cyclegan_cpu/` (checkpoints + sample grids).

## Data notes
- Downloaded assets (MONAI extra test data 0.8.1):
  - MRI: `testing_ixi_t1.tar.gz`, `Prostate_T2W_AX_1.nii`
  - CT: `spleen_test.nii.gz`, `copd1_highres_INSP_STD_COPD_img.nii.gz`
- `scripts/prepare_slices.py`:
  - Heuristically classifies volumes by filename keywords.
  - Percentile clipping (1/99), min–max to 8-bit, optional low-variance filtering, resize to square.
  - Writes `data/processed/manifest.json` for traceability and data verification.

## Configuration
- See `configs/cyclegan_cpu.yaml` for batch size, learning rate, and logging cadence. Increase `n_res_blocks` or epochs if you later use a GPU.
- Training is forced to CPU by default (`force_cpu: true`). Set to `false` to enable CUDA if available.

## Validation/Monitoring
- `scripts/verify_data.py` prints slice counts and basic intensity stats to ensure data isn’t degenerate before training.
- Sample grids saved during training show (MRI → fake CT → reconstructed MRI) and (CT → fake MRI → reconstructed CT) for quick qualitative checks.

## Next steps (optional)
- Add a Pix2Pix trainer once a paired MRI–CT dataset is available.
- Swap in a higher-fidelity dataset or increase image size/epochs if you add GPU resources.
