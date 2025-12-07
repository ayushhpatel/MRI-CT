"""Comprehensive CycleGAN Evaluation Pipeline for MRI <-> CT translation.

This script evaluates cycle consistency and translation quality using comprehensive metrics.
Provides detailed visualizations and statistical analysis of model performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

from models import ResnetGenerator
from datasets import list_images

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


class EvaluationDataset(Dataset):
    """Dataset for evaluation - loads individual MRI and CT images."""
    
    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, path.stem


def load_generators(checkpoint_path: Path, device: torch.device, n_blocks: int = 9) -> Tuple[ResnetGenerator, ResnetGenerator]:
    """Load trained CycleGAN generators from checkpoint."""
    G_AB = ResnetGenerator(n_blocks=n_blocks)  # MRI -> CT
    G_BA = ResnetGenerator(n_blocks=n_blocks)  # CT -> MRI
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    G_AB.load_state_dict(checkpoint["G_AB"])
    G_BA.load_state_dict(checkpoint["G_BA"])
    
    G_AB.to(device).eval()
    G_BA.to(device).eval()
    
    return G_AB, G_BA


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for SSIM/PSNR computation."""
    # Denormalize from [-1, 1] to [0, 1]
    img = (tensor.cpu().squeeze() + 1.0) / 2.0
    return img.clamp(0, 1).numpy()


def compute_comprehensive_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive metrics between original and reconstructed images."""
    orig_np = tensor_to_numpy(original)
    recon_np = tensor_to_numpy(reconstructed)
    
    # SSIM with data range [0, 1]
    ssim_val = ssim(orig_np, recon_np, data_range=1.0)
    
    # PSNR with data range [0, 1]
    psnr_val = psnr(orig_np, recon_np, data_range=1.0)
    
    # MSE
    mse_val = mse(orig_np, recon_np)
    
    # MAE (Mean Absolute Error)
    mae_val = np.mean(np.abs(orig_np - recon_np))
    
    # RMSE (Root Mean Square Error)
    rmse_val = np.sqrt(mse_val)
    
    return {
        'ssim': float(ssim_val),
        'psnr': float(psnr_val),
        'mse': float(mse_val),
        'mae': float(mae_val),
        'rmse': float(rmse_val)
    }


def create_visualization_grid(
    images: List[torch.Tensor], 
    titles: List[str], 
    save_path: Path,
    ncols: int = 6
) -> None:
    """Create and save visualization grid."""
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row, col = i // ncols, i % ncols
        img_np = tensor_to_numpy(img)
        axes[row, col].imshow(img_np, cmap='gray')
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(images), nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Create comprehensive metrics visualizations."""
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Metrics Distribution Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CycleGAN Evaluation Metrics Distribution', fontsize=16, fontweight='bold')
    
    metrics = ['ssim', 'psnr', 'mse', 'mae', 'rmse']
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink']
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Plot distributions for both directions
        mri_data = df[df['direction'] == 'MRI->CT->MRI'][metric]
        ct_data = df[df['direction'] == 'CT->MRI->CT'][metric]
        
        ax.hist(mri_data, alpha=0.6, label='MRI Cycle', bins=15, color=colors[0], density=True)
        ax.hist(ct_data, alpha=0.6, label='CT Cycle', bins=15, color=colors[1], density=True)
        
        ax.set_title(f'{metric.upper()} Distribution', fontweight='bold')
        ax.set_xlabel(f'{metric.upper()} Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'metrics_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation Matrix
    metrics_cols = ['ssim', 'psnr', 'mse', 'mae', 'rmse']
    correlation_matrix = df[metrics_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_dir / 'metrics_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Box Plot Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Metrics Comparison: MRI Cycle vs CT Cycle', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        df.boxplot(column=metric, by='direction', ax=ax)
        ax.set_title(f'{metric.upper()} by Direction')
        ax.set_xlabel('Cycle Direction')
        ax.set_ylabel(f'{metric.upper()} Value')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.suptitle('Metrics Comparison: MRI Cycle vs CT Cycle', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_dir / 'metrics_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_comprehensive_results(df: pd.DataFrame, output_dir: Path, checkpoint_name: str) -> Dict:
    """Save comprehensive evaluation results and create summary report."""
    
    # Calculate statistics by direction
    mri_stats = df[df['direction'] == 'MRI->CT->MRI'].describe()
    ct_stats = df[df['direction'] == 'CT->MRI->CT'].describe()
    overall_stats = df.describe()
    
    # Create summary dictionary
    summary = {
        'checkpoint': checkpoint_name,
        'evaluation_date': datetime.now().isoformat(),
        'total_samples': len(df),
        'mri_cycle_samples': len(df[df['direction'] == 'MRI->CT->MRI']),
        'ct_cycle_samples': len(df[df['direction'] == 'CT->MRI->CT']),
        'statistics': {
            'mri_cycle': mri_stats.to_dict(),
            'ct_cycle': ct_stats.to_dict(),
            'overall': overall_stats.to_dict()
        }
    }
    
    # Calculate key metrics
    mri_cycle = df[df['direction'] == 'MRI->CT->MRI']
    ct_cycle = df[df['direction'] == 'CT->MRI->CT']
    
    key_metrics = {
        'overall': {
            'mean_ssim': df['ssim'].mean(),
            'std_ssim': df['ssim'].std(),
            'mean_psnr': df['psnr'].mean(),
            'std_psnr': df['psnr'].std(),
        },
        'mri_cycle': {
            'mean_ssim': mri_cycle['ssim'].mean(),
            'std_ssim': mri_cycle['ssim'].std(),
            'mean_psnr': mri_cycle['psnr'].mean(),
            'std_psnr': mri_cycle['psnr'].std(),
        },
        'ct_cycle': {
            'mean_ssim': ct_cycle['ssim'].mean(),
            'std_ssim': ct_cycle['ssim'].std(),
            'mean_psnr': ct_cycle['psnr'].mean(),
            'std_psnr': ct_cycle['psnr'].std(),
        }
    }
    
    summary['key_metrics'] = key_metrics
    
    # Save JSON results
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create detailed text report
    report_lines = [
        "FINAL CYCLEGAN EVALUATION REPORT",
        "=" * 50,
        f"Checkpoint: {checkpoint_name}",
        f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Samples: {len(df)}",
        "",
        "📊 OVERALL PERFORMANCE",
        "-" * 30,
        f"Mean SSIM: {key_metrics['overall']['mean_ssim']:.4f} ± {key_metrics['overall']['std_ssim']:.4f}",
        f"Mean PSNR: {key_metrics['overall']['mean_psnr']:.2f} ± {key_metrics['overall']['std_psnr']:.2f} dB",
        "",
        "🔄 MRI CYCLE (MRI → CT → MRI)",
        "-" * 30,
        f"Mean SSIM: {key_metrics['mri_cycle']['mean_ssim']:.4f} ± {key_metrics['mri_cycle']['std_ssim']:.4f}",
        f"Mean PSNR: {key_metrics['mri_cycle']['mean_psnr']:.2f} ± {key_metrics['mri_cycle']['std_psnr']:.2f} dB",
        f"Mean MSE: {mri_cycle['mse'].mean():.6f} ± {mri_cycle['mse'].std():.6f}",
        f"Mean MAE: {mri_cycle['mae'].mean():.6f} ± {mri_cycle['mae'].std():.6f}",
        "",
        "🔄 CT CYCLE (CT → MRI → CT)",
        "-" * 30,
        f"Mean SSIM: {key_metrics['ct_cycle']['mean_ssim']:.4f} ± {key_metrics['ct_cycle']['std_ssim']:.4f}",
        f"Mean PSNR: {key_metrics['ct_cycle']['mean_psnr']:.2f} ± {key_metrics['ct_cycle']['std_psnr']:.2f} dB",
        f"Mean MSE: {ct_cycle['mse'].mean():.6f} ± {ct_cycle['mse'].std():.6f}",
        f"Mean MAE: {ct_cycle['mae'].mean():.6f} ± {ct_cycle['mae'].std():.6f}",
        "",
        "🎯 PERFORMANCE ANALYSIS",
        "-" * 30,
    ]
    
    # Add performance analysis
    overall_ssim = key_metrics['overall']['mean_ssim']
    overall_psnr = key_metrics['overall']['mean_psnr']
    
    if overall_ssim > 0.8:
        report_lines.append("✅ Excellent structural similarity (SSIM > 0.8)")
    elif overall_ssim > 0.6:
        report_lines.append("🟡 Good structural similarity (SSIM > 0.6)")
    else:
        report_lines.append("❌ Poor structural similarity (SSIM < 0.6)")
    
    if overall_psnr > 25:
        report_lines.append("✅ High image quality (PSNR > 25 dB)")
    elif overall_psnr > 20:
        report_lines.append("🟡 Moderate image quality (PSNR > 20 dB)")
    else:
        report_lines.append("❌ Low image quality (PSNR < 20 dB)")
    
    # Check cycle consistency balance
    mri_ssim = key_metrics['mri_cycle']['mean_ssim']
    ct_ssim = key_metrics['ct_cycle']['mean_ssim']
    ssim_diff = abs(mri_ssim - ct_ssim)
    
    if ssim_diff < 0.05:
        report_lines.append("✅ Balanced cycle consistency between directions")
    else:
        better_direction = "MRI cycle" if mri_ssim > ct_ssim else "CT cycle"
        report_lines.append(f"⚠️  Unbalanced performance - {better_direction} performs better")
    
    report_text = "\n".join(report_lines)
    
    with open(output_dir / 'evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return summary


def evaluate_cycle_consistency(
    checkpoint_path: Path,
    mri_dir: Path,
    ct_dir: Path,
    output_dir: Path,
    n_samples: int = 100,
    n_blocks: int = 9,
    device: torch.device = None
) -> None:
    """Enhanced evaluation function with comprehensive metrics and visualizations."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 Starting Comprehensive CycleGAN Evaluation")
    print("=" * 60)
    print(f"📂 Checkpoint: {checkpoint_path}")
    print(f"🖥️  Device: {device}")
    
    # Load generators
    G_AB, G_BA = load_generators(checkpoint_path, device, n_blocks)
    
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mri_paths = list_images(mri_dir)
    ct_paths = list_images(ct_dir)
    
    print(f"📁 Found {len(mri_paths)} MRI images, {len(ct_paths)} CT images")
    
    if len(mri_paths) < n_samples or len(ct_paths) < n_samples:
        print(f"⚠️  Requested {n_samples} samples but only have {len(mri_paths)} MRI and {len(ct_paths)} CT images")
        n_samples = min(n_samples, len(mri_paths), len(ct_paths))
    
    # Randomly sample images
    random.seed(42)  # For reproducibility
    selected_mri = random.sample(mri_paths, n_samples)
    selected_ct = random.sample(ct_paths, n_samples)
    
    mri_dataset = EvaluationDataset(selected_mri, transform)
    ct_dataset = EvaluationDataset(selected_ct, transform)
    
    mri_loader = DataLoader(mri_dataset, batch_size=1, shuffle=False)
    ct_loader = DataLoader(ct_dataset, batch_size=1, shuffle=False)
    
    # Results storage
    results = []
    vis_images = []
    vis_titles = []
    sample_images = []
    
    print(f"\\n🔄 Evaluating cycle consistency on {n_samples} samples per direction...")
    
    # Evaluate MRI -> CT -> MRI cycle
    print("Processing MRI → fake CT → reconstructed MRI...")
    
    with torch.no_grad():
        for i, (mri_img, mri_name) in enumerate(mri_loader):
            if i >= n_samples:
                break
                
            mri_img = mri_img.to(device)
            
            # Forward cycle: MRI -> fake CT -> reconstructed MRI
            fake_ct = G_AB(mri_img)
            recon_mri = G_BA(fake_ct)
            
            # Compute comprehensive metrics
            metrics = compute_comprehensive_metrics(mri_img, recon_mri)
            
            result = {
                'sample_id': f'mri_{i:03d}',
                'filename': mri_name[0],
                'direction': 'MRI->CT->MRI',
                **metrics
            }
            results.append(result)
            
            # Store sample images for first 10 samples
            if i < 10:
                sample_images.append({
                    'real_mri': tensor_to_numpy(mri_img[0]),
                    'fake_ct': tensor_to_numpy(fake_ct[0]),
                    'recon_mri': tensor_to_numpy(recon_mri[0]),
                    'sample_id': i,
                    'type': 'mri_cycle'
                })
            
            # Collect images for grid visualization (first 6 samples)
            if i < 6:
                vis_images.extend([
                    mri_img[0], fake_ct[0], recon_mri[0]
                ])
                vis_titles.extend([
                    f'Real MRI {i+1}', f'Fake CT {i+1}', f'Recon MRI {i+1}'
                ])
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{n_samples} MRI samples...")
    
    # Evaluate CT -> MRI -> CT cycle
    print("Processing CT → fake MRI → reconstructed CT...")
    
    with torch.no_grad():
        for i, (ct_img, ct_name) in enumerate(ct_loader):
            if i >= n_samples:
                break
                
            ct_img = ct_img.to(device)
            
            # Reverse cycle: CT -> fake MRI -> reconstructed CT
            fake_mri = G_BA(ct_img)
            recon_ct = G_AB(fake_mri)
            
            # Compute comprehensive metrics
            metrics = compute_comprehensive_metrics(ct_img, recon_ct)
            
            result = {
                'sample_id': f'ct_{i:03d}',
                'filename': ct_name[0],
                'direction': 'CT->MRI->CT',
                **metrics
            }
            results.append(result)
            
            # Store sample images for first 10 samples
            if i < 10:
                sample_images.append({
                    'real_ct': tensor_to_numpy(ct_img[0]),
                    'fake_mri': tensor_to_numpy(fake_mri[0]),
                    'recon_ct': tensor_to_numpy(recon_ct[0]),
                    'sample_id': i,
                    'type': 'ct_cycle'
                })
            
            # Collect images for grid visualization (first 6 samples)
            if i < 6:
                vis_images.extend([
                    ct_img[0], fake_mri[0], recon_ct[0]
                ])
                vis_titles.extend([
                    f'Real CT {i+1}', f'Fake MRI {i+1}', f'Recon CT {i+1}'
                ])
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{n_samples} CT samples...")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed CSV
    csv_path = output_dir / "detailed_evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\\n💾 Saved detailed results to {csv_path}")
    
    # Create sample images visualization
    if sample_images:
        create_sample_visualization(sample_images, output_dir)
    
    # Create main visualization grid
    vis_path = output_dir / "cycle_consistency_samples.png"
    create_visualization_grid(vis_images, vis_titles, vis_path, ncols=9)
    print(f"💾 Saved sample visualization to {vis_path}")
    
    # Create comprehensive metrics visualizations
    create_metrics_visualizations(df, output_dir)
    print(f"💾 Created comprehensive visualizations in {output_dir}/visualizations/")
    
    # Save comprehensive results and generate report
    checkpoint_name = checkpoint_path.name
    summary = save_comprehensive_results(df, output_dir, checkpoint_name)
    print(f"💾 Saved comprehensive report to {output_dir}/evaluation_report.txt")
    
    # Print enhanced summary
    print_enhanced_summary(df, checkpoint_name)


def create_sample_visualization(sample_images: List[Dict], output_dir: Path) -> None:
    """Create detailed sample visualization showing cycles."""
    fig, axes = plt.subplots(4, 10, figsize=(25, 10))
    fig.suptitle('CycleGAN Sample Results - First 10 Samples', fontsize=16, fontweight='bold')
    
    # MRI cycle samples (first 10)
    mri_samples = [s for s in sample_images if s['type'] == 'mri_cycle'][:10]
    ct_samples = [s for s in sample_images if s['type'] == 'ct_cycle'][:10]
    
    for i in range(10):
        # MRI Cycle Row 1-2
        if i < len(mri_samples):
            sample = mri_samples[i]
            axes[0, i].imshow(sample['real_mri'], cmap='gray')
            axes[0, i].set_title(f'Real MRI {i+1}', fontsize=8)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(sample['recon_mri'], cmap='gray')
            axes[1, i].set_title(f'Recon MRI {i+1}', fontsize=8)
            axes[1, i].axis('off')
        
        # CT Cycle Row 3-4
        if i < len(ct_samples):
            sample = ct_samples[i]
            axes[2, i].imshow(sample['real_ct'], cmap='gray')
            axes[2, i].set_title(f'Real CT {i+1}', fontsize=8)
            axes[2, i].axis('off')
            
            axes[3, i].imshow(sample['recon_ct'], cmap='gray')
            axes[3, i].set_title(f'Recon CT {i+1}', fontsize=8)
            axes[3, i].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, 'MRI\\nOriginal', rotation=90, fontsize=12, fontweight='bold', ha='center', va='center')
    fig.text(0.02, 0.6, 'MRI\\nReconstructed', rotation=90, fontsize=12, fontweight='bold', ha='center', va='center')
    fig.text(0.02, 0.4, 'CT\\nOriginal', rotation=90, fontsize=12, fontweight='bold', ha='center', va='center')
    fig.text(0.02, 0.25, 'CT\\nReconstructed', rotation=90, fontsize=12, fontweight='bold', ha='center', va='center')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05)
    plt.savefig(output_dir / 'detailed_sample_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_enhanced_summary(df: pd.DataFrame, checkpoint_name: str) -> None:
    """Print enhanced summary statistics."""
    mri_cycle = df[df['direction'] == 'MRI->CT->MRI']
    ct_cycle = df[df['direction'] == 'CT->MRI->CT']
    
    print(f"\\n{'='*80}")
    print("🏆 FINAL CYCLEGAN EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"📂 Checkpoint: {checkpoint_name}")
    print(f"📊 Total Samples: {len(df)} ({len(mri_cycle)} MRI cycle + {len(ct_cycle)} CT cycle)")
    
    print(f"\\n🔄 OVERALL PERFORMANCE")
    print(f"{'-'*40}")
    print(f"Mean SSIM: {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")
    print(f"Mean PSNR: {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f} dB")
    print(f"Mean MSE:  {df['mse'].mean():.6f} ± {df['mse'].std():.6f}")
    print(f"Mean MAE:  {df['mae'].mean():.6f} ± {df['mae'].std():.6f}")
    
    print(f"\\n🧠 MRI CYCLE (MRI → CT → MRI)")
    print(f"{'-'*40}")
    print(f"Mean SSIM: {mri_cycle['ssim'].mean():.4f} ± {mri_cycle['ssim'].std():.4f}")
    print(f"Mean PSNR: {mri_cycle['psnr'].mean():.2f} ± {mri_cycle['psnr'].std():.2f} dB")
    print(f"Mean MSE:  {mri_cycle['mse'].mean():.6f} ± {mri_cycle['mse'].std():.6f}")
    print(f"Mean MAE:  {mri_cycle['mae'].mean():.6f} ± {mri_cycle['mae'].std():.6f}")
    
    print(f"\\n🏥 CT CYCLE (CT → MRI → CT)")
    print(f"{'-'*40}")
    print(f"Mean SSIM: {ct_cycle['ssim'].mean():.4f} ± {ct_cycle['ssim'].std():.4f}")
    print(f"Mean PSNR: {ct_cycle['psnr'].mean():.2f} ± {ct_cycle['psnr'].std():.2f} dB")
    print(f"Mean MSE:  {ct_cycle['mse'].mean():.6f} ± {ct_cycle['mse'].std():.6f}")
    print(f"Mean MAE:  {ct_cycle['mae'].mean():.6f} ± {ct_cycle['mae'].std():.6f}")
    
    # Performance indicators
    overall_ssim = df['ssim'].mean()
    overall_psnr = df['psnr'].mean()
    
    print(f"\\n🎯 PERFORMANCE INDICATORS")
    print(f"{'-'*40}")
    
    if overall_ssim > 0.8:
        print("✅ Excellent structural similarity (SSIM > 0.8)")
    elif overall_ssim > 0.6:
        print("🟡 Good structural similarity (SSIM > 0.6)")
    else:
        print("❌ Poor structural similarity (SSIM < 0.6)")
    
    if overall_psnr > 25:
        print("✅ High image quality (PSNR > 25 dB)")
    elif overall_psnr > 20:
        print("🟡 Moderate image quality (PSNR > 20 dB)")
    else:
        print("❌ Low image quality (PSNR < 20 dB)")
    
    # Balance check
    ssim_diff = abs(mri_cycle['ssim'].mean() - ct_cycle['ssim'].mean())
    if ssim_diff < 0.05:
        print("✅ Balanced performance between directions")
    else:
        better = "MRI→CT→MRI" if mri_cycle['ssim'].mean() > ct_cycle['ssim'].mean() else "CT→MRI→CT"
        print(f"⚠️  Imbalanced performance - {better} performs better")
    
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CycleGAN Evaluation Pipeline")
    parser.add_argument("--checkpoint", type=Path, 
                       default=Path("../runs/hd_fast/checkpoints/step_008000.pt"),
                       help="Path to trained CycleGAN checkpoint")
    parser.add_argument("--mri-dir", type=Path, default=Path("../data/chaos/mri_slices"),
                       help="Directory containing MRI images")
    parser.add_argument("--ct-dir", type=Path, default=Path("../data/chaos/ct_slices"), 
                       help="Directory containing CT images")
    parser.add_argument("--output-dir", type=Path, 
                       default=Path(f"../final_hd_fast_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                       help="Directory to save evaluation results")
    parser.add_argument("--n-samples", type=int, default=100,
                       help="Number of samples to evaluate per direction")
    parser.add_argument("--n-blocks", type=int, default=9,
                       help="Number of ResNet blocks in generator")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu). Auto-detect if not specified")
    
    args = parser.parse_args()
    
    # Find best checkpoint if default doesn't exist
    if not args.checkpoint.exists():
        checkpoint_dir = Path("../runs/hd_fast/checkpoints")
        if checkpoint_dir.exists():
            checkpoints = sorted(list(checkpoint_dir.glob("*.pt")))
            if checkpoints:
                args.checkpoint = checkpoints[-1]  # Use latest checkpoint
                print(f"⚠️  Default checkpoint not found, using: {args.checkpoint}")
            else:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    if not args.mri_dir.exists():
        raise FileNotFoundError(f"MRI directory not found: {args.mri_dir}")
        
    if not args.ct_dir.exists():
        raise FileNotFoundError(f"CT directory not found: {args.ct_dir}")
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    print("🚀 Starting Final CycleGAN Evaluation on HD_FAST Results")
    print("=" * 70)
    print(f"📂 Checkpoint: {args.checkpoint}")
    print(f"📊 Samples per direction: {args.n_samples}")
    print(f"📁 Output directory: {args.output_dir}")
    
    evaluate_cycle_consistency(
        checkpoint_path=args.checkpoint,
        mri_dir=args.mri_dir,
        ct_dir=args.ct_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        n_blocks=args.n_blocks,
        device=device
    )
    
    print(f"\\n✅ Evaluation Complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print("🎉 Check the evaluation_report.txt for detailed analysis!")


if __name__ == "__main__":
    main()