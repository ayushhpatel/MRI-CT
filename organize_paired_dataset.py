"""
Script to organize the paired dataset for CycleGAN training.
This script will copy/organize the PNG images from the downloads folder
to the expected data structure for training.
"""

import os
import shutil
from pathlib import Path

def organize_paired_dataset():
    # Source paths
    source_base = Path("A:/FMI Assignments/MRI-CT/downloads/Paired MRI (T1, T2) and CT Scans Dataset")
    ct_source = source_base / "CT/PNG"
    mri_source = source_base / "T1-MRI/PNG"  # Using T1-MRI for this training
    
    # Target paths
    target_base = Path("A:/FMI Assignments/MRI-CT/data/paired_dataset")
    ct_target = target_base / "ct"
    mri_target = target_base / "mri"
    
    # Create target directories
    ct_target.mkdir(parents=True, exist_ok=True)
    mri_target.mkdir(parents=True, exist_ok=True)
    
    print("Organizing paired dataset...")
    print(f"Source CT: {ct_source}")
    print(f"Source MRI: {mri_source}")
    print(f"Target CT: {ct_target}")
    print(f"Target MRI: {mri_target}")
    
    # Copy CT images
    ct_count = 0
    for patient_dir in ct_source.glob("Patient_*"):
        print(f"Processing CT images from {patient_dir.name}...")
        for ct_file in patient_dir.glob("*.png"):
            # Create a unique filename: patient_slice.png
            new_name = f"{patient_dir.name}_{ct_file.name}"
            target_path = ct_target / new_name
            shutil.copy2(ct_file, target_path)
            ct_count += 1
    
    # Copy MRI images
    mri_count = 0
    for patient_dir in mri_source.glob("Patient_*"):
        print(f"Processing MRI images from {patient_dir.name}...")
        for mri_file in patient_dir.glob("*.png"):
            # Create a unique filename: patient_slice.png
            new_name = f"{patient_dir.name}_{mri_file.name}"
            target_path = mri_target / new_name
            shutil.copy2(mri_file, target_path)
            mri_count += 1
    
    print(f"\nDataset organization complete!")
    print(f"CT images copied: {ct_count}")
    print(f"MRI images copied: {mri_count}")
    print(f"Target directory: {target_base}")

if __name__ == "__main__":
    organize_paired_dataset()