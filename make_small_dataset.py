import random
import shutil
from pathlib import Path

def copy_subset(src_dir, dst_dir, count):
    """Randomly selects 'count' images from src_dir and copies them to dst_dir."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    images = list(src.glob("*.png"))

    if len(images) < count:
        raise ValueError(f"Not enough images in {src}. Found {len(images)}, need {count}")

    random.shuffle(images)
    selected = images[:count]

    for img in selected:
        shutil.copy(str(img), str(dst / img.name))

    print(f"[OK] Copied {len(selected)} → {dst}")


if __name__ == "__main__":
    # Source full dataset folders
    MRI_SRC = "data/chaos/mri_slices"
    CT_SRC  = "data/chaos/ct_slices"

    # New reduced dataset folders
    MRI_OUT = "data/chaos_small/mri"
    CT_OUT  = "data/chaos_small/ct"

    # samples per class
    N = 1500  # change to 1200, 2000, etc. if needed

    copy_subset(MRI_SRC, MRI_OUT, N)
    copy_subset(CT_SRC, CT_OUT, N)

    print("\n🎉 Small dataset created successfully!")
    print("Path: data/chaos_small/")
    print("Use this in config for fast training.")
