"""
Download small MRI/CT sample volumes from MONAI extra test data.
Targets a lightweight set so CPU-only training is feasible.
"""
from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

ASSETS = {
    "testing_ixi_t1.tar.gz": (
        "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/testing_ixi_t1.tar.gz",
        None,  # checksum not provided; kept for future verification
    ),
    "Prostate_T2W_AX_1.nii": (
        "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/Prostate_T2W_AX_1.nii",
        None,
    ),
    "spleen_test.nii.gz": (
        "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/spleen_test.nii.gz",
        None,
    ),
    "copd1_highres_INSP_STD_COPD_img.nii.gz": (
        "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/copd1_highres_INSP_STD_COPD_img.nii.gz",
        None,
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_extract_tar(path: Path, dest: Path) -> None:
    if path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(dest)


def download_asset(name: str, url: str, checksum: str | None, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    if target.exists():
        print(f"Skipping {name}; already present.")
        return target
    print(f"Downloading {name}...")
    urlretrieve(url, target)
    if checksum:
        observed = sha256(target)
        if observed != checksum:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {name}: expected {checksum}, got {observed}")
    if name.endswith(".tar.gz"):
        maybe_extract_tar(target, out_dir)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download small MRI/CT sample volumes.")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="Directory to store downloads.")
    args = parser.parse_args()

    for fname, (url, checksum) in ASSETS.items():
        download_asset(fname, url, checksum, args.out)

    print("Done. Raw files are in", args.out.resolve())


if __name__ == "__main__":
    main()
