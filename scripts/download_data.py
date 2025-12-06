import os
import zipfile
import shutil
from pathlib import Path
import requests


def download_file(url: str, dest: Path) -> None:
    """Download a file from a URL if it doesn't already exist."""
    if dest.exists():
        print(f"File already exists: {dest}")
        return
    # Try a few URL variants to accommodate Zenodo's download links
    attempts = [url]
    if "?" not in url:
        attempts.append(url + "?download=1")
    # try swapping 'records' -> 'record' if present
    if "/records/" in url:
        attempts.append(url.replace("/records/", "/record/"))
        if "?" not in url:
            attempts.append(url.replace("/records/", "/record/") + "?download=1")

    last_exc = None
    for attempt in attempts:
        try:
            print(f"Downloading {attempt} to {dest}...")
            resp = requests.get(attempt, stream=True, allow_redirects=True, timeout=30)
        except Exception as e:
            print(f"Error contacting {attempt}: {e}")
            last_exc = e
            continue
        if resp.status_code == 200:
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            print(f"Downloaded: {dest}")
            return
        else:
            print(f"Got HTTP {resp.status_code} for {attempt}, trying next...")
            last_exc = RuntimeError(f"HTTP {resp.status_code} for {attempt}")

    # If we reach here, all attempts failed
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to download {url}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP file into the target folder. Skips if extraction already looks complete."""
    if not zip_path.exists():
        print(f"ZIP not found, skipping extract: {zip_path}")
        return
    # Heuristic: if there's a folder with same base name already, skip
    base = zip_path.stem
    candidate = extract_to / base
    if candidate.exists() and any(candidate.iterdir()):
        print(f"Extraction appears present, skipping: {candidate}")
        return
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"Extracted: {zip_path}")


def summarize_folder(folder: Path) -> None:
    total_files = 0
    counts = {}
    for root, _, files in os.walk(folder):
        for f in files:
            total_files += 1
            ext = os.path.splitext(f)[1].lower()
            counts[ext] = counts.get(ext, 0) + 1
    print(f"Summary for {folder}:")
    print(f"  Total files: {total_files}")
    for ext, c in sorted(counts.items()):
        print(f"  {ext or '[no_ext]'}: {c}")


def main() -> None:
    raw_dir = Path("data/raw/chaos")
    raw_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "CHAOS_Train_Sets.zip": "https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip?download=1",
        "CHAOS_Test_Sets.zip": "https://zenodo.org/records/3431873/files/CHAOS_Test_Sets.zip?download=1",
    }

    for name, url in urls.items():
        zip_path = raw_dir / name
        download_file(url, zip_path)
        extract_zip(zip_path, raw_dir)

    # Optional: keep a small README note
    readme = raw_dir / "README_DOWNLOAD.txt"
    readme.write_text("Downloaded CHAOS Train/Test zips from Zenodo. Extracted to this folder.")

    summarize_folder(raw_dir)


if __name__ == "__main__":
    main()