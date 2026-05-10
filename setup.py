import os
import subprocess
import sys
from pathlib import Path

from crucible import Smelter, DEFAULT_NPY_PATH, DEFAULT_OUT_PATH

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/pixel_art")
KAGGLE_DATASET = "ebrahimelgazar/pixel-art"

# ---------------------------------------------------------------------------
# Kaggle credential check
# ---------------------------------------------------------------------------

def check_kaggle_credentials() -> bool: 
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_json = kaggle_json.exists()
    has_env = bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )

    if has_json or has_env:
        source = "kaggle.json" if has_json else "environment variables"
        print(f"[Setup] Kaggle credentials found via {source}.")
        return True

    print(
        "\n[Setup] ERROR: Kaggle credentials not found.\n"
        "Provide them via one of these two methods:\n\n"
        f"  Method 1 — Place kaggle.json at:\n"
        f"             {kaggle_json}\n\n"
        "  Method 2 — Set environment variables:\n"
        "             KAGGLE_USERNAME=your_username\n"
        "             KAGGLE_KEY=your_api_key\n\n"
        "Download your API token from:\n"
        "  https://www.kaggle.com/settings  ->  API  ->  Create New Token\n"
    )
    return False


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_dataset() -> bool:
    if DEFAULT_NPY_PATH.exists():
        print(f"[Setup] Dataset already present at '{DEFAULT_NPY_PATH}'. Skipping download.")
        return True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Downloading dataset '{KAGGLE_DATASET}' ...")

    import shutil
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        # Fallback: look for kaggle in the same Scripts dir as the running Python
        scripts_dir = Path(sys.executable).parent
        for candidate in [scripts_dir / "kaggle", scripts_dir / "kaggle.exe"]:
            if candidate.exists():
                kaggle_bin = str(candidate)
                break

    if kaggle_bin is None:
        print(
            "\n[Setup] ERROR: 'kaggle' executable not found on PATH.\n"
            "Make sure it is installed: pip install kaggle\n"
        )
        return False

    result = subprocess.run(
        [
            kaggle_bin,
            "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(DATA_DIR),
            "--unzip",
        ],
        capture_output=False,
    )

    if result.returncode != 0:
        print(
            f"\n[Setup] ERROR: Kaggle download failed (exit code {result.returncode}).\n"
            "Make sure the kaggle package is installed: pip install kaggle\n"
            "Also verify your credentials are correct.\n"
        )
        return False

    if not DEFAULT_NPY_PATH.exists():
        print(
            f"\n[Setup] ERROR: Download completed but '{DEFAULT_NPY_PATH}' not found.\n"
            "The dataset structure may have changed. Check the Kaggle dataset page:\n"
            "  https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art\n"
        )
        return False

    print(f"[Setup] Download complete. sprites.npy found at '{DEFAULT_NPY_PATH}'.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  The Crucible — First-Run Setup")
    print("=" * 60)

    # Step 1: Kaggle credentials
    if not check_kaggle_credentials():
        sys.exit(1)

    # Step 2: Download dataset
    if not download_dataset():
        sys.exit(1)

    # Step 3: Filter sprites
    print(f"\n[Setup] Running Smelter to filter sprites ...")
    smelter = Smelter()
    try:
        clean = smelter.smelt(DEFAULT_NPY_PATH)
    except FileNotFoundError as e:
        print(f"[Setup] ERROR: {e}")
        sys.exit(1)

    # Step 4: Save clean array
    DEFAULT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.save(str(DEFAULT_OUT_PATH), clean)
    print(f"[Setup] Saved {len(clean)} clean sprites to '{DEFAULT_OUT_PATH}'.")

    print("\n[Setup] Setup complete. Run the app with:")
    print("         python app.py\n")


if __name__ == "__main__":
    main()
