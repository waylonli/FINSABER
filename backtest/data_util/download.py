"""Download FINSABER dataset files from HuggingFace if they are missing locally."""

import os

HF_REPO_ID = "waylonli/FINSABER-data"

# Files to download: (path_in_repo, local_path)
DATASET_FILES = [
    ("data/finmem_data/stock_data_cherrypick_2000_2024.pkl",
     os.path.join("data", "finmem_data", "stock_data_cherrypick_2000_2024.pkl")),
    ("data/finmem_data/stock_data_sp500_2000_2024.pkl",
     os.path.join("data", "finmem_data", "stock_data_sp500_2000_2024.pkl")),
]


def ensure_datasets(files=None):
    """Check that required dataset files exist; download from HuggingFace if missing.

    Args:
        files: Optional list of (repo_path, local_path) tuples.
               Defaults to DATASET_FILES.
    """
    files = files or DATASET_FILES
    missing = [(repo_path, local_path) for repo_path, local_path in files
               if not os.path.exists(local_path)]

    if not missing:
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Warning: huggingface_hub is not installed. Cannot download missing datasets.")
        print("Install with: pip install huggingface_hub")
        print("Missing files:")
        for _, local_path in missing:
            print(f"  - {local_path}")
        return

    failed = []
    for repo_path, local_path in missing:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {repo_path} from {HF_REPO_ID} ...")
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=repo_path,
                repo_type="dataset",
                local_dir=".",
            )
            print(f"  Saved to {local_path}")
        except Exception as e:
            print(f"  Failed to download {repo_path}: {e}")
            failed.append(local_path)

    if failed:
        print("\nWarning: Some dataset files could not be downloaded:")
        for path in failed:
            print(f"  - {path}")
        print("You may need to download them manually from "
              f"https://huggingface.co/datasets/{HF_REPO_ID}")
    else:
        print("Dataset download complete.")
