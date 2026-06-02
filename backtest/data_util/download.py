"""Helpers for obtaining the public FINSABER-2 dataset from HuggingFace."""

from __future__ import annotations

import os
from pathlib import Path

from backtest.data_util.dataset_factory import get_finsaber2_data_root


HF_REPO_ID = "finsaber-team/FINSABER-V2-Data"


def ensure_datasets(local_dir: str | os.PathLike | None = None):
    """Download the FINSABER-2 parquet dataset snapshot if it is missing.

    The preferred workflow is to set ``FINSABER_DATA_ROOT`` to an existing
    parquet dataset root. This helper is provided for scripts that want to
    fetch the public HuggingFace snapshot into that location.
    """

    data_root = Path(local_dir) if local_dir is not None else get_finsaber2_data_root()
    if (data_root / "price_daily").exists():
        return data_root

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download FINSABER-2 data. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    data_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(data_root),
        local_dir_use_symlinks=False,
    )
    return data_root
