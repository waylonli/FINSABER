from __future__ import annotations

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from backtest.data_util.finsaber_dataset import FinsaberDataset
from backtest.data_util.finsaber_parquet_dataset import FinsaberParquetDataset
from backtest.data_util.trading_data import TradingData


DEFAULT_FINSABER2_DATA_ROOT = Path(r"I:\Data\finsaber2\sp500_2000_2025_parquet")


def get_finsaber2_data_root(root: str | Path | None = None) -> Path:
    """Resolve the FINSABER-2 parquet dataset root.

    Priority: explicit argument, ``FINSABER_DATA_ROOT`` environment variable,
    then the local development default used for this project.
    """

    return Path(root or os.getenv("FINSABER_DATA_ROOT") or DEFAULT_FINSABER2_DATA_ROOT)


def create_finsaber2_data_loader(
    root: str | Path | None = None,
    *,
    tickers: Iterable[str] | str | None = None,
    modalities: Iterable[str] = ("price", "news", "filing_k", "filing_q"),
    price_field: str = "adjusted_close",
    filing_merge_policy: str = "concat",
) -> FinsaberParquetDataset:
    """Create the default FINSABER-2 parquet data loader."""

    data_root = get_finsaber2_data_root(root)
    if not (data_root / "price_daily").exists():
        raise FileNotFoundError(
            f"FINSABER-2 parquet dataset not found at {data_root}. "
            "Set FINSABER_DATA_ROOT or pass --data_root to the experiment script."
        )
    return FinsaberParquetDataset(
        data_root,
        tickers=tickers,
        modalities=modalities,
        price_field=price_field,
        filing_merge_policy=filing_merge_policy,
    )


def resolve_trading_data(
    *,
    data_loader: TradingData | None = None,
    market_data_root: str | Path | None = None,
    market_data_info_path: str | Path | None = None,
    tickers: Iterable[str] | str | None = None,
    modalities: Iterable[str] = ("price", "news", "filing_k", "filing_q"),
    filing_merge_policy: str = "concat",
) -> TradingData:
    """Resolve strategy data input while preserving legacy pickle compatibility."""

    if data_loader is not None:
        return data_loader

    candidate = market_data_root or market_data_info_path
    if candidate is not None:
        candidate_path = Path(candidate)
        if candidate_path.suffix == ".pkl":
            if not candidate_path.exists():
                raise FileNotFoundError(
                    f"Legacy pickle dataset not found at {candidate_path}. "
                    "Use market_data_root/data_loader for FINSABER-2 parquet data."
                )
            warnings.warn(
                "Loading legacy pickle datasets is deprecated on main. "
                "Prefer FinsaberParquetDataset or market_data_root.",
                DeprecationWarning,
                stacklevel=2,
            )
            return FinsaberDataset(
                pickle_file=str(candidate_path),
                source_kind="legacy_pickle",
                filing_payload_kind="section_text",
            )
        return create_finsaber2_data_loader(
            candidate_path,
            tickers=tickers,
            modalities=modalities,
            filing_merge_policy=filing_merge_policy,
        )

    return create_finsaber2_data_loader(
        tickers=tickers,
        modalities=modalities,
        filing_merge_policy=filing_merge_policy,
    )


def trading_data_to_env_dict(
    data_loader: TradingData,
    *,
    start_date=None,
    end_date=None,
    tickers: Iterable[str] | str | None = None,
) -> dict:
    """Materialize a ``TradingData`` window into the dict shape expected by agents."""

    start_date = _normalize_date(start_date)
    end_date = _normalize_date(end_date)
    ticker_set = _normalize_tickers(tickers)

    subset = data_loader
    if start_date is not None and end_date is not None:
        subset = data_loader.get_subset_by_time_range(start_date, end_date)
        if subset is None:
            return {}

    result = {}
    for date in subset.get_date_range():
        date = _normalize_date(date)
        if start_date is not None and date < start_date:
            continue
        if end_date is not None and date > end_date:
            continue

        day = subset.get_data_by_date(date) or {}
        record = {"price": {}, "news": {}, "filing_k": {}, "filing_q": {}}
        for modality, values in day.items():
            if not isinstance(values, dict):
                continue
            target = record.setdefault(modality, {})
            for ticker, value in values.items():
                if ticker_set is not None and ticker not in ticker_set:
                    continue
                target[ticker] = value
        if record["price"]:
            result[date] = record

    return result


def _normalize_date(value):
    if value is None:
        return None
    if isinstance(value, str):
        return pd.to_datetime(value).date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    return value


def _normalize_tickers(tickers):
    if tickers is None or tickers == "all":
        return None
    if isinstance(tickers, str):
        return {tickers}
    return set(tickers)
