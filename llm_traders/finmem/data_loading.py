from __future__ import annotations

from typing import Mapping

from backtest.data_util import FinsaberParquetDataset, TradingData, resolve_trading_data
from backtest.data_util.filing_section_extractor import (
    FilingSectionOverlayDataset,
    with_filing_sections,
)


DEFAULT_FINMEM_SECTION_MAP = {
    "filing_k": {"form": "10-K", "item_key": "item_7"},
    "filing_q": {"form": "10-Q", "item_key": "part_i_item_2"},
}


def _copy_section_map(
    section_map: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    source = DEFAULT_FINMEM_SECTION_MAP if section_map is None else section_map
    return {
        str(modality): {
            "form": str(config["form"]),
            "item_key": str(config["item_key"]),
        }
        for modality, config in source.items()
    }


def _loader_tickers(data_loader: TradingData | None) -> list[str] | None:
    if data_loader is None:
        return None

    if hasattr(data_loader, "tickers"):
        tickers = getattr(data_loader, "tickers")
        if tickers is None:
            return None
        if isinstance(tickers, str):
            return None if tickers == "all" else [tickers]
        return list(tickers)

    if hasattr(data_loader, "get_tickers_list"):
        return list(data_loader.get_tickers_list())

    return None


def _narrow_data_loader_to_symbol(
    data_loader: TradingData | None,
    symbol: str,
) -> TradingData | None:
    if data_loader is None:
        return None

    loader_tickers = _loader_tickers(data_loader)
    if loader_tickers == [symbol]:
        return data_loader

    subset_start_date = getattr(data_loader, "start_date", None)
    subset_end_date = getattr(data_loader, "end_date", None)
    if (subset_start_date is None or subset_end_date is None) and hasattr(
        data_loader, "get_date_range"
    ):
        date_range = data_loader.get_date_range()
        if date_range:
            subset_start_date = subset_start_date or date_range[0]
            subset_end_date = subset_end_date or date_range[-1]

    if (
        subset_start_date is not None
        and subset_end_date is not None
        and hasattr(data_loader, "get_ticker_subset_by_time_range")
    ):
        subset = data_loader.get_ticker_subset_by_time_range(
            symbol,
            subset_start_date,
            subset_end_date,
        )
        if subset is not None:
            return subset

    rebuilt_loader = _rebuild_data_loader_with_symbol(data_loader, symbol)
    if rebuilt_loader is not None:
        return rebuilt_loader

    return data_loader


def _rebuild_data_loader_with_symbol(
    data_loader: TradingData,
    symbol: str,
) -> TradingData | None:
    if isinstance(data_loader, FilingSectionOverlayDataset):
        rebuilt_base_loader = _rebuild_data_loader_with_symbol(
            data_loader.base_loader,
            symbol,
        )
        if rebuilt_base_loader is None:
            return None
        # Preserve the existing section mapping and shared extraction cache when
        # rebuilding an overlay around a narrowed base loader.
        return with_filing_sections(
            rebuilt_base_loader,
            section_map=_copy_section_map(data_loader._section_map),
            failure_mode=data_loader.failure_mode,
            cache=data_loader._cache,
        )

    if isinstance(data_loader, FinsaberParquetDataset):
        # Rebuild the parquet loader with a single ticker so FinMem does not
        # materialize unrelated symbols before converting TradingData into the
        # agent-facing environment dict.
        return FinsaberParquetDataset(
            root=data_loader.root,
            start_date=data_loader.start_date,
            end_date=data_loader.end_date,
            tickers=[symbol],
            modalities=data_loader.modalities,
            price_field=data_loader.price_field,
            filing_merge_policy=data_loader.filing_merge_policy,
        )

    return None


def _ensure_filing_merge_policy(
    data_loader: TradingData | None,
    filing_merge_policy: str,
) -> TradingData | None:
    if data_loader is None:
        return None
    if isinstance(data_loader, FilingSectionOverlayDataset):
        return data_loader
    if (
        isinstance(data_loader, FinsaberParquetDataset)
        and data_loader.filing_merge_policy != filing_merge_policy
    ):
        # Rebuild the parquet loader so FinMem can request its preferred
        # duplicate-filing assembly policy even when a raw loader instance was
        # created earlier by a caller or experiment harness.
        return FinsaberParquetDataset(
            root=data_loader.root,
            start_date=data_loader.start_date,
            end_date=data_loader.end_date,
            tickers=data_loader.tickers,
            modalities=data_loader.modalities,
            price_field=data_loader.price_field,
            filing_merge_policy=filing_merge_policy,
        )
    return data_loader


def _unwrap_filing_section_overlay(
    data_loader: TradingData | None,
) -> TradingData | None:
    if isinstance(data_loader, FilingSectionOverlayDataset):
        return data_loader.base_loader
    return data_loader


def _validate_filing_payload_kind(filing_payload_kind: str) -> str:
    normalized = str(filing_payload_kind).strip().lower()
    if normalized not in {"auto", "raw_filing", "section_text"}:
        raise ValueError(
            "Unsupported filing_payload_kind. Expected 'auto', "
            "'raw_filing', or 'section_text'."
        )
    return normalized


def _resolve_filing_payload_kind(
    data_loader: TradingData,
    filing_payload_kind: str,
) -> str:
    normalized = _validate_filing_payload_kind(filing_payload_kind)
    if normalized != "auto":
        return normalized

    inferred = str(getattr(data_loader, "filing_payload_kind", "raw_filing")).strip().lower()
    if inferred not in {"raw_filing", "section_text"}:
        raise ValueError(
            "Unsupported inferred filing_payload_kind. Expected loader metadata "
            "to declare 'raw_filing' or 'section_text'."
        )
    return inferred


def prepare_finmem_trading_data(
    *,
    symbol: str,
    data_loader: TradingData | None = None,
    market_data_root=None,
    market_data_info_path=None,
    use_filing_sections: bool = True,
    filing_section_map: Mapping[str, Mapping[str, str]] | None = None,
    filing_payload_kind: str = "auto",
    filing_failure_mode: str = "empty",
    filing_merge_policy: str = "latest",
) -> TradingData:
    """Resolve FinMem data and optionally remap raw filings to target sections."""

    if not use_filing_sections:
        # Treat opt-out as a strict raw-data request: unwrap any existing filing
        # section overlay and skip merge-policy normalization or re-wrapping.
        narrowed_loader = _narrow_data_loader_to_symbol(
            _unwrap_filing_section_overlay(data_loader),
            symbol,
        )
        return resolve_trading_data(
            data_loader=narrowed_loader,
            market_data_root=market_data_root,
            market_data_info_path=market_data_info_path,
            tickers=[symbol],
        )

    narrowed_loader = _narrow_data_loader_to_symbol(data_loader, symbol)
    normalized_loader = _ensure_filing_merge_policy(
        narrowed_loader,
        filing_merge_policy,
    )
    market_data = resolve_trading_data(
        data_loader=normalized_loader,
        market_data_root=market_data_root,
        market_data_info_path=market_data_info_path,
        tickers=[symbol],
        filing_merge_policy=filing_merge_policy,
    )
    resolved_payload_kind = _resolve_filing_payload_kind(
        market_data,
        filing_payload_kind,
    )
    if (
        isinstance(market_data, FilingSectionOverlayDataset)
        or resolved_payload_kind == "section_text"
    ):
        return market_data

    return with_filing_sections(
        market_data,
        section_map=_copy_section_map(filing_section_map),
        failure_mode=filing_failure_mode,
    )
