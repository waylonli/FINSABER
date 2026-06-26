from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date as date_type
from typing import Any, Mapping

import pandas as pd

from backtest.data_util.trading_data import TradingData

from .upstream_extractor import (
    FilingRow,
    ItemSpec,
    audit_extraction_result,
    extract_items_from_filing,
    item_specs_for_request,
    resolve_requested_items,
)


@dataclass(frozen=True)
class SectionRequest:
    modality: str
    form: str
    item_key: str
    item_spec: ItemSpec


@dataclass(frozen=True)
class SectionCacheEntry:
    section_text: str | None
    section_status: str
    audit_status: str
    missing_items: tuple[str, ...]
    audit_flags: tuple[str, ...]


class FilingSectionOverlayDataset(TradingData):
    """TradingData wrapper that remaps filing modalities to extracted sections."""

    def __init__(
        self,
        base_loader: TradingData,
        *,
        section_map: Mapping[str, Mapping[str, str]],
        failure_mode: str = "empty",
        cache: dict[tuple[str, str, str, str], SectionCacheEntry] | None = None,
    ):
        if not isinstance(base_loader, TradingData):
            raise TypeError("base_loader must implement TradingData")
        self.base_loader = base_loader
        self.failure_mode = self._validate_failure_mode(failure_mode)
        self.source_kind = getattr(base_loader, "source_kind", "unknown")
        self.filing_payload_kind = "section_text"
        self._section_requests = self._normalize_section_map(section_map)
        self._section_map = {
            modality: {"form": request.form, "item_key": request.item_key}
            for modality, request in self._section_requests.items()
        }
        self._cache = cache if cache is not None else {}

    @staticmethod
    def _validate_failure_mode(failure_mode: str) -> str:
        normalized = str(failure_mode).strip().lower()
        if normalized not in {"empty", "raw", "raise"}:
            raise ValueError(
                "Unsupported failure_mode. Expected 'empty', 'raw', or 'raise'."
            )
        return normalized

    @staticmethod
    def _normalize_date(value) -> date_type:
        if isinstance(value, str):
            return pd.to_datetime(value).date()
        if isinstance(value, pd.Timestamp):
            return value.date()
        return value

    @staticmethod
    def _normalize_raw_text(raw_text: Any) -> str:
        # Filing extraction should only run on textual payloads. Failing fast
        # here keeps wiring mistakes visible instead of silently converting
        # unrelated payloads such as lists or dicts into bogus filing text.
        if raw_text is None:
            return ""
        if isinstance(raw_text, str):
            return raw_text
        if raw_text is pd.NA:
            return ""
        if isinstance(raw_text, float) and pd.isna(raw_text):
            return ""
        raise TypeError(
            "Filing section overlay expects filing modalities to contain "
            "string or missing payloads."
        )

    @staticmethod
    def _build_cache_key(
        *,
        modality: str,
        form: str,
        item_key: str,
        raw_text: str,
    ) -> tuple[str, str, str, str]:
        text_hash = hashlib.sha256(raw_text.encode("utf-8", "ignore")).hexdigest()
        return modality, form, item_key, text_hash

    @staticmethod
    def _normalize_section_map(
        section_map: Mapping[str, Mapping[str, str]],
    ) -> dict[str, SectionRequest]:
        if not section_map:
            raise ValueError("section_map must not be empty")

        normalized: dict[str, SectionRequest] = {}
        for modality, config in section_map.items():
            if not isinstance(config, Mapping):
                raise TypeError(
                    f"section_map[{modality!r}] must be a mapping with 'form' and 'item_key'."
                )

            form = str(config.get("form", "")).strip().upper()
            item_key = str(config.get("item_key", "")).strip().lower()
            if not form or not item_key:
                raise ValueError(
                    f"section_map[{modality!r}] must define non-empty 'form' and 'item_key'."
                )

            requested_items = resolve_requested_items([form], [item_key])
            specs = item_specs_for_request([form], requested_items)
            if len(specs) != 1:
                raise ValueError(
                    f"section_map[{modality!r}] must resolve to exactly one item spec."
                )

            normalized[str(modality)] = SectionRequest(
                modality=str(modality),
                form=form,
                item_key=item_key,
                item_spec=specs[0],
            )

        return normalized

    def _build_filing_row(
        self,
        *,
        ticker: str,
        current_date,
        form: str,
        raw_text: str,
    ) -> FilingRow:
        normalized_date = self._normalize_date(current_date)
        return FilingRow(
            date=str(normalized_date),
            symbol=ticker,
            cik="",
            accession="",
            form=form,
            year=normalized_date.year,
            filing_idx=0,
            filing_text=raw_text,
            source_file="",
            source_row_idx=0,
        )

    def _compute_cache_entry(
        self,
        *,
        request: SectionRequest,
        ticker: str,
        current_date,
        raw_text: str,
    ) -> SectionCacheEntry:
        filing = self._build_filing_row(
            ticker=ticker,
            current_date=current_date,
            form=request.form,
            raw_text=raw_text,
        )
        extraction_result = extract_items_from_filing(
            filing=filing,
            item_specs=[request.item_spec],
        )
        audit_result = audit_extraction_result(
            filing=filing,
            extraction_result=extraction_result,
            item_specs=[request.item_spec],
        )

        section = next(
            (
                payload
                for payload in extraction_result["sections"]
                if payload["item_key"] == request.item_key
            ),
            None,
        )
        item_result = next(
            (
                payload
                for payload in audit_result["item_results"]
                if payload["item_key"] == request.item_key
            ),
            None,
        )
        return SectionCacheEntry(
            section_text=None if section is None else str(section.get("section_text") or ""),
            section_status=str(extraction_result["section_status"]),
            audit_status=str(audit_result["audit_status"]),
            missing_items=tuple(str(item) for item in extraction_result["missing_items"]),
            audit_flags=tuple(
                str(flag)
                for flag in ([] if item_result is None else item_result.get("flags", []))
            ),
        )

    def _handle_failed_extraction(
        self,
        *,
        modality: str,
        ticker: str,
        current_date,
        raw_text: str,
        entry: SectionCacheEntry,
    ) -> str:
        if self.failure_mode == "empty":
            return ""
        if self.failure_mode == "raw":
            return raw_text
        raise ValueError(
            "Filing section extraction failed for "
            f"modality={modality!r}, ticker={ticker!r}, date={self._normalize_date(current_date)!s}, "
            f"section_status={entry.section_status!r}, audit_status={entry.audit_status!r}, "
            f"flags={list(entry.audit_flags)!r}."
        )

    def _extract_section_text(
        self,
        *,
        modality: str,
        ticker: str,
        current_date,
        raw_text: str,
        request: SectionRequest,
    ) -> str:
        normalized_text = self._normalize_raw_text(raw_text)
        if not normalized_text.strip():
            return ""

        cache_key = self._build_cache_key(
            modality=modality,
            form=request.form,
            item_key=request.item_key,
            raw_text=normalized_text,
        )
        if cache_key not in self._cache:
            self._cache[cache_key] = self._compute_cache_entry(
                request=request,
                ticker=ticker,
                current_date=current_date,
                raw_text=normalized_text,
            )

        entry = self._cache[cache_key]
        if entry.section_text and entry.audit_status != "fail":
            return entry.section_text
        return self._handle_failed_extraction(
            modality=modality,
            ticker=ticker,
            current_date=current_date,
            raw_text=normalized_text,
            entry=entry,
        )

    def _transform_day_data(self, current_date, day: dict[str, Any]) -> dict[str, Any]:
        if not day:
            return day

        transformed_day = dict(day)
        for modality, request in self._section_requests.items():
            values = day.get(modality)
            if not isinstance(values, dict) or not values:
                continue

            transformed_values = dict(values)
            changed = False
            for ticker, raw_text in values.items():
                new_text = self._extract_section_text(
                    modality=modality,
                    ticker=ticker,
                    current_date=current_date,
                    raw_text=raw_text,
                    request=request,
                )
                if new_text != raw_text:
                    changed = True
                transformed_values[ticker] = new_text

            if changed:
                transformed_day[modality] = transformed_values

        return transformed_day

    def get_data_by_date(self, date) -> dict[str, Any]:
        current_date = self._normalize_date(date)
        day = self.base_loader.get_data_by_date(current_date) or {}
        return self._transform_day_data(current_date, day)

    def get_ticker_price_by_date(self, ticker: str, date, price_field: str | None = None) -> float:
        return self.base_loader.get_ticker_price_by_date(ticker, date, price_field)

    def get_ticker_data_by_date(self, ticker: str, date) -> dict[str, Any]:
        daily_data = self.get_data_by_date(date)
        return {
            modality: values[ticker]
            for modality, values in daily_data.items()
            if isinstance(values, dict) and ticker in values
        }

    def get_tickers_list(self) -> list[str]:
        return self.base_loader.get_tickers_list()

    def get_subset_by_time_range(self, start_date, end_date):
        subset = self.base_loader.get_subset_by_time_range(start_date, end_date)
        if subset is None:
            return None
        return FilingSectionOverlayDataset(
            subset,
            section_map=self._section_map,
            failure_mode=self.failure_mode,
            cache=self._cache,
        )

    def get_ticker_subset_by_time_range(self, ticker: str, start_date, end_date):
        subset = self.base_loader.get_ticker_subset_by_time_range(ticker, start_date, end_date)
        if subset is None:
            return None
        return FilingSectionOverlayDataset(
            subset,
            section_map=self._section_map,
            failure_mode=self.failure_mode,
            cache=self._cache,
        )

    def get_date_range(self) -> list:
        return self.base_loader.get_date_range()

    def __getattr__(self, name: str):
        return getattr(self.base_loader, name)


def with_filing_sections(
    data_loader: TradingData,
    *,
    section_map: Mapping[str, Mapping[str, str]] | None,
    failure_mode: str = "empty",
    cache: dict[tuple[str, str, str, str], SectionCacheEntry] | None = None,
) -> TradingData:
    if not section_map:
        return data_loader
    return FilingSectionOverlayDataset(
        data_loader,
        section_map=section_map,
        failure_mode=failure_mode,
        cache=cache,
    )
