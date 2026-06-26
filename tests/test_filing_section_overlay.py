from datetime import date

import pandas as pd
import pytest

from backtest.data_util import FinsaberDataset, create_finsaber2_data_loader
from backtest.data_util.filing_section_extractor import (
    FilingSectionOverlayDataset,
    with_filing_sections,
)
from backtest.data_util.filing_section_extractor import dataset_overlay as overlay_module


def _narrative_lines(prefix: str, count: int) -> list[str]:
    sentence = (
        "The company improved revenue quality, expanded operating margin, "
        "managed expenses carefully, and invested in long-term platform "
        "capabilities across global operations."
    )
    return [f"{prefix} line {index}. {sentence}" for index in range(1, count + 1)]


def _build_10k_with_item_7() -> str:
    lines = [
        "ANNUAL REPORT",
        "PART I",
        "ITEM 1. BUSINESS",
    ]
    lines.extend(
        f"Business background line {index}. Prior-period operating context."
        for index in range(1, 191)
    )
    lines.extend(
        [
            "PART II",
            "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        ]
    )
    lines.extend(_narrative_lines("Management analysis", 24))
    lines.extend(
        [
            "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
            "Market risk summary.",
        ]
    )
    return "\n".join(lines)


def _build_10q_with_item_2() -> str:
    lines = [
        "QUARTERLY REPORT",
        "PART I",
        "ITEM 1. FINANCIAL STATEMENTS",
    ]
    lines.extend(
        f"Financial statement note line {index}. Historical accounting detail."
        for index in range(1, 191)
    )
    lines.append(
        "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS"
    )
    lines.extend(_narrative_lines("Quarterly management analysis", 20))
    lines.extend(
        [
            "ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
            "Market risk summary.",
        ]
    )
    return "\n".join(lines)


def _build_10q_without_item_2() -> str:
    lines = [
        "QUARTERLY REPORT",
        "PART I",
        "ITEM 1. FINANCIAL STATEMENTS",
    ]
    lines.extend(
        f"Financial statement note line {index}. Historical accounting detail."
        for index in range(1, 191)
    )
    lines.extend(
        [
            "ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
            "Market risk summary.",
        ]
    )
    return "\n".join(lines)


def test_overlay_extracts_sections_without_mutating_base_loader():
    current_date = date(2024, 1, 2)
    raw_k = _build_10k_with_item_7()
    raw_q = _build_10q_with_item_2()
    loader = FinsaberDataset(
        data={
            current_date: {
                "price": {
                    "AAA": {
                        "close": 10.0,
                        "adjusted_close": 10.0,
                        "volume": 100,
                    }
                },
                "news": {"AAA": ["known before signal"]},
                "filing_k": {"AAA": raw_k},
                "filing_q": {"AAA": raw_q},
            }
        }
    )
    section_map = {
        "filing_k": {"form": "10-K", "item_key": "item_7"},
        "filing_q": {"form": "10-Q", "item_key": "part_i_item_2"},
    }

    overlay = with_filing_sections(loader, section_map=section_map)
    transformed_day = overlay.get_data_by_date(current_date)
    transformed_ticker = overlay.get_ticker_data_by_date("AAA", current_date)

    assert transformed_day["price"]["AAA"]["adjusted_close"] == 10.0
    assert transformed_day["news"]["AAA"] == ["known before signal"]
    assert transformed_day["filing_k"]["AAA"].startswith(
        "Part II Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    )
    assert transformed_day["filing_q"]["AAA"].startswith(
        "Part I Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    )
    assert transformed_ticker["filing_k"] == transformed_day["filing_k"]["AAA"]
    assert transformed_ticker["filing_q"] == transformed_day["filing_q"]["AAA"]

    base_day = loader.get_data_by_date(current_date)
    assert base_day["filing_k"]["AAA"] == raw_k
    assert base_day["filing_q"]["AAA"] == raw_q


def test_overlay_returns_empty_string_when_extraction_fails():
    current_date = date(2024, 1, 2)
    loader = FinsaberDataset(
        data={
            current_date: {
                "price": {
                    "AAA": {
                        "close": 10.0,
                        "adjusted_close": 10.0,
                        "volume": 100,
                    }
                },
                "filing_q": {"AAA": _build_10q_without_item_2()},
            }
        }
    )
    overlay = FilingSectionOverlayDataset(
        loader,
        section_map={"filing_q": {"form": "10-Q", "item_key": "part_i_item_2"}},
        failure_mode="empty",
    )

    transformed_day = overlay.get_data_by_date(current_date)
    assert transformed_day["filing_q"]["AAA"] == ""


def test_overlay_rejects_non_string_filing_payloads():
    current_date = date(2024, 1, 2)
    loader = FinsaberDataset(
        data={
            current_date: {
                "price": {
                    "AAA": {
                        "close": 10.0,
                        "adjusted_close": 10.0,
                        "volume": 100,
                    }
                },
                "filing_q": {"AAA": ["not", "a", "filing", "string"]},
            }
        }
    )
    overlay = FilingSectionOverlayDataset(
        loader,
        section_map={"filing_q": {"form": "10-Q", "item_key": "part_i_item_2"}},
        failure_mode="empty",
    )

    with pytest.raises(TypeError, match="expects filing modalities to contain string or missing payloads"):
        overlay.get_data_by_date(current_date)


def test_overlay_reuses_cache_across_subset_loaders(monkeypatch):
    current_date = date(2024, 1, 2)
    loader = FinsaberDataset(
        data={
            current_date: {
                "price": {
                    "AAA": {
                        "close": 10.0,
                        "adjusted_close": 10.0,
                        "volume": 100,
                    }
                },
                "filing_q": {"AAA": _build_10q_with_item_2()},
            }
        }
    )
    overlay = FilingSectionOverlayDataset(
        loader,
        section_map={"filing_q": {"form": "10-Q", "item_key": "part_i_item_2"}},
        failure_mode="empty",
    )

    calls = {"count": 0}
    real_extract = overlay_module.extract_items_from_filing

    def counted_extract(*args, **kwargs):
        calls["count"] += 1
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(overlay_module, "extract_items_from_filing", counted_extract)

    overlay.get_data_by_date(current_date)
    subset = overlay.get_subset_by_time_range("2024-01-01", "2024-01-03")
    assert subset is not None
    subset.get_data_by_date(current_date)

    ticker_subset = overlay.get_ticker_subset_by_time_range("AAA", "2024-01-01", "2024-01-03")
    assert ticker_subset is not None
    ticker_subset.get_data_by_date(current_date)

    assert calls["count"] == 1


def test_overlay_extracts_sections_from_parquet_loader(tmp_path):
    current_date = pd.Timestamp("2024-01-02")
    price_dir = tmp_path / "price_daily" / "year=2024"
    price_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": current_date,
                "symbol": "AAA",
                "cik": "0001",
                "open": 10.0,
                "high": 12.0,
                "low": 9.0,
                "close": 10.0,
                "adjusted_close": 20.0,
                "volume": 100,
            }
        ]
    ).to_parquet(price_dir / "part-000.parquet", index=False)

    filing_dir = tmp_path / "filingq" / "year=2024"
    filing_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": current_date,
                "symbol": "AAA",
                "cik": "0001",
                "filing_idx": 0,
                "filing_text": _build_10q_with_item_2(),
                "text_len": 0,
                "year": 2024,
                "accession": "first",
            }
        ]
    ).to_parquet(filing_dir / "part-000.parquet", index=False)

    loader = create_finsaber2_data_loader(
        tmp_path,
        tickers=["AAA"],
        modalities=("price", "filing_q"),
    )
    overlay = with_filing_sections(
        loader,
        section_map={"filing_q": {"form": "10-Q", "item_key": "part_i_item_2"}},
    )

    transformed_day = overlay.get_data_by_date("2024-01-02")
    assert transformed_day["price"]["AAA"]["adjusted_close"] == 20.0
    assert transformed_day["filing_q"]["AAA"].startswith(
        "Part I Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    )
