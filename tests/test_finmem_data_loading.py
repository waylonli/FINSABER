import pickle
from datetime import date

import pandas as pd
import pytest

from backtest.data_util import FinsaberDataset, FinsaberParquetDataset
from backtest.data_util.filing_section_extractor import (
    FilingSectionOverlayDataset,
    with_filing_sections,
)
from llm_traders.finmem.data_loading import prepare_finmem_trading_data


def _narrative_lines(prefix: str, count: int) -> list[str]:
    sentence = (
        "The company improved revenue quality, expanded operating margin, "
        "managed expenses carefully, and invested in long-term platform "
        "capabilities across global operations."
    )
    return [f"{prefix} line {index}. {sentence}" for index in range(1, count + 1)]


def _build_10k_with_item_7(prefix: str = "Management analysis") -> str:
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
    lines.extend(_narrative_lines(prefix, 24))
    lines.extend(
        [
            "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
            "Market risk summary.",
        ]
    )
    return "\n".join(lines)


def _build_10q_with_item_2(prefix: str = "Quarterly management analysis") -> str:
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
    lines.extend(_narrative_lines(prefix, 20))
    lines.extend(
        [
            "ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
            "Market risk summary.",
        ]
    )
    return "\n".join(lines)


def test_prepare_finmem_trading_data_wraps_loader_and_extracts_expected_sections():
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
                "filing_k": {"AAA": _build_10k_with_item_7()},
                "filing_q": {"AAA": _build_10q_with_item_2()},
            }
        }
    )

    prepared = prepare_finmem_trading_data(symbol="AAA", data_loader=loader)
    day = prepared.get_data_by_date(current_date)

    assert isinstance(prepared, FilingSectionOverlayDataset)
    assert day["filing_k"]["AAA"].startswith(
        "Part II Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    )
    assert day["filing_q"]["AAA"].startswith(
        "Part I Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    )


def test_prepare_finmem_trading_data_can_leave_raw_filings_untouched():
    current_date = date(2024, 1, 2)
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
                "filing_q": {"AAA": raw_q},
            }
        }
    )

    prepared = prepare_finmem_trading_data(
        symbol="AAA",
        data_loader=loader,
        use_filing_sections=False,
    )

    assert prepared is loader
    assert prepared.get_data_by_date(current_date)["filing_q"]["AAA"] == raw_q


def test_prepare_finmem_trading_data_rebuilds_parquet_loader_for_latest_merge_policy(
    tmp_path,
):
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
    first_q = _build_10q_with_item_2(prefix="First duplicate filing")
    second_q = _build_10q_with_item_2(prefix="Second duplicate filing")
    pd.DataFrame(
        [
            {
                "date": current_date,
                "symbol": "AAA",
                "cik": "0001",
                "filing_idx": 0,
                "filing_text": first_q,
            },
            {
                "date": current_date,
                "symbol": "AAA",
                "cik": "0001",
                "filing_idx": 1,
                "filing_text": second_q,
            },
        ]
    ).to_parquet(filing_dir / "part-000.parquet", index=False)

    raw_loader = FinsaberParquetDataset(
        tmp_path,
        tickers=["AAA"],
        modalities=("price", "filing_q"),
    )

    prepared = prepare_finmem_trading_data(
        symbol="AAA",
        data_loader=raw_loader,
        filing_merge_policy="latest",
    )

    assert isinstance(prepared, FilingSectionOverlayDataset)
    assert isinstance(prepared.base_loader, FinsaberParquetDataset)
    assert prepared.base_loader is not raw_loader
    assert prepared.base_loader.filing_merge_policy == "latest"

    extracted = prepared.get_data_by_date("2024-01-02")["filing_q"]["AAA"]
    assert "Second duplicate filing line 1." in extracted
    assert "First duplicate filing line 1." not in extracted


def test_prepare_finmem_trading_data_rebuilds_overlay_base_loader_for_latest_merge_policy(
    tmp_path,
):
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
    first_q = _build_10q_with_item_2(prefix="First duplicate filing")
    second_q = _build_10q_with_item_2(prefix="Second duplicate filing")
    pd.DataFrame(
        [
            {
                "date": current_date,
                "symbol": "AAA",
                "cik": "0001",
                "filing_idx": 0,
                "filing_text": first_q,
            },
            {
                "date": current_date,
                "symbol": "AAA",
                "cik": "0001",
                "filing_idx": 1,
                "filing_text": second_q,
            },
        ]
    ).to_parquet(filing_dir / "part-000.parquet", index=False)

    raw_loader = FinsaberParquetDataset(
        tmp_path,
        tickers=["AAA"],
        modalities=("price", "filing_q"),
        filing_merge_policy="concat",
    )
    overlay_loader = with_filing_sections(
        raw_loader,
        section_map={"filing_q": {"form": "10-Q", "item_key": "part_i_item_2"}},
        failure_mode="empty",
    )

    prepared = prepare_finmem_trading_data(
        symbol="AAA",
        data_loader=overlay_loader,
        filing_merge_policy="latest",
    )

    assert isinstance(prepared, FilingSectionOverlayDataset)
    assert prepared is not overlay_loader
    assert isinstance(prepared.base_loader, FinsaberParquetDataset)
    assert prepared.base_loader.filing_merge_policy == "latest"

    extracted = prepared.get_data_by_date("2024-01-02")["filing_q"]["AAA"]
    assert "Second duplicate filing line 1." in extracted
    assert "First duplicate filing line 1." not in extracted


def test_prepare_finmem_trading_data_extracts_sections_from_legacy_raw_filing_pickles(tmp_path):
    current_date = date(2024, 1, 2)
    raw_q = _build_10q_with_item_2()
    payload = {
        current_date: {
            "price": {
                "AAA": {
                    "close": 10.0,
                    "adjusted_close": 10.0,
                    "volume": 100,
                }
            },
            "filing_q": {"AAA": raw_q},
        }
    }
    pickle_path = tmp_path / "legacy_env.pkl"
    with open(pickle_path, "wb") as file:
        pickle.dump(payload, file)

    prepared = prepare_finmem_trading_data(
        symbol="AAA",
        market_data_info_path=pickle_path,
    )

    assert isinstance(prepared, FilingSectionOverlayDataset)
    assert isinstance(prepared.base_loader, FinsaberDataset)
    assert prepared.base_loader.source_kind == "legacy_pickle"
    assert prepared.base_loader.filing_payload_kind == "raw_filing"
    assert prepared.get_data_by_date(current_date)["filing_q"]["AAA"].startswith(
        "Part I Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    )


def test_prepare_finmem_trading_data_preserves_legacy_section_payloads_when_declared(
    tmp_path,
):
    current_date = date(2024, 1, 2)
    section_text = (
        "Management discussion line 1. The company improved revenue quality.\n"
        "Management discussion line 2. The company expanded operating margin."
    )
    payload = {
        current_date: {
            "price": {
                "AAA": {
                    "close": 10.0,
                    "adjusted_close": 10.0,
                    "volume": 100,
                }
            },
            "filing_q": {"AAA": section_text},
        }
    }
    pickle_path = tmp_path / "legacy_env.pkl"
    with open(pickle_path, "wb") as file:
        pickle.dump(payload, file)

    prepared = prepare_finmem_trading_data(
        symbol="AAA",
        market_data_info_path=pickle_path,
        filing_payload_kind="section_text",
    )

    assert isinstance(prepared, FinsaberDataset)
    assert not isinstance(prepared, FilingSectionOverlayDataset)
    assert prepared.source_kind == "legacy_pickle"
    assert prepared.filing_payload_kind == "section_text"
    assert prepared.get_data_by_date(current_date)["filing_q"]["AAA"] == section_text


def test_prepare_finmem_trading_data_unwraps_overlay_when_sections_are_disabled():
    current_date = date(2024, 1, 2)
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
                "filing_q": {"AAA": raw_q},
            }
        }
    )
    overlay = with_filing_sections(
        loader,
        section_map={"filing_q": {"form": "10-Q", "item_key": "part_i_item_2"}},
    )

    prepared = prepare_finmem_trading_data(
        symbol="AAA",
        data_loader=overlay,
        use_filing_sections=False,
    )

    assert prepared is loader
    assert prepared.get_data_by_date(current_date)["filing_q"]["AAA"] == raw_q


def test_prepare_finmem_trading_data_rejects_invalid_payload_kind():
    loader = FinsaberDataset(data={})

    with pytest.raises(
        ValueError,
        match="Unsupported filing_payload_kind",
    ):
        prepare_finmem_trading_data(
            symbol="AAA",
            data_loader=loader,
            filing_payload_kind="mystery_payload",
        )
