from datetime import date

import pandas as pd
import pytest

from backtest.data_util import (
    FinsaberDataset,
    FinsaberParquetDataset,
    create_finsaber2_data_loader,
    resolve_trading_data,
    trading_data_to_env_dict,
)


def test_finsaber_dataset_adjusts_ohlc_and_filters_modalities():
    data = {
        date(2024, 1, 2): {
            "price": {
                "AAA": {
                    "open": 50.0,
                    "high": 110.0,
                    "low": 40.0,
                    "close": 100.0,
                    "adjusted_close": 200.0,
                    "volume": 1000,
                },
                "BBB": {"close": 10.0, "adjusted_close": 10.0, "volume": 500},
            },
            "news": {"AAA": ["known before signal"], "BBB": ["other"]},
        }
    }
    loader = FinsaberDataset(data=data)

    assert loader.get_ticker_price_by_date("AAA", date(2024, 1, 2), "adjusted_open") == 100.0
    assert loader.get_tickers_list() == ["AAA", "BBB"]

    subset = loader.get_ticker_subset_by_time_range("AAA", "2024-01-01", "2024-01-03")
    assert subset.get_data_by_date("2024-01-02")["news"] == {"AAA": ["known before signal"]}

    frame = loader.get_price_dataframe(tickers=["AAA"], adjust=True)
    row = frame.iloc[0]
    assert row["open"] == 100.0
    assert row["high"] == 220.0
    assert row["low"] == 80.0
    assert row["close"] == 200.0


def test_finsaber_parquet_dataset_reads_price_partitions_and_adjusts(tmp_path):
    price_dir = tmp_path / "price_daily" / "year=2024"
    price_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "symbol": "AAA",
                "cik": "0001",
                "open": 50.0,
                "high": 110.0,
                "low": 40.0,
                "close": 100.0,
                "adjusted_close": 200.0,
                "volume": 1000,
            }
        ]
    ).to_parquet(price_dir / "part-000.parquet", index=False)

    loader = FinsaberParquetDataset(tmp_path, tickers=["AAA"], modalities=("price",))

    assert loader.get_date_range() == [date(2024, 1, 2)]
    assert loader.get_tickers_list() == ["AAA"]
    bar = loader.get_data_by_date("2024-01-02")["price"]["AAA"]
    assert bar["adjusted_open"] == pytest.approx(100.0)

    frame = loader.get_price_dataframe(tickers=["AAA"], adjust=True)
    assert frame.iloc[0]["open"] == pytest.approx(100.0)
    assert frame.iloc[0]["close"] == pytest.approx(200.0)


def test_dataset_factory_materializes_agent_env_dict(tmp_path):
    price_dir = tmp_path / "price_daily" / "year=2024"
    price_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
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

    loader = create_finsaber2_data_loader(tmp_path, tickers=["AAA"], modalities=("price",))
    resolved = resolve_trading_data(data_loader=loader)
    env_data = trading_data_to_env_dict(resolved, start_date="2024-01-01", end_date="2024-01-03", tickers=["AAA"])

    assert list(env_data) == [date(2024, 1, 2)]
    assert env_data[date(2024, 1, 2)]["price"]["AAA"]["adjusted_open"] == pytest.approx(20.0)
