# Data

## Data Interface

All data loaders should implement `TradingData`.

```python
from backtest import TradingData

class MyData(TradingData):
    ...
```

Required methods:

- `get_data_by_date(date)`
- `get_ticker_price_by_date(ticker, date, price_field=None)`
- `get_ticker_data_by_date(ticker, date)`
- `get_tickers_list()`
- `get_subset_by_time_range(start_date, end_date)`
- `get_ticker_subset_by_time_range(ticker, start_date, end_date)`
- `get_date_range()`

## FINSABER-2 Parquet Layout

`FinsaberParquetDataset` expects:

```text
sp500_2000_2025_parquet/
  price_daily/year=YYYY/part-000.parquet
  news_items/year=YYYY/part-000.parquet
  filingk/year=YYYY/part-000.parquet
  filingq/year=YYYY/part-000.parquet
```

Price rows should include:

```text
date, symbol, cik, open, high, low, close, adjusted_close, volume
```

The loader computes:

```text
adjusted_open = open * adjusted_close / close
adjusted_high = high * adjusted_close / close
adjusted_low = low * adjusted_close / close
```

Raw `volume` is retained for liquidity caps.

## In-Memory Format

`FinsaberDataset` accepts:

```python
{
    date: {
        "price": {
            "AAPL": {
                "open": 187.15,
                "high": 188.44,
                "low": 183.89,
                "close": 185.64,
                "adjusted_close": 185.30,
                "volume": 82488700,
            }
        },
        "news": {"AAPL": ["..."]},
        "filing_k": {"AAPL": "..."},
        "filing_q": {"AAPL": "..."},
    }
}
```

You may add extra modalities such as:

- `earnings_call`
- `analyst_report`
- `macro`
- `alternative_data`

Keep the daily shape consistent: modality name, ticker key, payload.

## Avoiding Look-Ahead Bias

Date-only news and filings should be considered unavailable until the next decision point unless timestamps prove same-day availability. For strict backtests, prefer `execution_timing="next_open"`.
