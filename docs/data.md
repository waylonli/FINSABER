# Data

## Data Interface

All loaders should implement `TradingData`. Engines depend on this interface instead of a specific storage format, so users can plug in local parquet, database-backed loaders, or enriched datasets with extra modalities.

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

The contract is date-first. A loader returns all available modalities for one date, and each modality is keyed by ticker where applicable. This keeps strategies from accidentally scanning future rows.

## Parquet Layout

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

The loader lazily reads the selected date range and ticker universe. When `tickers="all"`, the ticker list is inferred from `price_daily`. The default price field is `adjusted_close`, but execution can use `adjusted_open`, `adjusted_high`, and `adjusted_low` when timing requires them.

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

## Implementing A Custom Loader

```python
from backtest import TradingData

class MyDataset(TradingData):
    def __init__(self, connection, tickers):
        self.connection = connection
        self.tickers = tickers

    def get_data_by_date(self, date):
        return {
            "price": self._load_prices(date),
            "earnings_call": self._load_calls(date),
        }

    def get_tickers_list(self):
        return list(self.tickers)
```

Implement the remaining abstract methods by filtering the same underlying source. If a modality is unavailable for a date, return an empty dictionary rather than leaking data from another date.

## Avoiding Look-Ahead Bias

Date-only news and filings should be considered unavailable until the next decision point unless timestamps prove same-day availability. For strict backtests, prefer `execution_timing="next_open"`.
