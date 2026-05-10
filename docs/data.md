# Data

Data is the most important part of a backtest. A good strategy can look bad on broken data, and a bad strategy can look good if future information leaks into the past.

## Mental Model

FINSABER expects data to be organized by date:

```text
date
  price
    ticker -> OHLCV bar
  news
    ticker -> list of articles
  filing_k
    ticker -> annual filing text
  filing_q
    ticker -> quarterly filing text
  optional extra modality
    ticker -> payload
```

The engine asks the loader: "What could the strategy know on this date?" The strategy should not query future dates directly.

## Data Interface

All loaders should implement `TradingData`. Engines depend on this interface instead of a specific storage format, so users can plug in local parquet, database-backed loaders, or enriched datasets with extra modalities.

```python
from finsaber import TradingData

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

## Price Field Requirements

The minimum useful daily bar is:

| Field | Required | Use |
| --- | --- | --- |
| `open` | Yes | Raw open. Used to derive adjusted open. |
| `high` | Yes | Raw high. Used to derive adjusted high. |
| `low` | Yes | Raw low. Used to derive adjusted low. |
| `close` | Yes | Raw close. Used with adjusted close to compute adjustment factor. |
| `adjusted_close` | Strongly recommended | Split/dividend-adjusted close. |
| `volume` | Strongly recommended | Raw share volume for liquidity caps. |
| `cik` | Optional | Useful for SEC filing alignment. |

If `adjusted_close` is missing, raw prices may create false jumps around stock splits. For serious historical equity tests, prefer adjusted OHLC for price simulation and raw volume for liquidity.

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

## Alignment Policy

Price, news, and filings do not always align perfectly. The loader should preserve what is known and avoid guessing aggressively.

| Situation | Recommended behavior |
| --- | --- |
| Price exists, no news | Return an empty or missing `news` entry for that ticker/date. |
| News exists, no price | Keep it for feature construction only if the strategy can handle missing tradable price. |
| Filing appears under ticker alias | Deduplicate by accession or CIK before feature construction. |
| Date-only text timestamp | Treat as available from the next decision point unless timestamps prove otherwise. |
| Delisted ticker has no future price | Let the engine skip or reject fills with missing future bars. |

## Implementing A Custom Loader

```python
from finsaber import TradingData

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

## Custom Dataset Checklist

- Implement all `TradingData` methods.
- Return Python `date` keys or normalize date strings consistently.
- Keep ticker symbols consistent across price, news, and filings.
- Include `adjusted_close` when possible.
- Keep raw `volume` unadjusted for liquidity calculations.
- Make `get_subset_by_time_range` and `get_ticker_subset_by_time_range` return a smaller loader, not a future-aware object.
- Add tests with a tiny in-memory dataset before using a full private dataset.

## Avoiding Look-Ahead Bias

Date-only news and filings should be considered unavailable until the next decision point unless timestamps prove same-day availability. For strict backtests, prefer `execution_timing="next_open"`.
