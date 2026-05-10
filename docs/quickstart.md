# Quick Start

## Buy And Hold On FINSABER-2 Parquet Data

```python
from backtest import FINSABERBt, FinsaberParquetDataset
from backtest.strategy.timing import BuyAndHoldStrategy

data = FinsaberParquetDataset(r"I:\Data\finsaber2\sp500_2000_2025_parquet")

config = {
    "data_loader": data,
    "tickers": ["AAPL"],
    "date_from": "2024-01-02",
    "date_to": "2024-01-10",
    "setup_name": "buy_hold_aapl",
    "cash": 100_000,
    "execution_timing": "next_open",
    "commission_per_share": 0.0049,
    "min_commission": 0.99,
    "slippage_perc": 0.0005,
    "liquidity_cap_pct": 0.025,
    "save_results": True,
    "silence": True,
}

results = FINSABERBt(config).run_iterative_tickers(BuyAndHoldStrategy)
print(results["AAPL"]["total_return"])
```

## In-Memory Dataset

For small experiments, wrap a date-keyed dictionary with `FinsaberDataset`.

```python
from datetime import date
from backtest import FINSABERBt, FinsaberDataset
from backtest.strategy.timing import BuyAndHoldStrategy

loader = FinsaberDataset(data={
    date(2024, 1, 2): {
        "price": {
            "DEMO": {
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "adjusted_close": 101.0,
                "volume": 1_000_000,
            }
        }
    }
})
```

See `examples/custom_dataset_example.py` for a runnable custom data example.

## Result Location

By default, results are written under:

```text
backtest/output/<setup_name>/<strategy_name>/
```

Set `log_base_dir` in the config to redirect output.
