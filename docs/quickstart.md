# Quick Start

This page walks through the shortest complete workflow: load data, define a run configuration, execute a strategy, and inspect output files.

## What You Will Run

The example buys and holds `AAPL` over a short date range. It is intentionally simple because the first goal is to confirm the framework, data loader, execution model, and result writer are working.

!!! tip "Start small"
    Use one ticker and a short date range first. After the run succeeds, expand to more tickers, longer periods, selectors, or LLM strategies.

## Step 1: Load Data

```python
from finsaber import FinsaberParquetDataset

data = FinsaberParquetDataset("/path/to/sp500_2000_2025_parquet")
```

The loader reads daily prices plus optional news and filings. It returns data through the common `TradingData` interface, so the engine does not need to know whether the source is parquet, memory, or a custom database.

## Step 2: Define The Run

```python
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
```

The important financial choice is `execution_timing="next_open"`. The strategy observes data for day `t`, submits an order, and the order fills at the next available adjusted open. This is usually safer than same-day close execution for date-level text data.

## Step 3: Run A Strategy

## Buy And Hold On Parquet Data

=== "Backtrader engine"

    ```python
    from finsaber import FINSABERBt, FinsaberParquetDataset
    from finsaber.strategy.timing import BuyAndHoldStrategy

    data = FinsaberParquetDataset("/path/to/sp500_2000_2025_parquet")

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

=== "Python-native engine"

    ```python
    from finsaber import FINSABER, FinsaberParquetDataset
    from finsaber.strategy.timing_llm import BaseStrategyIso

    class BuyOnce(BaseStrategyIso):
        def on_data(self, date, today_data, framework):
            bar = today_data["price"]["AAPL"]
            if "AAPL" not in framework.portfolio:
                framework.buy(date, "AAPL", bar["adjusted_close"], -1)

    data = FinsaberParquetDataset("/path/to/sp500_2000_2025_parquet")
    config = {
        "data_loader": data,
        "tickers": ["AAPL"],
        "date_from": "2024-01-02",
        "date_to": "2024-01-10",
        "setup_name": "buy_once_aapl",
        "execution_timing": "next_open",
        "save_results": True,
        "silence": True,
    }

    results = FINSABER(config).run_iterative_tickers(BuyOnce)
    ```

This run uses adjusted prices for fills and valuation, applies commission, caps orders to `2.5%` of prior average volume, and writes metrics and CSV artifacts under `backtest/output/buy_hold_aapl/BuyAndHoldStrategy/`.

## Step 4: Read The Result

```python
window = "2024-01-02_2024-01-10"
metrics = results[window]["AAPL"]

print(metrics["final_value"])
print(metrics["total_return"])
print(metrics["sharpe_ratio"])
```

If `save_results=True`, inspect:

```text
backtest/output/buy_hold_aapl/BuyAndHoldStrategy/
  run_config.json
  run_summary.csv
  2024-01-02_2024-01-10/AAPL/metrics.json
  2024-01-02_2024-01-10/AAPL/equity_curve.csv
  2024-01-02_2024-01-10/AAPL/orders.csv
```

Use `run_summary.csv` for comparisons across many tickers. Use `metrics.json` for one ticker. Use `equity_curve.csv` to plot portfolio value over time.

## Configuration Checklist

Every run should specify:

- `data_loader`: a `TradingData` implementation.
- `tickers`: a ticker list or `"all"`.
- `date_from` and `date_to`: inclusive backtest dates.
- `execution_timing`: usually `"next_open"` for date-level features.
- `setup_name`: a stable run name for output paths.
- Cost controls such as `commission_per_share`, `slippage_perc`, and `liquidity_cap_pct`.

See [Configuration](configuration.md) for the full field guide.

## In-Memory Dataset

For small experiments, wrap a date-keyed dictionary with `FinsaberDataset`.

```python
from datetime import date
from finsaber import FINSABERBt, FinsaberDataset
from finsaber.strategy.timing import BuyAndHoldStrategy

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

Use this format for unit tests, toy examples, or custom pipelines that already produce Python dictionaries. Use `FinsaberParquetDataset` for large historical runs.

See `examples/custom_dataset_example.py` for a runnable custom data example.

## Result Location

By default, results are written under:

```text
backtest/output/<setup_name>/<strategy_name>/
```

Set `log_base_dir` in the config to redirect output.
