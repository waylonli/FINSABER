# FINSABER-2

FINSABER-2 is the package-oriented backtesting framework for evaluating financial trading strategies over price, news, filings, and extensible market data. It is designed to support both traditional Backtrader strategies and LLM-style decision loops while keeping execution assumptions explicit.

## What Is Included

- A reusable Python package, `finsaber-backtest`, containing core backtesting code only.
- Pluggable data interfaces through `TradingData`.
- Built-in loaders for FINSABER dictionary data and FINSABER-2 partitioned parquet data.
- Explicit execution timing with `next_open` and `same_close`.
- Adjusted OHLC handling for split-adjusted simulation.
- Commission, slippage, liquidity-cap, and LLM-cost accounting.
- Structured result artifacts for metrics, trades, equity curves, rejected orders, and LLM costs.

## Package Boundary

The wheel intentionally excludes paper-specific agent and RL implementations:

- `llm_traders/`
- `rl_traders/`
- experiment runner scripts
- generated outputs and private datasets

Those integrations remain available in the repository for research experiments, but the package focuses on the reusable backtesting framework.

## Typical Workflow

```python
from backtest import FINSABERBt, FinsaberParquetDataset
from backtest.strategy.timing import BuyAndHoldStrategy

data = FinsaberParquetDataset(r"I:\Data\finsaber2\sp500_2000_2025_parquet")

config = {
    "data_loader": data,
    "tickers": ["AAPL"],
    "date_from": "2024-01-02",
    "date_to": "2024-01-10",
    "setup_name": "demo",
    "execution_timing": "next_open",
    "save_results": True,
    "silence": True,
}

results = FINSABERBt(config).run_iterative_tickers(BuyAndHoldStrategy)
print(results["AAPL"]["total_return"])
```

## When To Use Each Engine

Use `FINSABERBt` for Backtrader-compatible timing strategies and baseline technical strategies.

Use `FINSABER` for Python-native or LLM-style strategies that consume date-level data and submit orders through the framework object.
