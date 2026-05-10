# Examples

Examples are ordered from simplest to most research-specific. Start with package examples before running paper-specific agents.

## Custom Dataset

Run:

```bash
python examples/custom_dataset_example.py
```

This example creates an in-memory `FinsaberDataset`, runs `BuyAndHoldStrategy`, and prints total return.

Use it to learn the required dictionary shape before writing a custom loader.

## Research Experiment Launchers

These scripts are repository examples and are not included in the package wheel.

```bash
python examples/experiments/run_baselines_exp.py --setup selected_4 --include BuyAndHoldStrategy
```

```bash
python examples/experiments/run_llm_traders_exp.py \
  --setup selected_4 \
  --strategy FinMemStrategy \
  --strat_config_path strats_configs/finmem_config_normal.json
```

## Minimal Package Usage

```python
from backtest import FINSABERBt, FinsaberParquetDataset
from backtest.strategy.timing import BuyAndHoldStrategy

data = FinsaberParquetDataset("path/to/sp500_2000_2025_parquet")
config = {
    "data_loader": data,
    "tickers": ["AAPL"],
    "date_from": "2024-01-02",
    "date_to": "2024-01-10",
    "setup_name": "minimal",
    "save_results": False,
    "silence": True,
}
results = FINSABERBt(config).run_iterative_tickers(BuyAndHoldStrategy)
```

## Recommended Learning Sequence

1. Run the minimal package usage example with one ticker.
2. Enable `save_results=True` and inspect `run_summary.csv`.
3. Add `slippage_perc` and `liquidity_cap_pct`.
4. Replace `BuyAndHoldStrategy` with a simple moving-average strategy.
5. Replace `FinsaberParquetDataset` with your own `TradingData` implementation.
6. Only then run LLM or RL research launchers.
