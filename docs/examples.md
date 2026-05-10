# Examples

## Custom Dataset

Run:

```bash
python examples/custom_dataset_example.py
```

This example creates an in-memory `FinsaberDataset`, runs `BuyAndHoldStrategy`, and prints total return.

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
