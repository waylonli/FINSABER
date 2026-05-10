# Results

When `save_results=True`, FINSABER writes structured artifacts under:

```text
backtest/output/<setup_name>/<strategy_name>/
```

Configure a different root with:

```python
"log_base_dir": "runs"
```

## Artifact Schema

```text
run_config.json
run_manifest.json
run_summary.csv
<window>/<ticker>/metrics.json
<window>/<ticker>/equity_curve.csv
<window>/<ticker>/trades.csv
<window>/<ticker>/orders.csv
<window>/<ticker>/rejected_orders.csv
<window>/<ticker>/llm_costs.csv
```

Only files with available data are written.

`run_config.json` captures the resolved `TradeConfig`. `run_manifest.json` records generated artifacts and helps downstream scripts discover output files without hard-coding paths. `run_summary.csv` is the first file to inspect for comparing tickers, windows, and strategies.

## Important Metrics

- `total_return`
- `annual_return`
- `annual_volatility`
- `sharpe_ratio`
- `sortino_ratio`
- `max_drawdown`
- `total_commission`
- `total_slippage`
- `total_llm_cost`
- `total_trading_cost`

## Recommended Analysis Flow

Start with `run_summary.csv` for a flat overview. Then inspect per-ticker files:

- `orders.csv`: executed Backtrader orders.
- `trades.csv`: Python-engine trades.
- `rejected_orders.csv`: insufficient liquidity, invalid price, insufficient cash, or missing future bars.
- `equity_curve.csv`: portfolio value through time.

## Presentation Guidelines

For paper tables or dashboards, aggregate from `run_summary.csv` rather than scraping console logs. Console output is for progress monitoring; CSV and JSON artifacts are the stable interface for analysis.
