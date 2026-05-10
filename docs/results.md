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

| Metric | Meaning |
| --- | --- |
| `final_value` | Ending portfolio value after cash, positions, and forced final liquidation where applicable. |
| `total_return` | Total percentage gain or loss over the run. |
| `annual_return` | Total return annualized with a 252-trading-day convention. |
| `annual_volatility` | Standard deviation of daily returns, annualized by `sqrt(252)`. |
| `sharpe_ratio` | Excess annualized return divided by annualized volatility. |
| `sortino_ratio` | Excess annualized return divided by downside volatility. |
| `max_drawdown` | Largest peak-to-trough equity decline. |
| `total_commission` | Sum of commission charged on executed trades. |
| `total_slippage` | Sum of adverse fill-price impact. |
| `total_llm_cost` | Sum of tracked LLM provider cost. |
| `total_trading_cost` | Commission plus slippage plus LLM cost when configured. |

!!! note "Metric units"
    `total_return`, `annual_return`, and `annual_volatility` are fractions in artifacts. For example, `0.12` means `12%`. `max_drawdown` is currently reported as a percentage-style magnitude in the Python-native engine.

## Recommended Analysis Flow

Start with `run_summary.csv` for a flat overview. Then inspect per-ticker files:

- `orders.csv`: executed Backtrader orders.
- `trades.csv`: Python-engine trades.
- `rejected_orders.csv`: insufficient liquidity, invalid price, insufficient cash, or missing future bars.
- `equity_curve.csv`: portfolio value through time.

## How To Debug A Run

| Symptom | File to inspect | Likely cause |
| --- | --- | --- |
| Return is zero | `trades.csv`, `orders.csv` | Strategy never traded or all orders were rejected. |
| Many missing tickers | `run_summary.csv` | Tickers absent from date range or loader subset. |
| Unrealistically high return | `run_config.json` | Same-close timing, no costs, or no liquidity cap. |
| Large cost drag | `trades.csv`, `llm_costs.csv` | High turnover, high slippage, or expensive LLM calls. |
| Orders not filled | `rejected_orders.csv` | Missing future bars, insufficient cash, or liquidity constraints. |

## Presentation Guidelines

For paper tables or dashboards, aggregate from `run_summary.csv` rather than scraping console logs. Console output is for progress monitoring; CSV and JSON artifacts are the stable interface for analysis.

For financial reports, show at least total return, annual return, volatility, Sharpe ratio, max drawdown, total trading cost, and the execution assumptions used to produce them.
