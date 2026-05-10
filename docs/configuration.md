# Configuration

Every run is controlled by a dictionary that is converted into `TradeConfig`. Treat this config as the experiment contract: it records what data was used, what dates were tested, how orders were filled, and where outputs were written.

## Minimal Config

```python
config = {
    "data_loader": data,
    "tickers": ["AAPL"],
    "date_from": "2024-01-02",
    "date_to": "2024-03-01",
    "setup_name": "demo_aapl",
    "execution_timing": "next_open",
    "save_results": True,
    "silence": True,
}
```

This is enough to run a basic single-ticker backtest. Add costs and liquidity controls before using results for serious comparison.

## Required Decisions

| Field | Example | How to choose |
| --- | --- | --- |
| `data_loader` | `FinsaberParquetDataset(path)` | Object implementing `TradingData`. |
| `tickers` | `["AAPL", "MSFT"]` or `"all"` | Use a small list for debugging; use `"all"` only when the loader and strategy can handle the universe. |
| `date_from`, `date_to` | `"2020-01-01"` | Inclusive test range. Keep early smoke tests short. |
| `setup_name` | `"ma_cross_2020"` | Stable name used in output directories. |
| `execution_timing` | `"next_open"` | Prefer `next_open` for date-level features. |

## Capital And Cost Fields

| Field | Default | Meaning |
| --- | ---: | --- |
| `cash` | `100000.0` | Initial portfolio value. |
| `risk_free_rate` | `0.03` | Annual risk-free rate used in Sharpe and Sortino calculations. |
| `commission_per_share` | `0.0049` | Per-share commission before min/max constraints. |
| `min_commission` | `0.99` | Minimum commission per trade. |
| `max_commission_rate` | `0.01` | Maximum commission as a fraction of transaction value. |
| `slippage_perc` | `0.0` | Fixed adverse price movement per trade. |
| `slippage_impact` | `0.0` | Extra adverse price movement based on participation rate squared. |
| `liquidity_cap_pct` | `0.0` | Max order size as a fraction of average prior volume. |
| `llm_cost_as_trade_cost` | `True` | Adds tracked LLM costs into `total_trading_cost`. |

## Practical Presets

=== "Fast debugging"

    ```python
    config.update({
        "tickers": ["AAPL"],
        "date_from": "2024-01-02",
        "date_to": "2024-02-01",
        "save_results": False,
        "silence": True,
    })
    ```

=== "Research comparison"

    ```python
    config.update({
        "execution_timing": "next_open",
        "commission_per_share": 0.0049,
        "min_commission": 0.99,
        "slippage_perc": 0.0005,
        "liquidity_lookback_days": 20,
        "liquidity_min_history_days": 5,
        "liquidity_cap_pct": 0.025,
        "save_results": True,
    })
    ```

=== "LLM strategy"

    ```python
    config.update({
        "execution_timing": "next_open",
        "llm_cost_as_trade_cost": True,
        "save_results": True,
        "log_base_dir": "runs",
    })
    ```

## Rolling Windows And Selection

Rolling-window runs split the full date range into repeated evaluation periods. A selector can choose the ticker universe for each window.

```python
config.update({
    "rolling_window_size": 2,
    "rolling_window_step": 1,
    "selection_strategy": my_selector,
})
```

Selectors must use only data available inside the allowed training or prior window. Do not select tickers using returns from the future test period.

## Output Fields

| Field | Meaning |
| --- | --- |
| `save_results` | If `True`, writes CSV and JSON artifacts. |
| `log_base_dir` | Root output folder. Defaults to `backtest/output`. |
| `result_filename` | Optional pickle filename for legacy result compatibility. |
| `print_trades_table` | Enables more verbose trade printing where supported. |
| `silence` | Suppresses progress output and plots when `True`. |

## What To Save In Papers Or Reports

Always report the config fields that materially affect performance: ticker universe, date range, execution timing, adjusted price policy, commission, slippage, liquidity cap, and whether LLM cost is counted. The generated `run_config.json` is designed to make this reproducible.
