# Execution Model

FINSABER makes execution assumptions explicit so result differences can be traced back to timing, price adjustment, and cost settings.

## Timing

```python
"execution_timing": "next_open"
```

Supported values:

- `next_open`: signal generated on date `t`, order fills at the next available adjusted open.
- `same_close`: order fills on the same adjusted close.

Prefer `next_open` when features are date-level and exact intraday availability is unknown.

## Order Lifecycle

For Python-native strategies, the execution path is:

1. Strategy receives `today_data` for date `t`.
2. Strategy calls `framework.buy(...)` or `framework.sell(...)`.
3. The framework resolves the fill date and fill price according to `execution_timing`.
4. Liquidity cap reduces or rejects the requested quantity.
5. Slippage and commission are applied.
6. Cash, position, trade history, and rejection logs are updated.

For Backtrader strategies, the same assumptions are mapped into the Backtrader broker and custom sizers/observers where possible.

## Adjusted OHLC

For split-adjusted simulation:

```text
adjusted_open = open * adjusted_close / close
adjusted_high = high * adjusted_close / close
adjusted_low = low * adjusted_close / close
```

Raw OHLC can produce false price jumps around splits. Use adjusted OHLC for portfolio valuation and execution prices.

If only `close` and `adjusted_close` are available, the loader derives adjusted open/high/low by multiplying raw OHLC by `adjusted_close / close`. This keeps split adjustments internally consistent while preserving raw volume.

## Commission

```python
"commission_per_share": 0.0049,
"min_commission": 0.99,
"max_commission_rate": 0.01,
```

Commission is bounded by per-share, minimum commission, and maximum transaction-rate settings.

## Slippage

```python
"slippage_perc": 0.0005,
"slippage_impact": 0.0,
```

The Python engine applies:

```text
slippage_rate = slippage_perc + slippage_impact * participation_rate^2
```

Buy fills are worsened upward; sell fills are worsened downward.

## Liquidity Cap

```python
"liquidity_lookback_days": 20,
"liquidity_min_history_days": 1,
"liquidity_cap_pct": 0.025,
```

Orders are capped to:

```text
floor(average_prior_volume * liquidity_cap_pct)
```

Volume history uses prior bars only. If a cap is enabled but insufficient prior volume history exists, the Python engine rejects the order instead of silently filling it uncapped.

This cap is a participation-rate control, not a full market-impact model. For large-order simulations, combine it with nonzero `slippage_impact`.

## LLM Cost

LLM costs can be recorded and included in trading cost:

```python
"llm_cost_as_trade_cost": True
```

Use:

```python
from backtest.toolkit.llm_cost_monitor import add_openai_cost_from_response

add_openai_cost_from_response(response)
```

The result artifact `llm_costs.csv` stores the cost ledger when present.

LLM cost is deliberately tracked through a small framework utility rather than by parsing every external agent implementation. External strategies should call the monitor around provider requests so model, token usage, provider, and metadata are saved with the run.
