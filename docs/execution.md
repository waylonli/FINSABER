# Execution Model

FINSABER-2 makes execution assumptions explicit.

## Timing

```python
"execution_timing": "next_open"
```

Supported values:

- `next_open`: signal generated on date `t`, order fills at the next available adjusted open.
- `same_close`: order fills on the same adjusted close.

Prefer `next_open` when features are date-level and exact intraday availability is unknown.

## Adjusted OHLC

For split-adjusted simulation:

```text
adjusted_open = open * adjusted_close / close
adjusted_high = high * adjusted_close / close
adjusted_low = low * adjusted_close / close
```

Raw OHLC can produce false price jumps around splits. Use adjusted OHLC for portfolio valuation and execution prices.

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
