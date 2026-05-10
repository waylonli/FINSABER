# Execution Model

FINSABER makes execution assumptions explicit so result differences can be traced back to timing, price adjustment, and cost settings.

## One Trade From Signal To Fill

Assume the strategy sees Monday's data and decides to buy 100 shares.

| Step | What happens |
| --- | --- |
| Signal date | Strategy reads Monday data and submits `buy("AAPL", 100)`. |
| Timing rule | `next_open` delays the order until the next available trading day. |
| Base price | The engine uses Tuesday `adjusted_open`. |
| Liquidity | Quantity may be capped by prior average volume. |
| Slippage | Buy price is moved upward; sell price is moved downward. |
| Commission | Cash is reduced by trade value plus commission. |
| Portfolio update | Position, cash, trade history, and equity curve are updated. |

## Timing

```python
"execution_timing": "next_open"
```

Supported values:

- `next_open`: signal generated on date `t`, order fills at the next available adjusted open.
- `same_close`: order fills on the same adjusted close.

Prefer `next_open` when features are date-level and exact intraday availability is unknown.

| Timing | When to use | Main risk |
| --- | --- | --- |
| `next_open` | Daily bars, news, filings, LLM summaries, uncertain timestamps. | May be less optimistic because the market can move overnight. |
| `same_close` | Features known before close, such as intraday signals or pre-close data. | Easy to misuse with after-close or date-only information. |

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

Example: if a split makes raw close `50` while adjusted close is `25`, the adjustment factor is `0.5`. A raw open of `52` becomes an adjusted open of `26`.

## Commission

```python
"commission_per_share": 0.0049,
"min_commission": 0.99,
"max_commission_rate": 0.01,
```

Commission is bounded by per-share, minimum commission, and maximum transaction-rate settings.

The commission calculation is:

```text
commission = min(
    max(abs(quantity) * commission_per_share, min_commission),
    abs(quantity) * price * max_commission_rate
)
```

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

Example with `price=100`, `slippage_perc=0.0005`, and no impact term:

```text
buy fill  = 100 * (1 + 0.0005) = 100.05
sell fill = 100 * (1 - 0.0005) = 99.95
```

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

Example: if the prior 20-day average volume is `1,000,000` shares and `liquidity_cap_pct=0.025`, the largest order is `25,000` shares. A request for `100,000` shares is reduced to `25,000`.

## LLM Cost

LLM costs can be recorded and included in trading cost:

```python
"llm_cost_as_trade_cost": True
```

Use:

```python
from finsaber.toolkit.llm_cost_monitor import add_openai_cost_from_response

add_openai_cost_from_response(response)
```

The result artifact `llm_costs.csv` stores the cost ledger when present.

LLM cost is deliberately tracked through a small framework utility rather than by parsing every external agent implementation. External strategies should call the monitor around provider requests so model, token usage, provider, and metadata are saved with the run.

## Rejected Orders

Orders can be rejected or reduced. This is expected behavior, not necessarily an engine failure.

| Reason | Meaning |
| --- | --- |
| `invalid_price` | No usable execution price was available. |
| `insufficient_liquidity_history` | Liquidity cap is enabled but not enough prior volume exists. |
| `zero_quantity` | Requested size becomes zero after holdings, cash, or liquidity checks. |
| `insufficient_cash` | The account cannot afford even one share after costs. |
| `insufficient_holdings` | A sell order exceeds current position. |
| `no_future_bar` | A pending `next_open` order has no later bar to execute on. |

Inspect `rejected_orders.csv` before trusting a backtest. A strategy with many rejected orders may be testing an unrealistic sizing rule.
