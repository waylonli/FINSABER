# Strategies

FINSABER supports two strategy styles: Backtrader strategies for conventional event-driven backtests, and Python-native strategies for agents that operate directly on date-level dictionaries.

## Strategy Responsibilities

A strategy should answer only one question: "Given the information available now, what order should I request?"

The framework handles:

- Fill timing.
- Adjusted execution prices.
- Cash and position updates.
- Commission, slippage, liquidity caps, and LLM costs.
- Result artifacts and metrics.

Do not manually edit cash, positions, or result files inside a strategy.

## Decision Checklist

Before submitting an order, a strategy should know:

| Question | Example |
| --- | --- |
| What ticker am I trading? | `AAPL` |
| What data is visible today? | today's price bar, news list, filing text |
| What is the signal? | buy, sell, hold |
| What size do I want? | fixed shares, all-in, target weight |
| What timing will the framework apply? | `next_open` or `same_close` |

## Backtrader Strategies

Backtrader strategies run through `FINSABERBt`. Subclass `BaseStrategy` and implement `next()`.

```python
import backtrader as bt
from finsaber.strategy.timing.base_strategy import BaseStrategy

class MovingAverageCross(BaseStrategy):
    params = (
        ("fast", 20),
        ("slow", 60),
        ("prior_period", 252),
    )

    def __init__(self):
        super().__init__()
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow)

    def next(self):
        if self.fast_ma[0] > self.slow_ma[0] and not self.position:
            size = self._adjust_size_for_commission(int(self.broker.cash / self.data.close[0]))
            if size > 0:
                self.buy(size=size)
        elif self.fast_ma[0] < self.slow_ma[0] and self.position:
            self.close()
        self.post_next_actions()
```

Run:

```python
results = FINSABERBt(config).run_iterative_tickers(MovingAverageCross)
```

Backtrader strategies should call `self.post_next_actions()` at the end of `next()` so framework-level bookkeeping stays consistent.

Use Backtrader strategies when you need built-in indicators, Backtrader analyzers, or an event loop that resembles existing Backtrader code.

## LLM-Style Strategies

LLM-style strategies run through `FINSABER`. They receive all data for a date and trade through the framework object.

```python
from finsaber.strategy.timing_llm import BaseStrategyIso

class RuleBasedAgent(BaseStrategyIso):
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def on_data(self, date, today_data, framework):
        bar = today_data["price"][self.symbol]
        news = today_data.get("news", {}).get(self.symbol, [])
        text = " ".join(news).lower()
        if "upgrade" in text:
            framework.buy(date, self.symbol, bar["adjusted_close"], -1)
        elif self.symbol in framework.portfolio:
            framework.sell(date, self.symbol, bar["adjusted_close"], -1)
```

Run:

```python
from finsaber import FINSABER

results = FINSABER(config).run_iterative_tickers(
    RuleBasedAgent,
    strat_params={"symbol": "$symbol"},
)
```

Use `framework.buy(date, ticker, reference_price, quantity)` and `framework.sell(...)` instead of mutating cash or positions directly. Pass `quantity=-1` for all-in or full-exit behavior; the execution framework will still apply costs, liquidity, and cash checks.

The `reference_price` is stored for audit and used as a fallback if the execution bar is unavailable. Under normal `next_open` execution, the actual fill uses the next adjusted open from the data loader.

## Selector Strategies

Selection strategies choose tickers for rolling-window runs.

```python
from finsaber.strategy.selection import BaseSelector

class TopVolumeSelector(BaseSelector):
    def __init__(self, top_k=5):
        self.top_k = top_k

    def select(self, data_loader, start_date, end_date):
        window = data_loader.get_subset_by_time_range(start_date, end_date)
        avg_volume = {}
        for date in window.get_date_range():
            for ticker, bar in window.get_data_by_date(date)["price"].items():
                avg_volume.setdefault(ticker, []).append(bar["volume"])
        return sorted(
            avg_volume,
            key=lambda ticker: sum(avg_volume[ticker]) / len(avg_volume[ticker]),
            reverse=True,
        )[: self.top_k]
```

Pass it in config:

```python
config["selection_strategy"] = TopVolumeSelector(top_k=10)
```

Selectors should use only data inside the allowed training window. Avoid computing ranks from the full backtest period because that introduces universe-selection look-ahead bias.

## Strategy Testing Pattern

1. Test one ticker over a few weeks with `silence=False`.
2. Confirm trades appear in `orders.csv` or `trades.csv`.
3. Check `rejected_orders.csv`.
4. Expand to a longer date range.
5. Expand to multiple tickers.
6. Only then add LLM calls or expensive feature construction.
