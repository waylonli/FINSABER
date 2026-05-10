# Strategies

## Backtrader Strategies

Backtrader strategies run through `FINSABERBt`. Subclass `BaseStrategy` and implement `next()`.

```python
import backtrader as bt
from backtest.strategy.timing.base_strategy import BaseStrategy

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

## LLM-Style Strategies

LLM-style strategies run through `FINSABER`. They receive all data for a date and trade through the framework object.

```python
from backtest.strategy.timing_llm import BaseStrategyIso

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
from backtest import FINSABER

results = FINSABER(config).run_iterative_tickers(
    RuleBasedAgent,
    strat_params={"symbol": "$symbol"},
)
```

## Selector Strategies

Selection strategies choose tickers for rolling-window runs.

```python
from backtest.strategy.selection import BaseSelector

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
