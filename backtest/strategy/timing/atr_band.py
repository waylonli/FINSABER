# File: preliminary/strategy/timing/atr_band_strategy.py

import backtrader as bt
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.backtest_engine import BacktestingEngine


class ATRBandStrategy(BaseStrategy):
    params = (
        ("prior_period", 252 * 2),  # Train on the past 3 years of daily data
        ("atr_period", 14),
        ("multiplier", 1.5),
        ("total_days", 0),
    )

    def __init__(self, strat_params=None):
        super().__init__()
        self.atr = {}
        self.upper_band = {}
        self.lower_band = {}
        self.sma = {}

        for d in self.datas:
            self.atr[d] = bt.indicators.ATR(d, period=self.params.atr_period)
            self.sma[d] = bt.indicators.SimpleMovingAverage(d.close, period=self.params.atr_period)
            self.upper_band[d] = self.sma[d] + self.atr[d] * self.params.multiplier
            self.lower_band[d] = self.sma[d] - self.atr[d] * self.params.multiplier

    def next(self):
        for d in self.datas:
            if d.low[0] <= self.lower_band[d][0] and d.low[-1] > self.lower_band[d][-1]:
                self.buy(data=d, size=self._adjust_size_for_commission(int(self.broker.cash / d.close[0])))
            elif d.high[0] >= self.upper_band[d][0] and d.high[-1] < self.upper_band[d][-1]:
                self.sell(data=d, size=self.getposition(d).size)

        self.post_next_actions()

if __name__ == "__main__":
    # trade_config = {
    #     "tickers": "all",
    #     "silence": True,
    #     "setup_name": "random:50",
    # }

    trade_config = {
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        # "tickers": ["MSFT"],
        "silence": False,
        "setup_name": "selected_5",
        "silence": True,
    }
    # operator = BacktestingEngine(trade_config)
    # operator.run_rolling_window(ATRBandStrategy)
    from backtest.toolkit.operation_utils import aggregate_results_one_strategy
    aggregate_results_one_strategy("selected_5", "ATRBandStrategy")