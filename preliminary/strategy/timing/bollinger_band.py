# File: preliminary/strategy/timing/bollinger_bands_strategy.py

import backtrader as bt
from preliminary.strategy.timing.base_strategy import BaseStrategy
from preliminary.backtest_engine import BacktestingEngine


class BollingerBandsStrategy(BaseStrategy):
    params = (
        ("period", 20),
        ("devfactor", 2.0),
        ("total_days", 0),
    )

    def __init__(self):
        super().__init__()
        self.bbands = {}

        for d in self.datas:
            self.bbands[d] = bt.indicators.BollingerBands(d.close, period=self.params.period,
                                                          devfactor=self.params.devfactor)

    def next(self):
        for d in self.datas:
            if d.close[0] < self.bbands[d].lines.bot[0] and d.close[-1] >= self.bbands[d].lines.bot[-1]:
                self.buy(data=d, size=self._adjust_size_for_commission(int(self.broker.cash / d.close[0])))
            elif d.close[0] > self.bbands[d].lines.top[0] and d.close[-1] <= self.bbands[d].lines.top[-1]:
                self.sell(data=d, size=self.getposition(d).size)
        self.post_next_actions()


if __name__ == "__main__":
    # trade_config = {
    #     "tickers": "all",
    #     "silence": True,
    #     "selection_strategy": "random:50",
    # }
    trade_config = {
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        "silence": False,
        "selection_strategy": "selected_5",
    }
    operator = BacktestingEngine(trade_config)
    operator.run_rolling_window(BollingerBandsStrategy)