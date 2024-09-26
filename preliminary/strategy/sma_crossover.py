import os

import pandas as pd
import backtrader as bt
from preliminary.strategy.base_strategy import BaseStrategy
from preliminary.backtest_engine import BacktestingEngine


class SMACrossStrategy(BaseStrategy):
    params = (
        ('short_window', 5),
        ('long_window', 20),
        ('trade_size', 0.01),
    )

    def __init__(self):
        super().__init__()
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_window)
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_window)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def next(self):
        if self.crossover > 0:
            if self.position.size < 0:
                self.close()
            self.buy(size=self.calculate_trade_size())
            self.buys.append(self.data.datetime.date(0))
        elif self.crossover < 0:
            if self.position.size > 0:
                self.close()
            self.sell(size=self.calculate_trade_size())
            self.sells.append(self.data.datetime.date(0))

        self.post_next_actions()

    def calculate_trade_size(self):
        trade_cash = self.broker.getvalue() * self.params.trade_size
        return self.broker.getposition(self.data).size or int(trade_cash / self.data.close[0])


if __name__ == "__main__":
    trade_config = {
        "tickers": ["AAPL"],
    }
    operator = BacktestingEngine(trade_config)
    operator.execute_iter(SMACrossStrategy)
