import backtrader as bt
from preliminary.strategy.base_strategy import BaseStrategy
from preliminary.backtest_engine import BacktestingEngine

# Create a Strategy
class BuyAndHoldStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close

    def next(self):
        if not self.position:
            self.buy()

        self.post_next_actions()


if __name__ == '__main__':
    trade_config = {
        "tickers": ["AAPL"],
    }
    operator = BacktestingEngine(trade_config)
    operator.execute_iter(BuyAndHoldStrategy)
