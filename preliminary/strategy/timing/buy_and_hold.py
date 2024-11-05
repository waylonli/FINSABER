from preliminary.strategy.timing.base_strategy import BaseStrategy
from preliminary.backtest_engine import BacktestingEngine

# Create a Strategy
class BuyAndHoldStrategy(BaseStrategy):
    params = (
        ("total_days", 0),
    )

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close

    def next(self):
        if not self.position:
            max_size = self._adjust_size_for_commission(int(self.broker.cash / self.dataclose[0]))
            self.buy(size=max_size)
        self.post_next_actions()


if __name__ == '__main__':
    # trade_config = {
    #     "tickers": ["MSFT"],
    #     "silence": False,
    #     "date_from": "2016-01-01",
    #     "date_to": "2018-01-01",
    #     "selection_strategy": "debug",
    # }

    # trade_config = {
    #     "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
    #     "silence": False,
    #     "selection_strategy": "selected_5",
    # }
    trade_config = {
        "tickers": "all",
        "silence": True,
    }
    operator = BacktestingEngine(trade_config)
    # operator.execute_iter(BuyAndHoldStrategy)
    operator.run_rolling_window(BuyAndHoldStrategy)
