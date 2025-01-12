from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.backtest_engine import BacktestingEngine

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

    trade_config = {
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        "silence": False,
        "selection_strategy": "selected_5",
    }
    # trade_config = {
    #     "tickers": "all",
    #     "silence": True,
    #     "selection_strategy": "random:50",
    # }
    operator = BacktestingEngine(trade_config)
    # operator.execute_iter(BuyAndHoldStrategy)
    # operator.run_rolling_window(BuyAndHoldStrategy)
    print(operator.execute_iter(BuyAndHoldStrategy, test_config={
        "date_from": "2022-10-06",
        "date_to": "2023-04-10",
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        "selection_strategy": "cherry_pick_debug"
    }))