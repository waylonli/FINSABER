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
            self.buy()

        self.post_next_actions()


if __name__ == '__main__':
    trade_config = {
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        "silence": True,
        "selection_strategy": "selected_5",
    }
    # trade_config = {
    #     "tickers": "all",
    #     "silence": False,
    # }
    operator = BacktestingEngine(trade_config)
    # operator.execute_iter(BuyAndHoldStrategy)
    operator.run_rolling_window(BuyAndHoldStrategy)
