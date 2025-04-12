import backtrader as bt
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.backtest_engine import BacktestingEngine
from backtest.toolkit.operation_utils import aggregate_results_one_strategy

class SMACrossStrategy(BaseStrategy):
    params = (
        ("prior_period", 252 * 2),
        ('short_window', 10),
        ('long_window', 30),
        ('trade_size', 0.95),
        ("total_days", 0),
    )

    def __init__(self, strat_params=None):
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
            self.buy(size=self._adjust_size_for_commission(self.calculate_trade_size()))
            self.buys.append(self.data.datetime.date(0))
        elif self.crossover < 0:
            if self.position.size > 0:
                self.close()
            self.sell(size=self.calculate_trade_size())
            self.sells.append(self.data.datetime.date(0))

        self.post_next_actions()

    def calculate_trade_size(self):
        trade_cash = self.broker.cash * self.params.trade_size
        return int(trade_cash / self.data.close[0])


if __name__ == "__main__":
    trade_config = {
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        "silence": True,
        "setup_name": "selected_5",
    }
    # trade_config = {
    #     "tickers": "all",
    #     "silence": True,
    #     "setup_name": "random:100",
    # }
    # cherry_config = {
    #     "date_from": "2022-10-06",
    #     "date_to": "2023-04-10",
    #     "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
    #     "setup_name": "cherry_pick_both"
    # }
    operator = BacktestingEngine(trade_config)
    operator.run_rolling_window(SMACrossStrategy)
    aggregate_results_one_strategy(trade_config["selection_strategy"], SMACrossStrategy.__name__)

    # cherry_operator = BacktestingEngine(cherry_config)
    # cherry_operator.execute_iter(SMACrossStrategy, test_config=cherry_config)