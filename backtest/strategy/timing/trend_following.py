import backtrader as bt

from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.backtest_engine import BacktestingEngine
from backtest.toolkit.operation_utils import aggregate_results_one_strategy

class TrendFollowingStrategy(BaseStrategy):
    params = (
        ("prior_period", 252 * 2),
        ("atr_period", 10),
        ("period", 30),
        ("leverage", 0.95),  # To avoid full investment
        ("total_days", 0),
    )

    def __init__(self, strat_params=None):
        super().__init__()
        self.log_data = []  # Store portfolio values for logging
        self.order_list = []  # Store order details for logging
        self.highest = {}
        self.atr = {}
        self.stdev = {}
        self.max_close = {}

        for d in self.datas:
            # Set up indicators for each data feed
            self.highest[d] = bt.indicators.Highest(d.close, period=self.params.period)
            self.atr[d] = bt.indicators.ATR(d, period=self.params.atr_period)
            self.stdev[d] = bt.indicators.StdDev(d.close, period=self.params.period)

    def next(self):
        number_of_open_positions = len(
            [
                d
                for d in self.datas
                if d.close[0] >= self.highest[d][0] and self.stdev[d][0] != 0
            ]
        )
        for d in self.datas:
            if self.stdev[d][0] == 0:
                continue
            if d.close[0] >= self.highest[d][0] and self.getposition(d).size == 0:
                target = 1.0 / number_of_open_positions * self.params.leverage
                self.order_target_percent(d, target=target)
            elif (
                d.close[0] < self.highest[d][0] - 2 * self.atr[d][0]
                and self.getposition(d).size > 0
            ):
                self.order_target_percent(d, target=0.0)
        # Log portfolio value for performance analysis
        self.log_data.append(
            {
                "date": self.datas[0].datetime.date(0).isoformat(),
                "value": self.broker.getvalue(),
            }
        )
        self.post_next_actions()

    def get_latest_positions(self):
        # Retrieve the latest positions in the portfolio
        positions = {
            data._name: self.broker.getposition(data).size for data in self.datas
        }
        return positions


if __name__ == "__main__":
    trade_config = {
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        "silence": True,
        "setup_name": "selected_5",
    }
    # trade_config = {
    #     "tickers": "all",
    #     "silence": True,
    #     "setup_name": "random:50",
    # }
    operator = BacktestingEngine(trade_config)
    operator.run_rolling_window(TrendFollowingStrategy)
    aggregate_results_one_strategy(trade_config["selection_strategy"], TrendFollowingStrategy.__name__)