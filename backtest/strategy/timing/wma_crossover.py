import backtrader as bt

from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.backtest_engine import BacktestingEngine
from backtest.toolkit.operation_utils import aggregate_results_one_strategy

class WMAStrategy(BaseStrategy):
    params = (
        ("prior_period", 252 * 2),
        ('short_window', 10),
        ('long_window', 20),
        ('trade_size', 0.95),
        ("total_days", 0),
    )

    def __init__(self, strat_params=None):
        super().__init__()
        # Keep a reference to the "close" line in the data[0] dataseries
        # close price of all the orders
        self.short = self.params.short_window
        self.long = self.params.long_window

        self.dataclose = self.datas[0].close
        # order
        self.order = None
        # order price
        self.buyprice = None
        # order commission
        self.buycomm = None
        # simple moving average indicators
        wma1 = bt.ind.WMA(period=self.short)  # fast moving average
        wma2 = bt.ind.WMA(period=self.long)  # slow moving average
        self.crossover = bt.ind.CrossOver(wma1, wma2)  # crossover signal


    def next(self):
        if self.crossover > 0:  # if fast crosses slow to the upside
            if self.position.size < 0:
                self.close()
            self.buy(size=self._adjust_size_for_commission(self.calculate_trade_size()))
            self.buys.append(self.data.datetime.date(0))

        if self.crossover < 0:  # in the market & cross to the downside
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

    operator = BacktestingEngine(trade_config)
    # operator.execute_iter(WMAStrategy)
    operator.run_rolling_window(WMAStrategy)
    aggregate_results_one_strategy(trade_config["selection_strategy"], WMAStrategy.__name__)