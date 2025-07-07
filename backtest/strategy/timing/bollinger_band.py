# File: preliminary/strategy/timing/bollinger_bands_strategy.py
from backtest.toolkit.operation_utils import aggregate_results_one_strategy
import backtrader as bt
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.finsaber_bt import FINSABERBt


class BollingerBandsStrategy(BaseStrategy):
    params = (
        ("prior_period", 252 * 3),
        ("period", 20),
        ("devfactor", 2.0),
        ("total_days", 0),
    )

    def __init__(self, strat_params=None):
        super().__init__()
        self.bbands = {}

        for d in self.datas:
            self.bbands[d] = bt.indicators.BollingerBands(d.close, period=self.params.period,
                                                          devfactor=self.params.devfactor)

    def next(self):
        for d in self.datas:
            if d.close[0] < self.bbands[d].lines.bot[0] and d.close[-1] >= self.bbands[d].lines.bot[-1]:
                self.buy(data=d, size=self._adjust_size_for_commission(int(self.broker.cash / d.close[0])))
                self.buys.append(d.datetime.date(0))
                self.trades.append(
                    {
                        "date": d.datetime.date(0),
                        "action": "buy",
                        "price": d.close[0],
                        "size": self.getposition(d).size
                    }
                )
            elif d.close[0] > self.bbands[d].lines.top[0] and d.close[-1] <= self.bbands[d].lines.top[-1]:
                self.sell(data=d, size=self.getposition(d).size)
                self.sells.append(d.datetime.date(0))
                self.trades.append(
                    {
                        "date": d.datetime.date(0),
                        "action": "sell",
                        "price": d.close[0],
                        "size": -self.getposition(d).size
                    }
                )
        self.post_next_actions()


if __name__ == "__main__":
    # trade_config = {
    #     "tickers": "all",
    #     "silence": True,
    #     "setup_name": "random:50",
    # }
    trade_config = {
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        "silence": True,
        "setup_name": "selected_5",
    }
    # trade_config = {
    #     "tickers": ["AAPL"],
    #     "date_from": "2022-10-05",
    #     "date_to": "2023-06-10",
    #     "silence": False,
    #     "setup_name": "cherry_pick_debug_AAPL",
    # }
    operator = FINSABERBt(trade_config)
    operator.run_rolling_window(BollingerBandsStrategy)
    aggregate_results_one_strategy(trade_config["selection_strategy"], BollingerBandsStrategy.__name__)
    # operator.execute_iter(BollingerBandsStrategy, test_config=trade_config)