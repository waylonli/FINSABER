from backtest.strategy.selection import RandomSP500Selector
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.finsaber_bt import FINSABERBt
from backtest.toolkit.operation_utils import aggregate_results_one_strategy
from backtest.strategy.selection import *

# Create a Strategy
class BuyAndHoldStrategy(BaseStrategy):
    params = (
        ("prior_period", 252 * 3),
        ("total_days", 0),
    )

    def __init__(self, strat_params=None):
        super().__init__()
        self.dataclose = self.datas[0].close

    def next(self):
        if not self.position:
            max_size = self._adjust_size_for_commission(int(self.broker.cash / self.dataclose[0]))
            self.buy(size=max_size)
            self.trades.append(
                {
                    "date": self.datas[0].datetime.date(0),
                    "size": max_size,
                    "price": self.dataclose[0],
                    "action": "buy",
                }
            )
        self.post_next_actions()


if __name__ == '__main__':

    # trade_config = {
    #     "tickers": [
    #         # "TSLA",
    #         "NFLX",
    #         # "AMZN",
    #         # "MSFT",
    #         # "COIN"
    #     ],
    #     "silence": True,
    #     "setup_name": "debug",
    # }
    trade_config = {
        "tickers": "all",
        "silence": True,
        "setup_name": "debug",
        "selection_strategy": RandomSP500Selector(num_tickers=10, random_seed_setting="year"),
    }
    operator = FINSABERBt(trade_config)
    # operator.execute_iter(BuyAndHoldStrategy)
    metrics = operator.run_rolling_window(BuyAndHoldStrategy)
    print(metrics)
    # aggregate_results_one_strategy(trade_config["setup_name"], BuyAndHoldStrategy.__name__)
