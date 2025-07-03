import json

import os

from backtest.data_util import FinMemDataset
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.strategy.timing_llm.base_strategy_iso import BaseStrategyIso
from backtest.strategy.selection import RandomSP500Selector, MomentumSP500Selector, LowVolatilitySP500Selector
from backtest.finsaber_bt import FINSABERBt
from backtest.finsaber import FINSABER
from backtest.toolkit.operation_utils import aggregate_results_one_strategy

class ExperimentRunner:
    def __init__(self, output_dir: str = os.path.join("backtest", "output")):
        self.output_dir = output_dir
        self.mode = None

    def run(
            self,
            setup_name: str,
            strategy_class: BaseStrategy | BaseStrategyIso,
            custom_trade_config: dict = None,
            strat_config_path: str = None
    ):
        """
        :param setup_name: one of the following:
            - cherry_pick_both_finmem
            - cherry_pick_both_fincon
            - selected_5
            - selected_4
            - random_sp500_10
        :param strategy_class: strategy class to run
        :param trade_config: custom trade config to override the default
        :param strat_config_path: path to the strategy config file
        :return:
        """
        if setup_name == "cherry_pick_both_finmem":
            default_config = {
                "data_loader": FinMemDataset(pickle_file="data/finmem_data/stock_data_cherrypick_2000_2024.pkl"),
                "date_from": "2022-10-06",
                "date_to": "2023-04-10",
                "tickers": [
                    "TSLA",
                    "NFLX",
                    "AMZN",
                    "MSFT",
                    "COIN"
                ],
                "silence": False,
                "setup_name": setup_name
            }
            self.mode = "iter"
        elif setup_name == "cherry_pick_both_fincon":
            default_config = {
                "data_loader": FinMemDataset(pickle_file="data/finmem_data/stock_data_cherrypick_2000_2024.pkl"),
                "date_from": "2022-10-05",
                "date_to": "2023-06-10",
                "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN", "NIO", "GOOG", "AAPL"],
                "silence": True,
                "setup_name": setup_name
            }
            self.mode = "iter"
        elif setup_name in ["selected_5", "selected_4"]:
            default_config = {
                "data_loader": FinMemDataset(pickle_file="data/finmem_data/stock_data_cherrypick_2000_2024.pkl"),
                "date_from": "2004-01-01",
                "date_to": "2024-01-01",
                "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
                "silence": True,
                "setup_name": setup_name
            }
            self.mode = "rolling_window"
        elif setup_name.startswith("random_sp500_"):
            default_config = {
                "data_loader": FinMemDataset(pickle_file="data/finmem_data/stock_data_sp500_2000_2024.pkl"),
                "date_from": "2004-01-01",
                "date_to": "2024-01-01",
                "tickers": "all",
                "silence": True,
                "setup_name": setup_name,
                "selection_strategy": RandomSP500Selector(
                    num_tickers=int(setup_name.split("_")[-1]),
                    random_seed_setting="year"
                )
            }
            self.mode = "rolling_window"
        elif setup_name.startswith("momentum_sp500_"):
            default_config = {
                "data_loader": FinMemDataset(pickle_file="data/finmem_data/stock_data_sp500_2000_2024.pkl"),
                "date_from": "2004-01-01",
                "date_to": "2024-01-01",
                "tickers": "all",
                "silence": True,
                "setup_name": setup_name,
                "selection_strategy": MomentumSP500Selector(
                    num_tickers=int(setup_name.split("_")[-1]),
                    momentum_period=100,
                    skip_period=21,
                    training_period=2
                )
            }
            self.mode = "rolling_window"
        elif setup_name.startswith("lowvol_sp500_"):
            default_config = {
                "data_loader": FinMemDataset(pickle_file="data/finmem_data/stock_data_sp500_2000_2024.pkl"),
                "date_from": "2004-01-01",
                "date_to": "2024-01-01",
                "tickers": "all",
                "silence": True,
                "setup_name": setup_name,
                "selection_strategy": LowVolatilitySP500Selector(
                    num_tickers=int(setup_name.split("_")[-1]),
                    lookback_period=21,
                    training_period=2
                )
            }
            self.mode = "rolling_window"
        else:
            raise NotImplementedError(f"setup_name {setup_name} is not implemented")

        if custom_trade_config is not None:
            default_config.update(custom_trade_config)

        trade_config = default_config

        strat_config = json.load(open(strat_config_path)) if strat_config_path else None

        self._run_backtest(strategy_class, trade_config, strat_config)

    def _run_backtest(self, strategy_class, trade_config, strat_config):
        if strat_config:
            operator = FINSABER(trade_config)
        else:
            operator = FINSABERBt(trade_config)

        import pdb; pdb.set_trace()

        if self.mode == "rolling_window":
            operator.run_rolling_window(strategy_class, strat_params=strat_config)
        elif self.mode == "iter":
            operator.run_iterative_tickers(strategy_class, strat_params=strat_config)

        aggregate_results_one_strategy(trade_config["setup_name"], strategy_class.__name__, output_dir=self.output_dir)