import inspect
import os.path
import sys

from backtest.data_util import FinMemDataset
from backtest.experiment_runner import ExperimentRunner
from backtest.strategy.selection import RandomSP500Selector
from backtest.strategy.timing_iso import *
from backtest.toolkit.operation_utils import aggregate_results
from backtest.backtest_engine import BacktestingEngine

def run_llm_strategies(args):
    runner = ExperimentRunner(output_dir=args.output_dir)

    if "finmem" in args.strategy.lower():

        trade_config = {
            "tickers": "all",
            "silence": True,
            "data_loader": FinMemDataset(pickle_file="data/finmem_data/stock_data_sp500_2000_2014.pkl"),
        }

        llm_trader_class = FinMemStrategy

    elif "finagent" in args.strategy.lower():
        pass

    else:
        raise Exception("Unknown strategy")

    print("=" * 50)
    print("|" + f"Running strategy {llm_trader_class.__name__}".center(48) + "|")
    print("=" * 50 + "\n")
    runner.run(
        setup_name=args.setup,
        strategy_class=llm_trader_class,
        trade_config=trade_config,
        strat_config_path="strats_configs/finmem_config_normal.json"
    )

    return



if __name__ == "__main__":
    # args parsing
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--setup", type=str, required=True)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="backtest/output")

    args = parser.parse_args()

    run_llm_strategies(args)
