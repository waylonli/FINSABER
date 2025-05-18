import inspect
import os.path
import sys

from backtest.experiment_runner import ExperimentRunner
from backtest.toolkit.operation_utils import aggregate_results
from backtest.finsaber_bt import FINSABERBt

def run_timing_strategies(args):
    exclude_strats = args.exclude.split(",") if args.exclude else []
    include_strats = args.include.split(",") if args.include else []
    strat_namespace = sys.modules["backtest.strategy.timing"]

    runner = ExperimentRunner(output_dir=args.output_dir)

    custom_trade_config = {
        "rolling_window_size": args.rolling_window_size,
        "rolling_window_step": args.rolling_window_step,
        "date_from": args.date_from,
        "date_to": args.date_to,
    }

    # loop all the strategies defined in preliminary.strategy.timing
    for _, strategy in inspect.getmembers(strat_namespace):
        if not inspect.isclass(strategy):
            continue

        if len(include_strats) > 0 and strategy.__name__ not in include_strats:
            print(f"Skipping {strategy.__name__}")
            continue

        if strategy.__name__ in exclude_strats:
            print(f"Skipping {strategy.__name__}")
            continue

        print("=" * 50)
        print("|" + f"Running strategy {strategy.__name__}".center(48) + "|")
        print("=" * 50 + "\n")
        # try:
        runner.run(custom_trade_config=custom_trade_config, setup_name=args.setup, strategy_class=strategy)
        # except Exception as e:
        #     print(f"Error running strategy {strategy.__name__}: {e}")
        #     continue

    return



if __name__ == "__main__":
    # args parsing
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--setup", type=str, required=True)
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--include", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="backtest/output")
    parser.add_argument("--rolling_window_size", type=int, default=1)
    parser.add_argument("--rolling_window_step", type=int, default=1)
    parser.add_argument("--date_from", type=str, default="2005-01-01")
    parser.add_argument("--date_to", type=str, default="2007-01-01")

    args = parser.parse_args()

    run_timing_strategies(args)
