import inspect
import os.path
import sys

from backtest.experiment_runner import ExperimentRunner
from backtest.toolkit.operation_utils import aggregate_results
from backtest.backtest_engine import BacktestingEngine

def run_timing_strategies(args):
    exclude_strats = args.exclude.split(",") if args.exclude else []
    include_strats = args.include.split(",") if args.include else []
    strat_namespace = sys.modules["backtest.strategy.timing"]

    runner = ExperimentRunner(output_dir=args.output_dir)

    # loop all the strategies defined in preliminary.strategy.timing
    for _, strategy in inspect.getmembers(strat_namespace):
        if not inspect.isclass(strategy):
            continue

        if len(include_strats) > 0 and strategy.__name__ not in include_strats:
            print(f"Skipping {strategy.__name__}")
            continue

        if strategy.__name__ in exclude_strats:
            print(f"Skipping {strategy.__name__}")

        print("=" * 50)
        print("|" + f"Running strategy {strategy.__name__}".center(48) + "|")
        print("=" * 50 + "\n")
        # try:
        runner.run(setup_name=args.setup, strategy_class=strategy)
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

    args = parser.parse_args()

    run_timing_strategies(args)
