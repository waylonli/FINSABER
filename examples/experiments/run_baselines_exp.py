import argparse
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backtest.strategy.timing  # noqa: E402
from experiment_runner import ExperimentRunner  # noqa: E402


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
        "training_years": args.training_years,
    }

    for _, strategy in inspect.getmembers(strat_namespace):
        if not inspect.isclass(strategy):
            continue
        if include_strats and strategy.__name__ not in include_strats:
            print(f"Skipping {strategy.__name__}")
            continue
        if strategy.__name__ in exclude_strats:
            print(f"Skipping {strategy.__name__}")
            continue

        print("=" * 50)
        print("|" + f"Running strategy {strategy.__name__}".center(48) + "|")
        print("=" * 50 + "\n")
        runner.run(custom_trade_config=custom_trade_config, setup_name=args.setup, strategy_class=strategy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--setup", type=str, required=True)
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--include", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="backtest/output")
    parser.add_argument("--rolling_window_size", type=int, default=1)
    parser.add_argument("--rolling_window_step", type=int, default=1)
    parser.add_argument("--training_years", type=int, default=None)
    parser.add_argument("--date_from", type=str, default="2005-01-01")
    parser.add_argument("--date_to", type=str, default="2007-01-01")
    run_timing_strategies(parser.parse_args())
