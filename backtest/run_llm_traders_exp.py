from backtest.experiment_runner import ExperimentRunner
from backtest.strategy.timing_llm import *

def run_llm_strategies(args):
    runner = ExperimentRunner(output_dir=args.output_dir)

    print("=" * 50)
    print("|" + f"Running strategy {args.strategy}".center(48) + "|")
    print("=" * 50 + "\n")

    runner = ExperimentRunner(output_dir=args.output_dir)

    custom_trade_config = {
        "date_from": args.date_from,
        "date_to": args.date_to,
        "rolling_window_size": args.rolling_window_size,
        "rolling_window_step": args.rolling_window_step,
    }
    print("Period: ", custom_trade_config["date_from"], " to ", custom_trade_config["date_to"])

    if "finmem" in args.strategy.lower():
        runner.run(
            custom_trade_config=custom_trade_config,
            setup_name=args.setup,
            strategy_class=FinMemStrategy,
            strat_config_path=args.strat_config_path
        )

    elif "finagent" in args.strategy.lower():
        runner.run(
            custom_trade_config=custom_trade_config,
            setup_name=args.setup,
            strategy_class=FinAgentStrategy,
            strat_config_path=args.strat_config_path
        )

    else:
        raise Exception("Unknown strategy")

    return



if __name__ == "__main__":
    # args parsing
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--setup", type=str, required=True)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--strat_config_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="backtest/output")
    parser.add_argument("--date_from", type=str, default="2004-01-01")
    parser.add_argument("--date_to", type=str, default="2024-01-01")
    parser.add_argument("--rolling_window_size", type=int, default=1)
    parser.add_argument("--rolling_window_step", type=int, default=1)

    args = parser.parse_args()

    run_llm_strategies(args)
