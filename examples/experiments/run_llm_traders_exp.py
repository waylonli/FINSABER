import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Use a non-interactive matplotlib backend for the shared experiment launcher.
# This preserves console progress/output while preventing FINSABER's optional
# equity-curve plotting path from blocking in headless or GUI-sensitive runs.
os.environ.setdefault("MPLBACKEND", "Agg")

from experiment_runner import ExperimentRunner  # noqa: E402


def run_llm_strategies(args):
    runner = ExperimentRunner(output_dir=args.output_dir, data_root=args.data_root)

    custom_trade_config = {
        "date_from": args.date_from,
        "date_to": args.date_to,
        "rolling_window_size": args.rolling_window_size,
        "rolling_window_step": args.rolling_window_step,
    }
    print("Period: ", custom_trade_config["date_from"], " to ", custom_trade_config["date_to"])

    if "finmem" in args.strategy.lower():
        from llm_traders.finsaber_strategies.finmem import FinMemStrategy

        strategy_class = FinMemStrategy
    elif "finagent" in args.strategy.lower():
        from llm_traders.finsaber_strategies.finagent import FinAgentStrategy

        strategy_class = FinAgentStrategy
    elif "tradingagents" in args.strategy.lower():
        from llm_traders.finsaber_strategies.tradingagents import TradingAgentsStrategy

        strategy_class = TradingAgentsStrategy
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    runner.run(
        custom_trade_config=custom_trade_config,
        setup_name=args.setup,
        strategy_class=strategy_class,
        strat_config_path=args.strat_config_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM trader experiments")
    parser.add_argument("--setup", type=str, required=True)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--strat_config_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="backtest/output")
    parser.add_argument("--date_from", type=str, default="2004-01-01")
    parser.add_argument("--date_to", type=str, default="2024-01-01")
    parser.add_argument("--rolling_window_size", type=int, default=1)
    parser.add_argument("--rolling_window_step", type=int, default=1)
    parser.add_argument("--data_root", type=str, default=None)
    run_llm_strategies(parser.parse_args())
