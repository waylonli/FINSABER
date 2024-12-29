import inspect
import os.path
import sys
import pickle

from preliminary.operation_utils import aggregate_results
import preliminary.strategy.timing
from preliminary.backtest_engine import BacktestingEngine

def run_cherry_exps(mode:str="both_finmem"):
    if mode == "both_finmem":
        trade_config = {
            "date_from": "2022-10-06",
            "date_to": "2023-04-10",
            "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
            "silence": True,
            "selection_strategy": "cherry_pick_both_finmem"
        }
    elif mode == "extend_symbol_finmem":
        with open(os.path.join("preliminary", "output", "random_50", "random_50_symbols.txt"), "r") as f:
            test_symbols = f.readlines()
        trade_config = {
            "date_from": "2022-10-06",
            "date_to": "2023-04-10",
            "tickers": test_symbols,
            "silence": True,
            "selection_strategy": "cherry_pick_extend_symbol_finmem"
        }
    elif mode == "both_fincon":
        trade_config = {
            "date_from": "2022-10-05",
            "date_to": "2023-06-10",
            "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN", "NIO", "GOOG", "AAPL"],
            "silence": True,
            "selection_strategy": "cherry_pick_both_fincon"
        }
    elif mode == "extend_symbol_fincon":
        with open(os.path.join("preliminary", "output", "random_50", "random_50_symbols.txt"), "r") as f:
            test_symbols = f.readlines()
        trade_config = {
            "date_from": "2022-10-05",
            "date_to": "2023-06-10",
            "tickers": test_symbols,
            "silence": True,
            "selection_strategy": "cherry_pick_extend_symbol_fincon"
        }


    strat_namespace = sys.modules["preliminary.strategy.timing"]
    eval_metrics = []

    # loop all the strategies defined in  preliminary.strategy.timing
    for _, strategy in inspect.getmembers(strat_namespace):
        if not inspect.isclass(strategy):
            continue
        print("="*50)
        print("|" + f"Running strategy {strategy.__name__}".center(48) + "|")
        print("="*50+"\n")
        try:
            operator = BacktestingEngine(trade_config)
            metric = operator.execute_iter(strategy, test_config=trade_config)
            eval_metrics.append(metric)
        except Exception as e:
            print(f"Error running strategy {strategy.__name__}: {e}")
            continue

    print("Finish backtesting all strategies, aggregating results...")
    aggregate_results(trade_config["selection_strategy"])

    return

if __name__ == "__main__":
    # args parsing
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)

    args = parser.parse_args()

    if args.exp == "cherry":
        run_cherry_exps(mode=args.mode)