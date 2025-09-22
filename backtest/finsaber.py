import copy
from datetime import datetime
import pandas as pd
from rich.progress import Progress, track
import pickle
import os
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from tqdm import tqdm

from backtest.strategy.selection import FinMemSelector
from backtest.toolkit.custom_exceptions import InsufficientTrainingDataException
from backtest.toolkit.trade_config import TradeConfig
from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper
from backtest.toolkit.operation_utils import aggregate_results_one_strategy
from backtest.toolkit.llm_cost_monitor import reset_llm_cost, get_llm_cost
from backtest.toolkit.execution_helper import ExecutionContext
from backtest.data_util.market_data_provider import MarketDataProvider


class FINSABER:
    def __init__(self, trade_config: dict):
        self.trade_config = TradeConfig.from_dict(trade_config)
        self.framework = FINSABERFrameworkHelper(
            initial_cash=self.trade_config.cash,
            risk_free_rate=self.trade_config.risk_free_rate,
            commission_per_share=self.trade_config.__dict__.get("commission", 0.0049),
            min_commission=self.trade_config.__dict__.get("min_commission", 0.99)
        )
        self.data_loader = self.trade_config.data_loader
        self.market_data = MarketDataProvider(
            adv_window=self.trade_config.execution.liquidity.adv_window,
        )


    def run_rolling_window(self, strategy_class, rolling_window_size=None, rolling_window_step=None, strat_params=None):
        rolling_window_size = rolling_window_size or self.trade_config.rolling_window_size
        rolling_window_step = rolling_window_step or self.trade_config.rolling_window_step # in years

        date_from = pd.to_datetime(self.trade_config.date_from)
        date_to = pd.to_datetime(self.trade_config.date_to)
        total_years = (date_to.year - date_from.year) + 1

        rolling_windows = []

        # get the first year
        start_year = date_from.year
        # get rolling windows
        for i in range(0, total_years - rolling_window_size, rolling_window_step):
            # get yyyy-mm-dd
            start_date = f"{start_year + i}-01-01"
            end_date = f"{start_year + i + rolling_window_size}-01-01"
            rolling_windows.append((start_date, end_date))

        print(rolling_windows)

        if self.trade_config.setup_name in ["selected_4", "selected_5", "cherry_pick_both_finmem"]:
            stock_selector = FinMemSelector()
        else:
            # TODO: implement other selection strategies
            stock_selector = self.trade_config.selection_strategy


        eval_metrics = {}
        for window in tqdm(rolling_windows):
            #
            # subset_data = {date: self.all_data[date] for date in window}
            strat_params["date_from"] = window[0]
            strat_params["date_to"] = window[-1]
            self.trade_config.tickers = stock_selector.select(self.trade_config.data_loader, window[0], window[1])
            print(f"Selected tickers for the period {window[0]} to {window[1]}: {self.trade_config.tickers}")

            self.trade_config.date_from = window[0]
            self.trade_config.date_to = window[-1]

            metrics = self.run_iterative_tickers(strategy_class, strat_params, tickers=self.trade_config.tickers, delist_check=True)

            # window_key = f"{window[0]}_{window[-1]}"
            eval_metrics.update(metrics)

        # Save results if required
        if self.trade_config.save_results:
            output_dir = os.path.join(self.trade_config.log_base_dir,
                                      self.trade_config.setup_name.replace(":", "_"), strategy_class.__name__)
            filename = f"{date_from.date()}_{date_to.date()}.pkl" if self.trade_config.result_filename is None else self.trade_config.result_filename
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, filename), "wb") as f:
                pickle.dump(eval_metrics, f)

        return eval_metrics

    def _print_results(self, metrics, ticker):
        max_drawdown = metrics.get("max_drawdown", 0)
        total_return = metrics.get("total_return", 0)
        annual_return = metrics.get("annual_return", 0)
        annual_volatility = metrics.get("annual_volatility", 0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        sortino_ratio = metrics.get("sortino_ratio", 0)
        total_commission = metrics.get("total_commission", 0)

        print("\n" + "=" * 50)
        print(f"Ticker: {ticker}")
        print(f"Total Return (%): {total_return:.3%}")
        print(f"Annual Return (%): {annual_return:.3%}")
        print(f"Max Drawdown (%): {-max_drawdown:.3f}%")
        print(f"Annual Volatility (%): {annual_volatility:.3%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Sortino Ratio: {sortino_ratio:.3f}")
        print(f"Total Commission: ${total_commission:.3f}")
        print("=" * 50)

    def _plot_equity_curve(self, equity_with_time, ticker):
        plt.figure(figsize=(10, 6))
        plt.plot(equity_with_time["datetime"], equity_with_time["equity"], label="Equity Curve")
        plt.title(f"Equity Curve for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.show()

    def run_iterative_tickers(self, strategy_class, strat_params=None, tickers=None, delist_check=False):
        reset_llm_cost()
        tickers = tickers or self.trade_config.tickers
        if isinstance(tickers, str) and tickers.lower() == "all":
            tickers = self.data_loader.get_tickers_list()

        eval_metrics = {}
        with Progress() as progress:  # Use Progress
            task = progress.add_task("Iterative Tickers Backtesting", total=len(tickers))
            for ticker in tickers:
                progress.update(task,
                                description=f"Backtesting {ticker} on {self.trade_config.date_from} to {self.trade_config.date_to}")

                subset_data = self.data_loader.get_subset_by_time_range(self.trade_config.date_from, self.trade_config.date_to)

                # check if the ticker is in the data
                try:
                    first_day = subset_data.get_date_range()[0]
                except:
                    print(f"No data found for the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    import pdb; pdb.set_trace()
                if ticker not in subset_data.get_tickers_list():
                    print(f"Ticker {ticker} not in the data for the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    continue

                self.framework.reset()
                execution_context = ExecutionContext(
                    config=copy.deepcopy(self.trade_config.execution),
                    market_data=self.market_data,
                )
                self.framework.set_execution_context(execution_context)
                success_or_not = self.framework.load_backtest_data_single_ticker(
                    subset_data,
                    ticker,
                    start_date=self.trade_config.date_from,
                    end_date=self.trade_config.date_to
                )

                if not success_or_not:
                    print(f"Ticker {ticker} not in the data for the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    continue

                resolved_params = self.auto_resolve_params(
                    strat_params,
                    {
                        "date_from": self.trade_config.date_from,
                        "date_to": self.trade_config.date_to,
                        "symbol": ticker
                    }
                )

                # check if there's valid data for the testing and training period
                try:
                    strategy = strategy_class(**resolved_params)
                except InsufficientTrainingDataException as e:
                    print(f"Insufficient training data for {ticker} in the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    continue

                # detect if it is because of the insufficient training data
                try:
                    strategy.train() if hasattr(strategy, "train") else None
                except InsufficientTrainingDataException as e:
                    print(f"Insufficient training data for {ticker} in the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    continue

                status = self.framework.run(strategy, delist_check=delist_check)

                if not status:
                    print(f"Skipping {ticker}...")
                    continue

                metrics = self.framework.evaluate(strategy)
                try:
                    equity_with_time = pd.DataFrame({
                        "datetime": strategy.equity_date,
                        "equity": strategy.equity
                    })
                except:
                    print(len(strategy.equity_date), len(strategy.equity))
                    import pdb; pdb.set_trace()
                metrics["equity_with_time"] = equity_with_time
                eval_metrics[ticker] = metrics

                if not self.trade_config.silence:
                    self._print_results(metrics, ticker)
                    self._plot_equity_curve(equity_with_time, ticker)


        if self.trade_config.save_results:
            eval_metrics = {f"{self.trade_config.date_from}_{self.trade_config.date_to}": eval_metrics}
            output_dir = os.path.join(self.trade_config.log_base_dir, self.trade_config.setup_name.replace(":", "_"), strategy.__class__.__name__)
            filename = f"{self.trade_config.date_from}_{self.trade_config.date_to}.pkl" if self.trade_config.result_filename is None else self.trade_config.result_filename
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, filename), "wb") as f:
                pickle.dump(eval_metrics, f)

            aggregate_results_one_strategy(self.trade_config.setup_name.replace(":", "_"), strategy_class.__name__)

        # print the estimated cost
        print(f"Finish backtesting period {self.trade_config.date_from} to {self.trade_config.date_to}. Estimated cost: ${get_llm_cost()}")
        return eval_metrics

    def auto_resolve_params(self, strat_params, trade_config):
        resolved_params = {}

        for key, value in strat_params.items():
            if isinstance(value, str) and value.startswith("$"):
                if value[1:] in trade_config:
                    resolved_params[key] = trade_config[value[1:]]
                else:
                    raise ValueError(f"Unsupported dynamic parameter: {key}")
            else:
                resolved_params[key] = value

        return resolved_params
