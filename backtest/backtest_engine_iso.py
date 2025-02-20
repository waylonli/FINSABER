from datetime import datetime
import pandas as pd
from rich.progress import Progress, track
import pickle
import os
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from tqdm import tqdm

from backtest.toolkit.trade_config import TradeConfig
from backtest.toolkit.backtest_framework_iso import BacktestFrameworkIso
from backtest.toolkit.operation_utils import aggregate_results_one_strategy
from backtest.toolkit.llm_cost_monitor import reset_llm_cost, get_llm_cost


class BacktestingEngineIso:
    def __init__(self, trade_config: dict):
        self.trade_config = TradeConfig.from_dict(trade_config)
        self.framework = BacktestFrameworkIso(
            initial_cash=self.trade_config.cash,
            risk_free_rate=self.trade_config.risk_free_rate,
            commission_per_share=self.trade_config.__dict__.get("commission", 0.0049),
            min_commission=self.trade_config.__dict__.get("min_commission", 0.99)
        )
        self.all_data = None
        self._load_data(self.trade_config.all_data)
        self._setup_tickers()

    def _setup_tickers(self):
        if self.trade_config.tickers == "all" and self.trade_config.selection_strategy.startswith("random"):
            num_tickers = int(self.trade_config.selection_strategy.split(":")[1])
            tickers_data = list(self.all_data[next(iter(self.all_data))]["price"].keys())

            np.random.seed(42)
            sampled_tickers = np.random.choice(tickers_data, num_tickers, replace=False)

            output_dir = os.path.join(self.trade_config.log_base_dir, f"random_{num_tickers}")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"random_{num_tickers}_symbols.txt"), "w") as f:
                f.write("\n".join(sampled_tickers))

            self.trade_config.tickers = list(sampled_tickers)

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
        eval_metrics = {}
        for window in tqdm(rolling_windows):
            #
            # subset_data = {date: self.all_data[date] for date in window}

            strat_params["date_from"] = window[0]
            strat_params["date_to"] = window[-1]


            self.trade_config.date_from = window[0]
            self.trade_config.date_to = window[-1]
            # eval_metrics[ticker
            metrics = self.run_iterative_tickers(strategy_class, strat_params, tickers=self.trade_config.tickers)

            window_key = f"{window[0]}_{window[-1]}"
            eval_metrics[window_key] = metrics

        # Save results if required
        if self.trade_config.save_results:
            output_dir = os.path.join(self.trade_config.log_base_dir,
                                      self.trade_config.selection_strategy.replace(":", "_"), strategy_class.__name__)
            filename = f"{self.trade_config.date_from}_{self.trade_config.date_to}.pkl" if self.trade_config.result_filename is None else self.trade_config.result_filename
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

    def run_iterative_tickers(self, strategy_class, strat_params=None, tickers=None):
        reset_llm_cost()
        tickers = tickers or self.trade_config.tickers
        if isinstance(tickers, str) and tickers.lower() == "all":
            tickers = list(self.all_data[next(iter(self.all_data))]["price"].keys())

        eval_metrics = {}
        with Progress() as progress:  # Use Progress
            task = progress.add_task("Iterative Tickers Backtesting", total=len(tickers))
            for ticker in tickers:
                progress.update(task,
                                description=f"Backtesting {ticker} on {self.trade_config.date_from} to {self.trade_config.date_to}")

                subset_data = {date: self.all_data[date] for date in self.all_data if date >= pd.to_datetime(self.trade_config.date_from).date() and date <= pd.to_datetime(self.trade_config.date_to).date()}
                # check if the ticker is in the data
                try:
                    first_day = list(subset_data.keys())[0]
                except:
                    print(f"No data found for the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    import pdb; pdb.set_trace()
                if ticker not in subset_data[first_day]["price"]:
                    print(f"Ticker {ticker} not in the data for the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    continue

                self.framework.reset()
                self.framework.load_backtest_data(subset_data, start_date=self.trade_config.date_from,
                                         end_date=self.trade_config.date_to)

                resolved_params = self.auto_resolve_params(
                    strat_params,
                    {
                        "date_from": self.trade_config.date_from,
                        "date_to": self.trade_config.date_to,
                        "symbol": ticker
                    }
                )
                strategy = strategy_class(**resolved_params)

                # detect if it is because of the insufficient training data
                try:
                    strategy.train() if hasattr(strategy, "train") else None
                except KeyError as e:
                    if ticker in str(e):
                        continue
                    else:
                        raise e


                self.framework.run(strategy)
                metrics = self.framework.evaluate(strategy)
                equity_with_time = pd.DataFrame({
                    "datetime": strategy.equity_date,
                    "equity": strategy.equity
                })
                metrics["equity_with_time"] = equity_with_time
                eval_metrics[ticker] = metrics

                if not self.trade_config.silence:
                    self._print_results(metrics, ticker)
                    self._plot_equity_curve(equity_with_time, ticker)

        if self.trade_config.save_results:
            eval_metrics = {f"{self.trade_config.date_from}_{self.trade_config.date_to}": eval_metrics}
            output_dir = os.path.join(self.trade_config.log_base_dir, self.trade_config.selection_strategy.replace(":", "_"), strategy.__class__.__name__)
            filename = f"{self.trade_config.date_from}_{self.trade_config.date_to}.pkl" if self.trade_config.result_filename is None else self.trade_config.result_filename
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, filename), "wb") as f:
                pickle.dump(eval_metrics, f)

            aggregate_results_one_strategy(self.trade_config.selection_strategy.replace(":", "_"), strategy_class.__name__)

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

    def _load_data(self, data):
        if isinstance(data, str):
            with open(data, 'rb') as file:
                self.all_data = pickle.load(file)
        elif isinstance(data, dict):
            self.all_data = data
        else:
            raise ValueError("Data format not supported.")
