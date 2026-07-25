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
from backtest.toolkit.llm_cost_monitor import reset_llm_cost, get_llm_cost, get_llm_cost_ledger
from backtest.toolkit.result_writer import write_result_artifacts


class FINSABER:
    def __init__(self, trade_config: dict):
        self.trade_config = TradeConfig.from_dict(trade_config)
        self.framework = FINSABERFrameworkHelper(
            initial_cash=self.trade_config.cash,
            risk_free_rate=self.trade_config.risk_free_rate,
            commission_per_share=self.trade_config.commission_per_share,
            min_commission=self.trade_config.min_commission,
            max_commission_rate=self.trade_config.max_commission_rate,
            execution_timing=self.trade_config.execution_timing,
            slippage_perc=self.trade_config.slippage_perc,
            slippage_impact=self.trade_config.slippage_impact,
            liquidity_lookback_days=self.trade_config.liquidity_lookback_days,
            liquidity_min_history_days=self.trade_config.liquidity_min_history_days,
            liquidity_cap_pct=self.trade_config.liquidity_cap_pct,
        )
        self.data_loader = self.trade_config.data_loader

    def _result_output_dir(self, strategy_class):
        if self.trade_config.result_output_dir:
            # Allow callers to pin benchmark outputs to a run-scoped directory
            # without changing the legacy setup/strategy default tree.
            return os.fspath(self.trade_config.result_output_dir)
        setup_name = str(self.trade_config.setup_name or "default").replace(":", "_")
        return os.path.join(
            self.trade_config.log_base_dir,
            setup_name,
            strategy_class.__name__,
        )

    def _window_key(self):
        return f"{self.trade_config.date_from}_{self.trade_config.date_to}"

    @staticmethod
    def _safe_path_component(value):
        return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")

    def _checkpoint_path(self, strategy_class, ticker):
        window_key = self._safe_path_component(self._window_key())
        ticker_key = self._safe_path_component(ticker)
        return os.path.join(
            self._result_output_dir(strategy_class),
            "checkpoints",
            window_key,
            f"{ticker_key}.pkl",
        )

    def _load_ticker_checkpoint(self, strategy_class, ticker):
        if not (self.trade_config.save_results and self.trade_config.resume_from_checkpoint):
            return None
        checkpoint_path = self._checkpoint_path(strategy_class, ticker)
        if not os.path.exists(checkpoint_path):
            return None
        with open(checkpoint_path, "rb") as file:
            payload = pickle.load(file)
        if isinstance(payload, dict) and payload.get("status") == "complete":
            return payload.get("metrics")
        return payload if isinstance(payload, dict) else None

    def _save_ticker_checkpoint(self, strategy_class, ticker, metrics):
        if not (self.trade_config.save_results and self.trade_config.checkpoint_results):
            return
        checkpoint_path = self._checkpoint_path(strategy_class, ticker)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        payload = {
            "schema_version": 1,
            "status": "complete",
            "window": self._window_key(),
            "ticker": ticker,
            "strategy": strategy_class.__name__,
            "metrics": metrics,
        }
        temp_path = f"{checkpoint_path}.tmp"
        with open(temp_path, "wb") as file:
            pickle.dump(payload, file)
        os.replace(temp_path, checkpoint_path)

    def _write_partial_results(self, strategy_class, eval_metrics):
        if not (self.trade_config.save_results and self.trade_config.checkpoint_results):
            return
        output_dir = self._result_output_dir(strategy_class)
        os.makedirs(output_dir, exist_ok=True)
        write_result_artifacts(output_dir, self.trade_config.to_dict(), {self._window_key(): eval_metrics})


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

        if not self.trade_config.silence:
            print(f"Rolling windows: {rolling_windows}")

        if self.trade_config.setup_name in ["selected_4", "selected_5", "cherry_pick_both_finmem"]:
            stock_selector = FinMemSelector()
        else:
            # TODO: implement other selection strategies
            stock_selector = self.trade_config.selection_strategy


        eval_metrics = {}
        for window in tqdm(rolling_windows, disable=self.trade_config.silence):
            #
            # subset_data = {date: self.all_data[date] for date in window}
            strat_params["date_from"] = window[0]
            strat_params["date_to"] = window[-1]
            self.trade_config.tickers = stock_selector.select(self.trade_config.data_loader, window[0], window[1])
            if not self.trade_config.silence:
                print(f"Selected tickers for the period {window[0]} to {window[1]}: {self.trade_config.tickers}")

            self.trade_config.date_from = window[0]
            self.trade_config.date_to = window[-1]

            metrics = self.run_iterative_tickers(strategy_class, strat_params, tickers=self.trade_config.tickers, delist_check=True)

            # window_key = f"{window[0]}_{window[-1]}"
            eval_metrics.update(metrics)

        # Save results if required
        if self.trade_config.save_results:
            output_dir = self._result_output_dir(strategy_class)
            filename = f"{date_from.date()}_{date_to.date()}.pkl" if self.trade_config.result_filename is None else self.trade_config.result_filename
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, filename), "wb") as f:
                pickle.dump(eval_metrics, f)
            write_result_artifacts(output_dir, self.trade_config.to_dict(), eval_metrics)

        return eval_metrics

    def _print_results(self, metrics, ticker):
        max_drawdown = metrics.get("max_drawdown", 0)
        total_return = metrics.get("total_return", 0)
        annual_return = metrics.get("annual_return", 0)
        annual_volatility = metrics.get("annual_volatility", 0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        sortino_ratio = metrics.get("sortino_ratio", 0)
        total_commission = metrics.get("total_commission", 0)
        total_slippage = metrics.get("total_slippage", 0)
        total_llm_cost = metrics.get("total_llm_cost", 0)
        total_external_cost = metrics.get("total_external_cost", 0)
        total_trading_cost = metrics.get("total_trading_cost", 0)

        print("\n" + "=" * 50)
        print(f"Ticker: {ticker}")
        print(f"Total Return (%): {total_return:.3%}")
        print(f"Annual Return (%): {annual_return:.3%}")
        print(f"Max Drawdown (%): {-max_drawdown:.3f}%")
        print(f"Annual Volatility (%): {annual_volatility:.3%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Sortino Ratio: {sortino_ratio:.3f}")
        print(f"Total Commission: ${total_commission:.3f}")
        print(f"Total Slippage: ${total_slippage:.3f}")
        print(f"Total LLM Cost: ${total_llm_cost:.3f}")
        print(f"Total External Cost: ${total_external_cost:.3f}")
        print(f"Total Trading Cost: ${total_trading_cost:.3f}")
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
                llm_cost_before = get_llm_cost()
                llm_ledger_start = len(get_llm_cost_ledger())
                checkpoint_metrics = self._load_ticker_checkpoint(strategy_class, ticker)
                if checkpoint_metrics is not None:
                    eval_metrics[ticker] = checkpoint_metrics
                    progress.update(task, advance=1,
                                    description=f"Loaded checkpoint for {ticker} on {self.trade_config.date_from} to {self.trade_config.date_to}")
                    if not self.trade_config.silence:
                        print(f"Loaded checkpoint for {ticker} on {self.trade_config.date_from} to {self.trade_config.date_to}.")
                    continue

                progress.update(task,
                                description=f"Backtesting {ticker} on {self.trade_config.date_from} to {self.trade_config.date_to}")

                subset_data = self.data_loader.get_subset_by_time_range(self.trade_config.date_from, self.trade_config.date_to)

                # check if the ticker is in the data
                try:
                    first_day = subset_data.get_date_range()[0]
                except (AttributeError, IndexError):
                    if not self.trade_config.silence:
                        print(f"No data found for the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    continue
                if ticker not in subset_data.get_tickers_list():
                    if not self.trade_config.silence:
                        print(f"Ticker {ticker} not in the data for the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    continue

                self.framework.reset()
                success_or_not = self.framework.load_backtest_data_single_ticker(
                    subset_data,
                    ticker,
                    start_date=self.trade_config.date_from,
                    end_date=self.trade_config.date_to
                )

                if not success_or_not:
                    if not self.trade_config.silence:
                        print(f"Ticker {ticker} not in the data for the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    continue

                resolved_params = self.auto_resolve_params(
                    strat_params,
                    {
                        "date_from": self.trade_config.date_from,
                        "date_to": self.trade_config.date_to,
                        "symbol": ticker,
                        "data_loader": self.data_loader,
                        "tickers": self.trade_config.tickers,
                    }
                )

                # check if there's valid data for the testing and training period
                try:
                    strategy = strategy_class(**resolved_params)
                except InsufficientTrainingDataException as e:
                    if not self.trade_config.silence:
                        print(f"Insufficient training data for {ticker} in the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    continue

                # detect if it is because of the insufficient training data
                try:
                    strategy.train() if hasattr(strategy, "train") else None
                except InsufficientTrainingDataException as e:
                    if not self.trade_config.silence:
                        print(f"Insufficient training data for {ticker} in the period {self.trade_config.date_from} to {self.trade_config.date_to}. Skipping...")
                    continue

                external_cost_offset = get_llm_cost()
                if self.trade_config.llm_cost_as_trade_cost:
                    training_llm_cost = external_cost_offset - llm_cost_before
                    self.framework.charge_external_cost(self.trade_config.date_from, training_llm_cost, reason="llm_training_cost")

                status = self.framework.run(strategy, delist_check=delist_check,
                                            external_cost_getter=get_llm_cost if self.trade_config.llm_cost_as_trade_cost else None,
                                            external_cost_offset=external_cost_offset,
                                            external_cost_reason="llm_inference_cost")
                # Let strategies persist strategy-local artifacts even when the
                # outer framework finishes without receiving a "done" signal.
                if hasattr(strategy, "finalize_backtest_artifacts"):
                    strategy.finalize_backtest_artifacts(status)

                if not status:
                    if not self.trade_config.silence:
                        print(f"Skipping {ticker}...")
                    continue

                metrics = self.framework.evaluate(strategy)
                total_llm_cost = get_llm_cost() - llm_cost_before
                metrics["total_llm_cost"] = total_llm_cost
                metrics["llm_cost_records"] = pd.DataFrame(get_llm_cost_ledger()[llm_ledger_start:])
                equity_with_time = pd.DataFrame({
                    "datetime": strategy.equity_date,
                    "equity": strategy.equity
                })
                metrics["equity_with_time"] = equity_with_time
                metrics["external_costs"] = pd.DataFrame(self.framework.external_costs)
                metrics["trades"] = pd.DataFrame(self.framework.history)
                metrics["rejected_orders"] = pd.DataFrame(self.framework.rejected_orders)
                eval_metrics[ticker] = metrics
                self._save_ticker_checkpoint(strategy_class, ticker, metrics)
                self._write_partial_results(strategy_class, eval_metrics)
                progress.update(task, advance=1)

                if not self.trade_config.silence:
                    self._print_results(metrics, ticker)
                    self._plot_equity_curve(equity_with_time, ticker)


        if self.trade_config.save_results:
            eval_metrics = {f"{self.trade_config.date_from}_{self.trade_config.date_to}": eval_metrics}
            output_dir = self._result_output_dir(strategy_class)
            filename = f"{self.trade_config.date_from}_{self.trade_config.date_to}.pkl" if self.trade_config.result_filename is None else self.trade_config.result_filename
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, filename), "wb") as f:
                pickle.dump(eval_metrics, f)
            write_result_artifacts(output_dir, self.trade_config.to_dict(), eval_metrics)

            aggregate_results_one_strategy(
                self.trade_config.setup_name.replace(":", "_"),
                strategy_class.__name__,
                output_dir=self.trade_config.log_base_dir,
                strategy_output_dir=output_dir,
            )

        # print the estimated cost
        if not self.trade_config.silence:
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
