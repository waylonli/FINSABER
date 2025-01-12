import pandas as pd
from tqdm import tqdm
import pickle
import os
from backtest.toolkit.backtest_framework_iso import BacktestFrameworkIso

class BacktestingEngineIso:
    def __init__(self, config: dict):
        # initialize the backtesting framework with the provided config
        self.framework = BacktestFrameworkIso(
            initial_cash=config.cash,
            risk_free_rate=config.risk_free_rate,
            commission_per_share=config.commission,
            min_commission=config.min_commission
        )


    def run_rolling_window(
            self,
            strategy,
            rolling_window_size,
            rolling_window_step):
        date_range = list(self.framework.data.keys())
        rolling_windows = []

        start_idx = 0
        while start_idx + rolling_window_size <= len(date_range):
            rolling_windows.append(date_range[start_idx:start_idx + rolling_window_size])
            start_idx += rolling_window_step

        eval_metrics = {}
        for window in tqdm(rolling_windows, desc="Rolling Window Backtesting"):
            self.framework.cash = self.framework.initial_cash
            self.framework.portfolio = {}
            self.framework.history = []

            subset_data = {date: self.framework.data[date] for date in window}
            self.framework.data = subset_data

            self.framework.run(strategy)
            metrics = self.framework.evaluate()

            window_key = f"{window[0]}_{window[-1]}"
            eval_metrics[window_key] = metrics

        return eval_metrics

    def run_iterative_tickers(
            self,
            strategy,
            tickers: list[str]
    ):
        eval_metrics = {}

        for ticker in tqdm(tickers, desc="Evaluating Tickers"):
            subset_data = {date: {"price": {ticker: data['price'].get(ticker)}} for date, data in
                           self.framework.data.items() if ticker in data['price']}

            self.framework.cash = self.framework.initial_cash
            self.framework.portfolio = {}
            self.framework.history = []
            self.framework.data = subset_data

            self.framework.run(strategy)
            metrics = self.framework.evaluate()
            eval_metrics[ticker] = metrics

        return eval_metrics

    def save_results(self, results, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

# Example usage:
# from backtest_framework import BacktestFramework, sample_strategy

# data = {your_data_structure}
# framework = BacktestFramework(initial_cash=100000)
# framework.load_data(data)

# engine = BacktestingEngine(framework, config={})

# Rolling window evaluation:
# rolling_metrics = engine.run_rolling_window(sample_strategy, rolling_window_size=30, rolling_window_step=15)
# print(rolling_metrics)

# Iterative tickers evaluation:
# tickers = ['AAPL', 'MSFT', 'GOOGL']
# ticker_metrics = engine.run_iterative_tickers(sample_strategy, tickers)
# print(ticker_metrics)

# Save results:
# engine.save_results(rolling_metrics, "results/rolling_metrics.pkl")
