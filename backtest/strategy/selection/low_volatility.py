# https://paperswithbacktest.com/paper/the-volatility-effect-lower-risk-without-lower-return#top
import os
import backtrader as bt
import pandas as pd
import numpy as np
import datasets as ds
from tqdm import tqdm
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.finsaber_bt import FINSABERBt
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.finsaber_bt import FINSABERBt


class LowVolatilityStrategy(BaseStrategy):
    params = (
        ("lookback_period", 21),  # 1 month of daily data
        ("rebalance_period", 21),  # Monthly rebalancing
        ("leverage", 1.0),
        ("total_days", 0),
    )

    def __init__(self):
        super().__init__()
        self.rebalance_counter = 0
        self.data_volatility = {}
        self.selected_stocks = []
        self.log_data = []  # Store portfolio values for logging
        # Add a timer to handle monthly rebalance
        self.add_timer(when=bt.Timer.SESSION_START, monthdays=[1], monthcarry=True)

    def notify_timer(self, timer, when, *args, **kwargs):
        self.rebalance()

    def next(self):
        self.log_data.append(
            {
                "date": self.datas[0].datetime.date(0).isoformat(),
                "value": self.broker.getvalue(),
            }
        )
        self.post_next_actions()

    def rebalance(self):
        # Calculate volatilities
        self.data_volatility = {}
        for d in self.datas:
            prices = d.close.get(size=self.params.lookback_period)
            if len(prices) < self.params.lookback_period:
                continue
            weekly_returns = [
                np.log(prices[i] / prices[i - 5]) for i in range(5, len(prices), 5)
            ]
            is_tradable_stock = (
                len(weekly_returns) > 0 and np.std(weekly_returns) != 0.0
            )
            if is_tradable_stock:
                self.data_volatility[d._name] = np.std(weekly_returns)

        # Select stocks with lowest volatility (top quartile)
        sorted_stocks = sorted(self.data_volatility.items(), key=lambda x: x[1])
        num_stocks = len(sorted_stocks) // 10
        self.selected_stocks = [x[0] for x in sorted_stocks[:num_stocks]]

        # Adjust portfolio
        for d in self.datas:
            if d._name in self.selected_stocks:
                self.order_target_percent(
                    d, target=1.0 / num_stocks * self.params.leverage
                )
            else:
                self.order_target_percent(d, target=0.0)

    def get_latest_positions(self):
        # Retrieve the latest positions in the portfolio
        positions = {
            data._name: self.broker.getposition(data).size for data in self.datas
        }
        return positions


if __name__ == "__main__":
    def preprocess_df(df):
        # Adjust the data for splits and dividends
        df["date"] = pd.to_datetime(df.index)
        # reset index
        df.reset_index(drop=True, inplace=True)
        df = df[df["date"] > "1990-01-01"].copy()
        adj_factor = df["adj_close"] / df["close"]
        df["adj_open"] = df["open"] * adj_factor
        df["adj_high"] = df["high"] * adj_factor
        df["adj_low"] = df["low"] * adj_factor
        df = df[["date", "symbol", "adj_open", "adj_high", "adj_low", "adj_close"]]
        df.rename(
            columns={
                "adj_open": "open",
                "adj_high": "high",
                "adj_low": "low",
                "adj_close": "close",
            },
            inplace=True,
        )

        # Pivot the data to have a column for each stock
        pivot_df = pd.pivot_table(
            df,
            index="date",
            columns="symbol",
            values=["open", "high", "low", "close"],
            aggfunc="first",
        )
        full_date_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
        pivot_df = pivot_df.reindex(full_date_range)

        symbol_dfs = []

        # Process data and add to Cerebro
        for symbol in pivot_df.columns.levels[1][::5]:
            symbol_df = pivot_df.xs(symbol, axis=1, level=1, drop_level=False).copy()
            symbol_df.columns = symbol_df.columns.droplevel(1)
            symbol_df.reset_index(inplace=True)
            symbol_df.rename(columns={"index": "date"}, inplace=True)
            symbol_df.set_index("date", inplace=True)
            symbol_df.ffill(inplace=True)
            symbol_df.bfill(inplace=True)

            # Skip stocks with low share prices
            if symbol_df["close"].min() < 5:
                continue
            symbol_df["symbol"] = symbol
            symbol_dfs.append(symbol_df)

        return symbol_dfs

    trade_config = {
        "tickers": "all",
        "strategy_type": "selection",
    }
    operator = FINSABERBt(trade_config)
    # operator.execute_all(LowVolatilityStrategy, process=preprocess_df)
    operator.run_rolling_window(LowVolatilityStrategy, process=preprocess_df)
