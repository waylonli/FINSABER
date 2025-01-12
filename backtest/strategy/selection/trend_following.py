# https://paperswithbacktest.com/paper/does-trend-following-work-on-stocks#top
import backtrader as bt
import pandas as pd
from tqdm import tqdm
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.backtest_engine import BacktestingEngine


class TrendFollowingStrategy(BaseStrategy):
    params = (
        ("atr_period", 10),
        ("period", 21),  # Equivalent to 10 years of daily data
        ("leverage", 0.9),  # To avoid full investment
        ("total_days", 0),
    )

    def __init__(self):
        super().__init__()
        self.log_data = []  # Store portfolio values for logging
        self.order_list = []  # Store order details for logging
        self.highest = {}
        self.atr = {}
        self.stdev = {}
        self.max_close = {}
        for d in self.datas:
            # Set up indicators for each data feed
            self.highest[d] = bt.indicators.Highest(d.close, period=self.params.period)
            self.atr[d] = bt.indicators.ATR(d, period=self.params.atr_period)
            self.stdev[d] = bt.indicators.StdDev(d.close, period=self.params.period)

    def next(self):
        number_of_open_positions = len(
            [
                d
                for d in self.datas
                if d.close[0] >= self.highest[d][0] and self.stdev[d][0] != 0
            ]
        )
        for d in self.datas:
            if self.stdev[d][0] == 0:
                continue
            if d.close[0] >= self.highest[d][0] and self.getposition(d).size == 0:
                target = 1.0 / number_of_open_positions * self.params.leverage
                self.order_target_percent(d, target=target)
            elif (
                d.close[0] < self.highest[d][0] - 2 * self.atr[d][0]
                and self.getposition(d).size > 0
            ):
                self.order_target_percent(d, target=0.0)
        # Log portfolio value for performance analysis
        self.log_data.append(
            {
                "date": self.datas[0].datetime.date(0).isoformat(),
                "value": self.broker.getvalue(),
            }
        )

        self.post_next_actions()

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
        for symbol in pivot_df.columns.levels[1][::2]:
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
    operator = BacktestingEngine(trade_config)
    # operator.execute_all(TrendFollowingStrategy, process=preprocess_df)
    operator.run_rolling_window(TrendFollowingStrategy, process=preprocess_df)