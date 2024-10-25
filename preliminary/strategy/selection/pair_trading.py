# https://paperswithbacktest.com/paper/pairs-trading-performance-of-a-relative-value-arbitrage-rule#top

import backtrader as bt
import numpy as np
import itertools as it
import pandas as pd
import datasets as ds
from tqdm import tqdm
from preliminary.strategy.timing.base_strategy import BaseStrategy
from preliminary.backtest_engine import BacktestingEngine


class PairsTradingStrategy(BaseStrategy):
    params = (
        ("period", 21),  # Period for rolling window (12 months of trading days)
        ("leverage", 0.8),  # Leverage to apply
        ("selection_month", 1),  # Month for stock selection
        ("max_traded_pairs", 8),  # Maximum number of traded pairs
        ("total_days", 0),
    )

    def __init__(self):
        super().__init__()
        self.log_data = []  # Store portfolio values for logging
        self.history_price = {d._name: [] for d in self.datas}
        self.traded_pairs = []
        self.traded_quantity = {}
        self.sorted_pairs = []
        self.symbols = [d._name for d in self.datas]
        self.add_timer(when=bt.Timer.SESSION_START, monthdays=[1], monthcarry=True)
        self.broker.set_coc(True)

    def notify_timer(self, timer, when, *args, **kwargs):
        if self.datas[0].datetime.date(0).month == self.params.selection_month:
            self.update_pairs()

    def update_pairs(self):
        self.symbol_pairs = list(it.combinations(self.symbols, 2))

        distances = {}
        for pair in self.symbol_pairs:
            is_tradable_a = (
                len(self.history_price[pair[0]]) == self.params.period
                and self.history_price[pair[0]][-1] != self.history_price[pair[0]][-3]
            )
            is_tradable_b = (
                len(self.history_price[pair[1]]) == self.params.period
                and self.history_price[pair[1]][-1] != self.history_price[pair[1]][-3]
            )
            if is_tradable_a and is_tradable_b:
                distance = self.distance(
                    self.history_price[pair[0]], self.history_price[pair[1]]
                )
                if distance > 0:  # Avoid pairs with no data
                    distances[pair] = distance

        if distances:
            self.sorted_pairs = [
                x[0] for x in sorted(distances.items(), key=lambda x: x[1])
            ]
        # Liquidate all positions
        for d in self.datas:
            self.order_target_percent(d, target=0)
        self.traded_pairs.clear()
        self.traded_quantity.clear()

    def next(self):
        self.log_data.append(
            {
                "date": self.datas[0].datetime.date(0).isoformat(),
                "value": self.broker.getvalue(),
            }
        )
        self.rebalance()
        self.post_next_actions()

    def rebalance(self):
        for d in self.datas:
            self.history_price[d._name].append(d.close[0])
            if len(self.history_price[d._name]) > self.params.period:
                self.history_price[d._name].pop(0)

        pairs_to_remove = []

        for pair in self.sorted_pairs:
            price_a = list(self.history_price[pair[0]])
            price_b = list(self.history_price[pair[1]])
            norm_a = np.array(price_a) / price_a[0]
            norm_b = np.array(price_b) / price_b[0]

            spread = norm_a - norm_b
            mean = np.mean(spread)
            std = np.std(spread)
            actual_spread = spread[-1]

            traded_portfolio_value = (
                self.broker.getvalue()
                / self.params.max_traded_pairs
                * self.params.leverage
            )
            if actual_spread > 0 + 2 * std or actual_spread < 0 - 2 * std:
                if pair not in self.traded_pairs:
                    if len(self.traded_pairs) < self.params.max_traded_pairs:
                        symbol_a = pair[0]
                        symbol_b = pair[1]
                        a_price = price_a[-1]
                        b_price = price_b[-1]

                        if norm_a[-1] > norm_b[-1]:
                            long_q = int(traded_portfolio_value / b_price)
                            short_q = -int(traded_portfolio_value / a_price)
                            if (
                                self.getpositionbyname(symbol_a).size == 0
                                and self.getpositionbyname(symbol_b).size == 0
                            ):
                                self.sell(
                                    self.getdatabyname(symbol_a),
                                    size=abs(short_q),
                                    exectype=bt.Order.Market,
                                )
                                self.buy(
                                    self.getdatabyname(symbol_b),
                                    size=abs(long_q),
                                    exectype=bt.Order.Market,
                                )

                                self.traded_quantity[pair] = (short_q, long_q)
                                self.traded_pairs.append(pair)
                        else:
                            long_q = int(traded_portfolio_value / a_price)
                            short_q = -int(traded_portfolio_value / b_price)
                            if (
                                self.getpositionbyname(symbol_a).size == 0
                                and self.getpositionbyname(symbol_b).size == 0
                            ):
                                self.buy(
                                    self.getdatabyname(symbol_a),
                                    size=abs(long_q),
                                    exectype=bt.Order.Market,
                                )
                                self.sell(
                                    self.getdatabyname(symbol_b),
                                    size=abs(short_q),
                                    exectype=bt.Order.Market,
                                )

                                self.traded_quantity[pair] = (long_q, short_q)
                                self.traded_pairs.append(pair)
            else:
                if pair in self.traded_pairs and pair in self.traded_quantity:
                    if self.traded_quantity[pair][0] > 0:
                        self.sell(
                            self.getdatabyname(pair[0]),
                            size=abs(self.traded_quantity[pair][0]),
                            exectype=bt.Order.Market,
                        )
                        self.buy(
                            self.getdatabyname(pair[1]),
                            size=abs(self.traded_quantity[pair][1]),
                            exectype=bt.Order.Market,
                        )
                    else:
                        self.buy(
                            self.getdatabyname(pair[0]),
                            size=abs(self.traded_quantity[pair][0]),
                            exectype=bt.Order.Market,
                        )
                        self.sell(
                            self.getdatabyname(pair[1]),
                            size=abs(self.traded_quantity[pair][1]),
                            exectype=bt.Order.Market,
                        )
                    pairs_to_remove.append(pair)

        for pair in pairs_to_remove:
            self.traded_pairs.remove(pair)
            del self.traded_quantity[pair]

    def distance(self, price_a, price_b):
        norm_a = np.array(price_a) / price_a[0]
        norm_b = np.array(price_b) / price_b[0]
        return sum((norm_a - norm_b) ** 2)

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

        # Calculate liquidity (volume * adj_close) for the most recent date
        most_recent_date = df["date"].max()
        most_recent_data = df[df["date"] == most_recent_date].copy()
        most_recent_data["liquidity"] = (
                most_recent_data["volume"] * most_recent_data["adj_close"]
        )

        # List the symbols by decreasing liquidity
        sorted_liquidity = most_recent_data.sort_values(by="liquidity", ascending=False)[
            ["symbol", "liquidity"]
        ]
        most_liq_symbols = sorted_liquidity["symbol"].tolist()[:1200]
        df = df[df["symbol"].isin(most_liq_symbols)]

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
        for symbol in pivot_df.columns.levels[1]:
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

    backtest_engine = BacktestingEngine(trade_config)
    # backtest_engine.execute_all(PairsTradingStrategy, process=preprocess_df)
    backtest_engine.run_rolling_window(PairsTradingStrategy, process=preprocess_df)