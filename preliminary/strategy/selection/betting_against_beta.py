# https://paperswithbacktest.com/paper/betting-against-beta#backtest
import pandas as pd
import numpy as np
from tqdm import tqdm
from preliminary.strategy.timing.base_strategy import BaseStrategy
from preliminary.backtest_engine import BacktestingEngine

# Define the strategy
class BettingAgainstBetaStrategy(BaseStrategy):
    params = (
        ("leverage", 0.9),
        ("total_days", 0),
        ("period", 252),  # 1-year rolling window
        ("rebalance_months", 1),  # Rebalance every month
        ("quantile", 10),  # Decile ranking
        ("leverage_cap", 2.0),  # Cap on leverage
    )

    def __init__(self):
        super().__init__()
        self.log_data = []
        self.long_positions = []
        self.short_positions = []
        self.long_leverage = 1.0
        self.short_leverage = 1.0
        self.rebalance_counter = 0

        # Initialize a dictionary to store past prices
        self.prices = {d._name: [] for d in self.datas}
        print(self.prices)
        self.spy_data = self.getdatabyname('SPY')


    def next(self):
        self.log_data.append({
            "date": self.datas[0].datetime.date(0).isoformat(),
            "value": self.broker.getvalue(),
        })

        # Update price history
        for data in self.datas:
            self.prices[data._name].append(data.close[0])

        # Rebalance monthly
        if self.rebalance_counter % (21 * self.params.rebalance_months) == 0:
            self.rebalance()

        self.rebalance_counter += 1

        self.post_next_actions()

    def rebalance(self):
        # Ensure we have enough data
        if len(self.prices['SPY']) < self.params.period:
            return

        # Calculate beta for each stock
        betas = {}
        market_returns = pd.Series(self.prices['SPY'][-self.params.period:]).pct_change().dropna()

        for data in self.datas:
            if data._name == 'SPY':
                continue
            stock_prices = self.prices[data._name][-self.params.period:]
            if len(stock_prices) < self.params.period:
                continue
            if stock_prices[-1] == stock_prices[-3]:  # Is tradable
                continue
            stock_returns = pd.Series(stock_prices).pct_change().dropna()
            if len(stock_returns) != len(market_returns):
                continue  # Ensure both series are of equal length
            cov = np.cov(stock_returns, market_returns)[0][1]
            var = market_returns.var()
            beta = cov / var if var != 0 else 0
            betas[data._name] = beta

        if len(betas) < self.params.quantile:
            return

        # Sort stocks by beta
        sorted_betas = sorted(betas.items(), key=lambda x: x[1])
        quantile = int(len(sorted_betas) / self.params.quantile)

        # Select long and short positions
        n = 0
        self.long_positions = [symbol for symbol, beta in sorted_betas[(n * quantile):((n + 1) * quantile)]]
        self.short_positions = [symbol for symbol, beta in sorted_betas[-quantile:]]


        # Calculate average betas
        long_mean_beta = np.mean([betas[symbol] for symbol in self.long_positions])
        short_mean_beta = np.mean([betas[symbol] for symbol in self.short_positions])

        # Cap leverage
        self.long_leverage = min(self.params.leverage_cap, abs(1.0 / long_mean_beta)) if long_mean_beta != 0 else 1.0
        self.short_leverage = min(self.params.leverage_cap, abs(1.0 / short_mean_beta)) if short_mean_beta != 0 else 1.0

        # Adjust positions
        self.adjust_positions()

    def adjust_positions(self):
        # Close positions not in long or short lists
        for data in self.datas:
            position = self.getposition(data).size
            if position != 0 and data._name not in self.long_positions + self.short_positions:
                self.close(data)

        # Set new positions
        long_weight = self.params.leverage * self.long_leverage / len(self.long_positions)
        short_weight = -self.params.leverage * self.short_leverage / len(self.short_positions)

        for symbol in self.long_positions:
            data = self.getdatabyname(symbol)
            self.order_target_percent(data, long_weight)

        for symbol in self.short_positions:
            data = self.getdatabyname(symbol)
            self.order_target_percent(data, short_weight)

    def stop(self):
        self.pbar.close()

    def get_latest_positions(self):
        positions = {
            data._name: self.broker.getposition(data).size for data in self.datas
        }
        return positions


if __name__ == "__main__":
    def preprocess_df(stock_df):
        etf_df = pd.read_csv("data/etfs_daily.csv")
        if "date" not in stock_df.columns:
            stock_df["date"] = pd.to_datetime(stock_df.index)

        stock_df.reset_index(drop=True, inplace=True)

        adj_factor = stock_df["adj_close"] / stock_df["close"]
        stock_df["adj_open"] = stock_df["open"] * adj_factor
        stock_df["adj_high"] = stock_df["high"] * adj_factor
        stock_df["adj_low"] = stock_df["low"] * adj_factor

        # Calculate liquidity (volume * adj_close) for the most recent date
        most_recent_date = stock_df["date"].max()
        most_recent_data = stock_df[stock_df["date"] == most_recent_date].copy()
        most_recent_data["liquidity"] = (
                most_recent_data["volume"] * most_recent_data["adj_close"]
        )

        # List the symbols by decreasing liquidity
        sorted_liquidity = most_recent_data.sort_values(by="liquidity", ascending=False)[
            ["symbol", "liquidity"]
        ]
        most_liq_symbols = sorted_liquidity["symbol"].tolist()[:1200]
        stock_df = stock_df[stock_df["symbol"].isin(most_liq_symbols)]

        adj_factor = etf_df["adj_close"] / etf_df["close"]
        etf_df["adj_open"] = etf_df["open"] * adj_factor
        etf_df["adj_high"] = etf_df["high"] * adj_factor
        etf_df["adj_low"] = etf_df["low"] * adj_factor
        # Refactor as a data frame
        etf_df = etf_df[["date", "symbol", "adj_open", "adj_high", "adj_low", "adj_close"]]
        etf_df.rename(
            columns={
                "adj_open": "open",
                "adj_high": "high",
                "adj_low": "low",
                "adj_close": "close",
            },
            inplace=True,
        )
        # Filter for SPY
        spy_df = etf_df[etf_df['symbol'] == 'SPY'].copy()
        spy_df["date"] = pd.to_datetime(spy_df["date"])

        df = pd.concat([stock_df, spy_df], axis=0, ignore_index=True)

        # Pivot the data
        pivot_df = pd.pivot_table(
            df,
            index="date",
            columns="symbol",
            values=["open", "high", "low", "close"],
            aggfunc="first",
        )
        # Compute the full date range
        full_date_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
        pivot_df = pivot_df.reindex(full_date_range)

        trading_days = pd.bdate_range(start=full_date_range.min(), end=full_date_range.max())
        symbol_dfs = []

        for symbol in pivot_df.columns.levels[1]:
            symbol_df = pivot_df.xs(symbol, axis=1, level=1, drop_level=False).copy()
            symbol_df.columns = symbol_df.columns.droplevel(1)
            symbol_df.reset_index(inplace=True)
            symbol_df.rename(columns={"index": "date"}, inplace=True)
            symbol_df.set_index("date", inplace=True)
            symbol_df.ffill(inplace=True)
            symbol_df.bfill(inplace=True)
            # Filter out non-trading days
            symbol_df = symbol_df[symbol_df.index.isin(trading_days)]
            symbol_df["symbol"] = symbol
            symbol_dfs.append(symbol_df)

        return symbol_dfs


    trade_config = {
        "tickers": "all",
        "strategy_type": "selection",
        "silence": True,
    }
    operator = BacktestingEngine(trade_config)
    # operator.execute_all(BettingAgainstBetaStrategy, process=preprocess_df)
    operator.run_rolling_window(BettingAgainstBetaStrategy, process=preprocess_df)