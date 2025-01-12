# https://paperswithbacktest.com/paper/momentum-effects-in-country-equity-indexes#top
import backtrader as bt
import pandas as pd
from tqdm import tqdm
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.backtest_engine import BacktestingEngine

class MomentumFactorEffectinStocks(BaseStrategy):
    params = (
        ("momentum_period", 21),  # 1 months of daily data
        ("skip_period", 21),  # Skip the most recent month
        ("quantile", 5),  # Number of quantiles
        ("leverage", 0.9),
        ("total_days", 0),
    )

    def __init__(self):
        super().__init__()
        self.log_data = []  # Store portfolio values for logging
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
        # Calculate 12-month return excluding the most recent month
        perf = {}
        for d in self.datas:
            if len(d.close) < self.params.momentum_period:
                continue
            if d.close[0] == d.close[-2]:  # Is tradable
                continue
            perf[d._name] = (
                d.close[-self.params.skip_period]
                / d.close[-self.params.momentum_period]
            ) - 1
        # Select stocks based on quantile
        if len(perf) < self.params.quantile:
            return
        sorted_by_perf = sorted(perf, key=perf.get)
        quantile = int(len(sorted_by_perf) / self.params.quantile)
        long_stocks = sorted_by_perf[-quantile:]
        short_stocks = sorted_by_perf[:quantile]
        nb_stocks = len(long_stocks) + len(short_stocks)
        for d in self.datas:
            if (
                d._name not in long_stocks + short_stocks
                and self.broker.getposition(d).size != 0
            ):
                self.order_target_percent(d, target=0)
            elif d._name in short_stocks:
                self.order_target_percent(
                    d,
                    target=-1 / nb_stocks * self.params.leverage,
                )
            elif d._name in long_stocks:
                self.order_target_percent(
                    d,
                    target=1 / nb_stocks * self.params.leverage,
                )

    def get_latest_positions(self):
        positions = {
            data._name: self.broker.getposition(data).size for data in self.datas
        }
        return positions



if __name__ == "__main__":
    def preprocess_df(df):
        df["date"] = pd.to_datetime(df.index)
        df.reset_index(drop=True, inplace=True)

        pivot_df = pd.pivot_table(
            df,
            index="date",
            columns="symbol",
            values=["open", "high", "low", "close"],
            aggfunc="first",
        )
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
    }
    operator = BacktestingEngine(trade_config)
    # operator.execute_all(MomentumFactorEffectinStocks, process=preprocess_df)
    operator.run_rolling_window(MomentumFactorEffectinStocks, process=preprocess_df)