# https://paperswithbacktest.com/paper/equity-returns-at-the-turn-of-the-month#top
from datetime import datetime, timedelta
import backtrader as bt
from preliminary.strategy.timing.base_strategy import BaseStrategy
import pandas as pd
from dotenv import load_dotenv
from preliminary.backtest_engine import BacktestingEngine
load_dotenv()

class TurnOfTheMonthStrategy(BaseStrategy):
    params = (
        ("before_end_of_month_days", 5),
        ("after_start_of_month_business_days", 3),
        ("total_days", 0),
    )

    def __init__(self):
        super().__init__()
        self.order = None
        self.days_held = 0
        self.sell_flag = False
        self.log_data = []  # Store portfolio values for backtest info
        # self.pbar = tqdm(total=self.params.total_days)

        # Schedule the monthly end/start operations
        self.add_timer(
            when=bt.Timer.SESSION_START,
            monthdays=list(range(1, 32)),
            monthcarry=True,
        )
        self.print_log = False

    def notify_timer(self, timer, when, *args, **kwargs):
        if self.is_third_business_day(when):
            self.order_target_percent(self.data, target=0)
        elif self.is_month_end(when):
            self.order_target_percent(self.data, target=1)

    def is_month_end(self, when):
        return (
            when + timedelta(days=self.params.before_end_of_month_days)
        ).month != when.month

    def is_third_business_day(self, when):
        business_day_count = 0
        current_day = datetime(when.year, when.month, 1)
        while current_day <= when:
            if current_day.weekday() < 5:  # Monday to Friday are business days
                business_day_count += 1
            if business_day_count == self.params.after_start_of_month_business_days:
                return current_day == when
            current_day += timedelta(days=1)
        return False

    def next(self):
        # self.pbar.update(1)
        self.log_data.append(
            {
                "date": self.datas[0].datetime.date(0).isoformat(),
                "value": self.broker.getvalue(),
            }
        )
        self.post_next_actions()


    def get_latest_positions(self):
        positions = {
            data._name: self.broker.getposition(data).size for data in self.datas
        }
        return positions



if __name__ == "__main__":
    def cal_total_days(df):
        if "date" in df.columns:
            # set as index if it is not
            df.set_index("date", inplace=True)
            df.index = pd.to_datetime(df.index)
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        return len(full_date_range)

    def preprocess_df(df):
        adj_factor = df["adj_close"] / df["close"]
        df["adj_open"] = df["open"] * adj_factor
        df["adj_high"] = df["high"] * adj_factor
        df["adj_low"] = df["low"] * adj_factor
        df = df[["symbol", "adj_open", "adj_high", "adj_low", "adj_close"]]
        df.rename(
            columns={
                "adj_open": "open",
                "adj_high": "high",
                "adj_low": "low",
                "adj_close": "close",
            },
            inplace=True,
        )
        columns_to_keep = [
            "symbol",
            "open",
            "high",
            "low",
            "close",
        ]
        df = df[columns_to_keep]

        indices_data = pd.read_csv("data/indices_daily.csv")

        mapping_etfs = {}
        for etf in df["symbol"].unique():
            mapping_etfs[etf] = "SPX"

        # the index is the date, make it as a column
        df.reset_index(inplace=True)
        indices_data.reset_index(inplace=True)

        df["date"] = pd.to_datetime(df["date"])
        indices_data["date"] = pd.to_datetime(indices_data["date"])

        
        frames = []
        for etf, index in mapping_etfs.items():
            # Get the ETF & Index data
            etf_data = df[df["symbol"] == etf]
            if etf_data.empty:
                raise ValueError(f"Data not found for {etf}")

            index_data = indices_data[indices_data["symbol"] == index]
            if index_data.empty:
                raise ValueError(f"Data not found for {index}")

            # Find the first overlapping date
            common_dates = etf_data["date"].isin(index_data["date"])
            first_common_date = etf_data.loc[common_dates, "date"].min()

            if pd.isnull(first_common_date):
                raise ValueError(f"No common date found for {etf} and {index}")

            etf_first_common = etf_data[etf_data["date"] == first_common_date]
            index_first_common = index_data[index_data["date"] == first_common_date]

            # Compute the adjustment factor (using closing prices for simplicity)
            adjustment_factor = (
                    etf_first_common["close"].values[0] / index_first_common["close"].values[0]
            )

            # Adjust index data before the first common date
            index_data_before_common = index_data[
                index_data["date"] < first_common_date
                ].copy()
            for column in ["open", "high", "low", "close"]:
                index_data_before_common.loc[:, column] *= adjustment_factor
            index_data_before_common.loc[:, "symbol"] = etf

            # Combine adjusted index data with ETF data
            combined_data = pd.concat([index_data_before_common, etf_data])
            frames.append(combined_data)

        # Concatenate all frames to form the final dataframe
        result_df = (
            pd.concat(frames).sort_values(by=["date", "symbol"]).reset_index(drop=True)
        )
        result_df.index = result_df["date"]
        return result_df

    trade_config = {
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "ADBE", "NFLX"],
        "silence": True,
        "strategy_type": "timing"
    }

    operator = BacktestingEngine(trade_config)
    # operator.execute_iter(TurnOfTheMonthStrategy, process=preprocess_df, total_days=cal_total_days)
    operator.run_rolling_window(TurnOfTheMonthStrategy, process=preprocess_df, total_days=cal_total_days)