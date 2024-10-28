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

        # Schedule the monthly end/start operations
        self.add_timer(
            when=bt.Timer.SESSION_START,
            monthdays=list(range(1, 32)),
            monthcarry=True,
        )

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
    trade_config = {
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        "silence": True,
        "selection_strategy": "selected_5",
    }
    # trade_config = {
    #     "tickers": "all",
    #     "silence": False,
    # }

    operator = BacktestingEngine(trade_config)
    # operator.execute_iter(TurnOfTheMonthStrategy, process=preprocess_df, total_days=cal_total_days)
    operator.run_rolling_window(TurnOfTheMonthStrategy)