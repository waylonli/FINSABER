import backtrader as bt

from preliminary.strategy.timing.base_strategy import BaseStrategy
from preliminary.backtest_engine import BacktestingEngine

class TrendFollowingStrategy(BaseStrategy):
    params = (
        ("leverage", 5),
        ("total_days", 0),
        ("risk_factor", 0.002),  # 20 basis points
        ("atr_period", 20),
        ("ma_short", 50),
        ("ma_long", 100),
        ("breakout_period", 50),
        ("atr_multiplier", 3),
        ("total_days", 0),
    )

    def __init__(self):
        super().__init__()
        self.log_data = []
        self.inds = dict()
        for d in self.datas:
            self.inds[d] = dict()
            self.inds[d]["ma_short"] = bt.indicators.SimpleMovingAverage(
                d.close, period=self.params.ma_short
            )
            self.inds[d]["ma_long"] = bt.indicators.SimpleMovingAverage(
                d.close, period=self.params.ma_long
            )
            self.inds[d]["atr"] = bt.indicators.AverageTrueRange(
                d, period=self.params.atr_period
            )
            self.inds[d]["highest"] = bt.indicators.Highest(
                d.close, period=self.params.breakout_period
            )
            self.inds[d]["lowest"] = bt.indicators.Lowest(
                d.close, period=self.params.breakout_period
            )
            self.inds[d]["entry_high"] = None
            self.inds[d]["entry_low"] = None

    def next(self):
        self.log_data.append(
            {
                "date": self.datas[0].datetime.date(0).isoformat(),
                "value": self.broker.getvalue(),
            }
        )

        for d in self.datas:
            if not self.is_tradable(d):
                continue
            if len(d) < max(
                self.params.ma_long, self.params.breakout_period, self.params.atr_period
            ):
                continue

            pos = self.getposition(d).size
            ma_short = self.inds[d]["ma_short"][0]
            ma_long = self.inds[d]["ma_long"][0]
            close = d.close[0]
            atr = self.inds[d]["atr"][0]
            highest = self.inds[d]["highest"][-1]
            lowest = self.inds[d]["lowest"][-1]

            if pos == 0:
                if ma_short > ma_long and close >= highest:
                    size = self.calculate_position_size(d, atr)
                    if size > 0:
                        self.buy(data=d, size=size)
                        self.inds[d]["entry_high"] = close
                elif ma_short < ma_long and close <= lowest:
                    size = self.calculate_position_size(d, atr)
                    if size > 0:
                        self.sell(data=d, size=size)
                        self.inds[d]["entry_low"] = close

            elif pos > 0:
                self.inds[d]["entry_high"] = (
                    max(self.inds[d]["entry_high"], close)
                    if self.inds[d]["entry_high"]
                    else close
                )
                if (
                    close
                    <= self.inds[d]["entry_high"] - self.params.atr_multiplier * atr
                ):
                    self.close(data=d)
                    self.inds[d]["entry_high"] = None

            elif pos < 0:
                self.inds[d]["entry_low"] = (
                    min(self.inds[d]["entry_low"], close)
                    if self.inds[d]["entry_low"]
                    else close
                )
                if (
                    close
                    >= self.inds[d]["entry_low"] + self.params.atr_multiplier * atr
                ):
                    self.close(data=d)
                    self.inds[d]["entry_low"] = None

        self.post_next_actions()

    def calculate_position_size(self, data, atr):
        risk_per_trade = self.params.risk_factor * self.broker.getvalue()
        if atr < 0.001:  # Avoid division by zero
            return 0
        size = risk_per_trade / (atr * data.close[0])
        size *= self.params.leverage
        return size

    def is_tradable(self, data):
        if len(data.close) < 3:
            return False
        return data.close[0] != data.close[-2]

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
    operator.run_rolling_window(TrendFollowingStrategy)