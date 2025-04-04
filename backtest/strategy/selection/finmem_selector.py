from backtest.strategy.selection import BaseSelector

class FinMemSelector(BaseSelector):
    def __init__(self):
        pass

    def select(self, *args, **kwargs):
        return ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]