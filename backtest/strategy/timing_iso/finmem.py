from backtest.strategy.timing_iso.base_strategy_iso import BaseStrategyIso


class FinMemStrategy(BaseStrategyIso):
    def __init__(self):
        pass

    def on_data(self, date, prices, framework):
        """
        This method should be implemented by subclasses.
        :param date: Current date of the backtest
        :param prices: Dictionary of ticker prices for the current date
        :param framework: Instance of the BacktestFramework
        """
        raise NotImplementedError("The on_data method must be implemented by the strategy.")

    def train_agents(self):
