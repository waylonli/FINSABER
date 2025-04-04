import logging
import colorlog

class BaseStrategyIso:
    def __init__(self):
        self.trades = []
        self.trade_returns = []
        self.buys = []
        self.sells = []
        self.peak_equity = 0
        self.equity = []
        self.equity_date = []
        # Set up the logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = 0
        handler = logging.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                "DEBUG": "grey",
                "INFO": "cyan",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def on_data(self, date, data_loader, framework):
        """
        This method should be implemented by subclasses.
        :param date: Current date of the backtest
        :param data_loader: Instance of the BacktestDataset
        :param framework: Instance of the BacktestFramework
        """
        raise NotImplementedError("The on_data method must be implemented by the strategy.")

    def update_info(self, date, data_loader, framework):
        self.equity_date.append(date)
        try:
            self.equity.append(framework.cash + sum(
                [framework.portfolio[ticker]['quantity'] * data_loader.get_ticker_price_by_date(ticker, date) for ticker in framework.portfolio]
            ))
        except Exception as e:
            self.logger.error(f"Error updating equity: {e}")

        # if len(self.equity) > 1 and self.equity[-1] != self.equity[-2]:
        #     self.logger.info(f"Equity updated for {date}: {self.equity[-1]}")

    def disable_logger(self):
        self.logger.disabled = True