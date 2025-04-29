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
        :param framework: Instance of the FINSABERBtFrameworkHelper
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

    def _adjust_size_for_commission(self, max_size):
        # Define the commission info to access the commission calculation
        commission_info = self.broker.getcommissioninfo(self.data)

        # Get the current cash and price
        cash = self.broker.get_cash()
        price = self.data.close[0]

        if price <= 1e-8:
            return 0

        # Adjust the max_size based on commission constraints
        while max_size > 0:
            # Estimate the commission for the current size
            commission_cost = commission_info._getcommission(size=max_size, price=price, pseudoexec=True)

            # Check if the cash can cover both stock purchase and commission cost
            if cash >= (max_size * price + commission_cost):
                return max_size
            max_size -= 1

        # Return 0 if cash cannot cover even the smallest possible order with commission
        return 0