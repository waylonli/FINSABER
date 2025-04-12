import backtrader as bt
import pandas as pd


class BaseStrategy(bt.Strategy):
    def __init__(self):
        super().__init__()
        self.trades = []
        self.trade_returns = []
        self.buys = []
        self.sells = []
        self.equity = []
        self.equity_date = []
        self.peak_equity = 0
        self.print_log = False
    
    def post_next_actions(self):
        current_equity = self.broker.getvalue()
        self.equity.append(current_equity)
        self.equity_date.append(pd.to_datetime(self.data.datetime.date(0)))
        self.peak_equity = max(self.peak_equity, current_equity)

    def notify_order(self, order: bt.Order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 等待订单完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'Buy，Price: %.2f, Share: %.2f, Cost: %.2f, Commission %.2f' %
                    (order.executed.price,
                     order.executed.size,
                     order.executed.value,
                     order.executed.comm),
                    self.print_log
                )
                self.buys.append(pd.to_datetime(self.data.datetime.date(0)))
            else:
                self.log(
                    'Sell，Price: %.2f, Share: %.2f, Gross: %.2f, Commission %.2f' %
                    (order.executed.price,
                     -order.executed.size,
                     order.executed.price * (-order.executed.size) - order.executed.value,
                     order.executed.comm),
                    self.print_log
                )
                self.sells.append(pd.to_datetime(self.data.datetime.date(0)))

        # 如果订单保证金不足，将不会执行，而是执行以下拒绝程序
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled(取消)/Margin()/Rejected(拒绝)")

        self.order = None

    def log(self, txt: str, print_log: bool = False, dt: bt.date = None):
        if print_log:
            dt = dt or self.data.datetime.date(0)
            print("%s, %s, " % (dt, txt))

    def notify_trade(self, trade: bt.Trade):
        if trade.isclosed:
            self.trades.append(trade)
            # assert trade.price is a non-zero value
            if trade.price != 0:
                returns = trade.pnlcomm / trade.price
            else:
                print("Detect zero trade price at %s for trade %s" % (self.data.datetime.date(0), trade))
                returns = 0
            self.trade_returns.append(returns)

    def _adjust_size_for_commission(self, max_size):
        # Define the commission info to access the commission calculation
        commission_info = self.broker.getcommissioninfo(self.data)

        # Get the current cash and price
        cash = self.broker.get_cash()
        price = self.data.close[0]

        if price <= 1e-8:
            return 0

        MAX_RETRIES = 100
        retries = 0
        # Adjust the max_size based on commission constraints
        while max_size > 0:
            # Estimate the commission for the current size
            commission_cost = commission_info._getcommission(size=max_size, price=price, pseudoexec=True)

            # Check if the cash can cover both stock purchase and commission cost
            if cash >= (max_size * price + commission_cost):
                return max_size

            max_size = int((cash - commission_cost) / price)


        # Return 0 if cash cannot cover even the smallest possible order with commission
        return 0