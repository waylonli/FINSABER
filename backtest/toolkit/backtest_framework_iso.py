import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from backtest.strategy.timing_iso.base_strategy_iso import BaseStrategyIso

class BacktestFrameworkIso:
    def __init__(self, initial_cash=100000, risk_free_rate=0.0, commission_per_share=0.0049, min_commission=0.99):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.portfolio = {}
        self.history = []
        self.risk_free_rate = risk_free_rate
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.data = None

    def load_backtest_data(self, data: dict or str, start_date: datetime = None, end_date: datetime = None):
        if isinstance(data, str):
            with open(data, 'rb') as file:
                self.data = pickle.load(file)
        elif isinstance(data, dict):
            self.data = data
        else:
            raise ValueError("Data format not supported.")

        if start_date is not None and end_date is not None:
            self.data = {d: data[d] for d in data if pd.to_datetime(start_date).date() <= d <= pd.to_datetime(end_date).date()}


    def calculate_commission(self, quantity):
        commission = abs(quantity) * self.commission_per_share
        return max(commission, self.min_commission)

    def buy(self, date, ticker, price, quantity):
        if quantity >= 0:
            cost = price * quantity
            commission = self.calculate_commission(quantity)
            total_cost = cost + commission
        elif quantity == -1:
            # Buy all available cash, taking into account commission
            total_cost = self.cash
            commission = self.calculate_commission(int(total_cost / price))
            total_cost -= commission
            quantity = int(total_cost / price)
            total_cost = price * quantity + commission

        if self.cash >= total_cost:
            self.cash -= total_cost
            if ticker in self.portfolio:
                self.portfolio[ticker]['quantity'] += quantity
            else:
                self.portfolio[ticker] = {'quantity': quantity, 'price': price}
            self.history.append({'date': date, 'ticker': ticker, 'type': 'buy', 'price': price, 'quantity': quantity, 'commission': commission})
        else:
            print(f"Insufficient cash to buy {quantity} of {ticker} on {date}")

    def sell(self, date, ticker, price, quantity):
        if ticker in self.portfolio and self.portfolio[ticker]['quantity'] >= quantity:
            revenue = price * quantity
            commission = self.calculate_commission(quantity)
            net_revenue = revenue - commission
            self.cash += net_revenue
            self.portfolio[ticker]['quantity'] -= quantity
            if self.portfolio[ticker]['quantity'] == 0:
                del self.portfolio[ticker]
            self.history.append({'date': date, 'ticker': ticker, 'type': 'sell', 'price': price, 'quantity': quantity, 'commission': commission})
        else:
            print(f"Insufficient holdings to sell {quantity} of {ticker} on {date}")

    def run(self, strategy: BaseStrategyIso):
        for date, data in self.data.items():
            prices = data['price']
            strategy.on_data(date, prices, self)
            strategy.update_info(date, prices, self)
            # import pdb; pdb.set_trace()
        # if there are any remaining holdings, sell them all at the last date
        try:
            last_date = list(self.data.keys())[-1]
        except:
            import pdb; pdb.set_trace()
        for ticker in list(self.portfolio.keys()):
            price = self.data[last_date]['price'][ticker]
            quantity = self.portfolio[ticker]['quantity']
            self.sell(last_date, ticker, price, quantity)

    def evaluate(self, strategy: BaseStrategyIso):
        final_value = self.cash + sum(
            [self.portfolio[ticker]['quantity'] * self.data[list(self.data.keys())[-1]]['price'][ticker]
             for ticker in self.portfolio])

        assert len(strategy.equity) > 1, "Equity data is missing."
        daily_returns = pd.Series([strategy.equity[i] / strategy.equity[i - 1] - 1
                                   for i in range(1, len(strategy.equity))])

        total_return = (final_value / self.initial_cash) - 1
        annual_return = (1 + total_return) ** (252 / len(self.data)) - 1
        annual_volatility = daily_returns.std() * np.sqrt(252)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0

        sharpe_ratio = (daily_returns.mean() * 252 - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        sortino_ratio = (daily_returns.mean() * 252 - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        total_commission = sum([trade['commission'] for trade in self.history])

        # Calculate maximum drawdown
        equity_series = pd.Series(strategy.equity)
        running_max = equity_series.cummax()
        drawdowns = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdowns.min()) * 100

        return {
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_commission': total_commission,
            'max_drawdown': max_drawdown
        }

    # def plot(self):
    #     dates = [trade['date'] for trade in self.history]
    #     equity_curve = [
    #         self.initial_cash + sum(
    #             [
    #                 trade['price'] * trade['quantity'] * (1 if trade['type'] == 'sell' else -1)
    #                 for trade in self.history[:i+1]
    #             ]
    #         ) for i in range(len(self.history))
    #     ]
    #
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(dates, equity_curve, label="Equity Curve")
    #     plt.title("Equity Curve")
    #     plt.xlabel("Date")
    #     plt.ylabel("Equity")
    #     plt.legend()
    #     plt.show()
    #     import pdb; pdb.set_trace()

    def reset(self):
        self.cash = self.initial_cash
        self.portfolio = {}
        self.history = []
        self.data = None

# class SampleStrategy(BaseStrategyIso):
#     def on_data(self, date, prices, framework):
#         for ticker, price in prices.items():
#             if price < 500:  # Example buy condition
#                 framework.buy(date, ticker, price, 10)
#             elif price > 1000:  # Example sell condition
#                 framework.sell(date, ticker, price, 10)
#
# # Example usage:
# data = pickle.load(open("data/finmem_data/03_model_input/synthetic_dataset.pkl", "rb"))
# framework = BacktestFramework()
# framework.load_data(data)
# strategy = SampleStrategy()
# framework.run(strategy)
# metrics = framework.evaluate()
# print(metrics)
# framework.plot()