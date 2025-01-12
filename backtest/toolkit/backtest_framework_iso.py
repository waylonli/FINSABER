import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from backtest.strategy.timing_iso.base_strategy_iso import BaseStrategyIso

class BacktestFramework:
    def __init__(self, initial_cash=100000, risk_free_rate=0.0, commission_per_share=0.0049, min_commission=0.99):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.portfolio = {}
        self.history = []
        self.risk_free_rate = risk_free_rate
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

    def load_data(self, data):
        if isinstance(data, str):
            with open(data, 'rb') as file:
                self.data = pickle.load(file)
        elif isinstance(data, dict):
            self.data = data
        else:
            raise ValueError("Data format not supported.")

    def calculate_commission(self, quantity):
        commission = abs(quantity) * self.commission_per_share
        return max(commission, self.min_commission)

    def buy(self, date, ticker, price, quantity):
        cost = price * quantity
        commission = self.calculate_commission(quantity)
        total_cost = cost + commission
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

    def evaluate(self):
        final_value = self.cash + sum(
            [self.portfolio[ticker]['quantity'] * self.data[list(self.data.keys())[-1]]['price'][ticker]
             for ticker in self.portfolio])

        returns = [
            trade['price'] * trade['quantity'] * (1 if trade['type'] == 'sell' else -1) / self.initial_cash
            for trade in self.history
        ]

        daily_returns = pd.Series(returns).pct_change().dropna()

        total_return = (final_value / self.initial_cash) - 1
        annual_return = (1 + total_return) ** (252 / len(self.data)) - 1
        annual_volatility = daily_returns.std() * np.sqrt(252)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0

        sharpe_ratio = (daily_returns.mean() * 252 - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        sortino_ratio = (daily_returns.mean() * 252 - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        total_commission = sum([trade['commission'] for trade in self.history])

        return {
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_commission': total_commission
        }

    def plot(self):
        dates = [trade['date'] for trade in self.history]
        equity_curve = [
            self.initial_cash + sum(
                [
                    trade['price'] * trade['quantity'] * (1 if trade['type'] == 'sell' else -1)
                    for trade in self.history[:i+1]
                ]
            ) for i in range(len(self.history))
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(dates, equity_curve, label="Equity Curve")
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.show()

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