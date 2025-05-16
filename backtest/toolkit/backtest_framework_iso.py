import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
# from backtest.data_util import BacktestDataset
# from backtest.strategy.timing_llm.base_strategy_iso import BaseStrategyIso

class FINSABERFrameworkHelper:
    def __init__(self, initial_cash=100000, risk_free_rate=0.0, commission_per_share=0.0049, min_commission=0.99, max_commission_rate=0.01):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.portfolio = {}
        self.history = []
        self.risk_free_rate = risk_free_rate
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.max_commission_rate = max_commission_rate
        self.data_loader = None

    def load_backtest_data(
            self,
            data,
            start_date: datetime = None,
            end_date: datetime = None
    ):
        if start_date is not None and end_date is not None:
            self.data_loader = data.get_subset_by_time_range(start_date, end_date)
        else:
            self.data_loader = data

        return True if self.data_loader is not None else False

    def load_backtest_data_single_ticker(
            self,
            data,
            ticker: str,
            start_date: datetime = None,
            end_date: datetime = None,
    ):
        if start_date is not None and end_date is not None:
            self.data_loader = data.get_ticker_subset_by_time_range(ticker, start_date, end_date)
        else:
            self.data_loader = data

        return True if self.data_loader is not None else False


    def calculate_commission(self, quantity, price):
        commission = abs(quantity) * self.commission_per_share
        txn_amount = abs(quantity * price)
        return max(self.min_commission, min(commission, txn_amount * self.max_commission_rate))

    def buy(self, date, ticker, price, quantity):
        if quantity >= 0:
            cost = price * quantity
            commission = self.calculate_commission(quantity, price)
            total_cost = cost + commission
        elif quantity == -1:
            # Buy all available cash, taking into account commission
            total_cost = self.cash
            commission = self.calculate_commission(int(total_cost / price), price)
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
            commission = self.calculate_commission(quantity, price)
            net_revenue = revenue - commission
            self.cash += net_revenue
            self.portfolio[ticker]['quantity'] -= quantity
            if self.portfolio[ticker]['quantity'] == 0:
                del self.portfolio[ticker]
            self.history.append({'date': date, 'ticker': ticker, 'type': 'sell', 'price': price, 'quantity': quantity, 'commission': commission})
        else:
            print(f"Insufficient holdings to sell {quantity} of {ticker} on {date}")

    def run(self, strategy, delist_check=True):
        date_range = self.data_loader.get_date_range()
        last_data_date = date_range[-1]
        if delist_check:
            end_date_year = last_data_date.year
            all_expected_trading_days = pd.bdate_range(start=f"{end_date_year}-01-01", end=f"{end_date_year}-12-31")
            last_expected_date = all_expected_trading_days[-1].date()
            print(f"Last expected date: {last_expected_date}")
            if last_data_date < last_expected_date:
                print(f"Current symbol appears to be delisted on {last_data_date}, adjust the end date for 7 days ahead announcement.")
                date_range = [d for d in date_range if d <= (last_data_date - pd.Timedelta(days=7))] # remove the last 7 days for delisting announcement

        if len(date_range) < 21:
            print(f"Not enough data for backtesting. Only {len(date_range)} days available.")
            return False

        for date in date_range:
            status = strategy.on_data(date, self.data_loader.get_data_by_date(date), self)
            strategy.update_info(date, self.data_loader, self)

            if status == "done":
                break

        # if there are any remaining holdings, sell them all at the last date
        for ticker in list(self.portfolio.keys()):
            price = self.data_loader.get_ticker_price_by_date(ticker, date_range[-1])
            quantity = self.portfolio[ticker]['quantity']
            self.sell(date_range[-1], ticker, price, quantity)

        return True

    def evaluate(self, strategy):
        final_value = self.cash + sum(
            [self.portfolio[ticker]['quantity'] * self.data_loader.get_ticker_price_by_date(ticker, self.data_loader.get_date_range()[-1])
             for ticker in self.portfolio])

        assert len(strategy.equity) > 1, "Equity data is missing."
        daily_returns = pd.Series([strategy.equity[i] / strategy.equity[i - 1] - 1
                                   for i in range(1, len(strategy.equity))])

        total_return = (final_value / self.initial_cash) - 1
        annual_return = (1 + total_return) ** (252 / len(self.data_loader.get_date_range())) - 1
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
        self.data_loader = None

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
# framework = FINSABERBtFrameworkHelper()
# framework.load_data(data)
# strategy = SampleStrategy()
# framework.run(strategy)
# metrics = framework.evaluate()
# print(metrics)
# framework.plot()