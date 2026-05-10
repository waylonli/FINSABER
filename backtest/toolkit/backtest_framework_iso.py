import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from backtest.toolkit.execution import (
    adjusted_bar_price,
    apply_liquidity_cap,
    apply_slippage,
    calculate_commission,
    prior_volume_stats,
)
# from backtest.data_util import BacktestDataset
# from backtest.strategy.timing_llm.base_strategy_iso import BaseStrategyIso

class FINSABERFrameworkHelper:
    def __init__(
            self,
            initial_cash=100000,
            risk_free_rate=0.0,
            commission_per_share=0.0049,
            min_commission=0.99,
            max_commission_rate=0.01,
            execution_timing="next_open",
            slippage_perc=0.0,
            slippage_impact=0.0,
            liquidity_lookback_days=20,
            liquidity_min_history_days=1,
            liquidity_cap_pct=0.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.portfolio = {}
        self.history = []
        self.rejected_orders = []
        self.pending_orders = []
        self.risk_free_rate = risk_free_rate
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.max_commission_rate = max_commission_rate
        self.execution_timing = execution_timing
        self.slippage_perc = slippage_perc
        self.slippage_impact = slippage_impact
        self.liquidity_lookback_days = liquidity_lookback_days
        self.liquidity_min_history_days = liquidity_min_history_days
        self.liquidity_cap_pct = liquidity_cap_pct
        self.data_loader = None
        self.requested_end_date = None

    def load_backtest_data(
            self,
            data,
            start_date: datetime = None,
            end_date: datetime = None
    ):
        if start_date is not None and end_date is not None:
            self.data_loader = data.get_subset_by_time_range(start_date, end_date)
            self.requested_end_date = pd.to_datetime(end_date).date()
        else:
            self.data_loader = data
            self.requested_end_date = None

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
            self.requested_end_date = pd.to_datetime(end_date).date()
        else:
            self.data_loader = data
            self.requested_end_date = None

        return True if self.data_loader is not None else False


    def calculate_commission(self, quantity, price):
        return calculate_commission(
            quantity,
            price,
            commission_per_share=self.commission_per_share,
            min_commission=self.min_commission,
            max_commission_rate=self.max_commission_rate,
        )

    def buy(self, date, ticker, price, quantity):
        self._submit_order(date, ticker, "buy", quantity, reference_price=price)

    def sell(self, date, ticker, price, quantity):
        self._submit_order(date, ticker, "sell", quantity, reference_price=price)

    def _submit_order(self, date, ticker, side, quantity, reference_price=None):
        order = {
            "signal_date": date,
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "reference_price": reference_price,
        }
        if self.execution_timing == "same_close":
            self._execute_order(date, order, price_field="adjusted_close")
        else:
            self.pending_orders.append(order)

    def _get_execution_price(self, date, ticker, price_field, fallback=None):
        bar = self.data_loader.get_data_by_date(date).get("price", {}).get(ticker)
        if bar is None:
            return fallback
        return adjusted_bar_price(bar, price_field)

    def _resolve_quantity(self, date, side, ticker, requested_quantity, price, force=False):
        if requested_quantity == -1 and side == "buy":
            requested_quantity = int(self.cash / price)
        elif requested_quantity == -1 and side == "sell":
            requested_quantity = self.portfolio.get(ticker, {}).get("quantity", 0)
        else:
            requested_quantity = int(requested_quantity)

        if side == "sell":
            requested_quantity = min(requested_quantity, self.portfolio.get(ticker, {}).get("quantity", 0))

        volume_stats = prior_volume_stats(self.data_loader, ticker, date, self.liquidity_lookback_days)
        average_volume = volume_stats["average_volume"]
        volume_observations = volume_stats["observations"]
        if (
            self.liquidity_cap_pct > 0
            and not force
            and volume_observations < self.liquidity_min_history_days
        ):
            return 0, average_volume, volume_observations, "insufficient_liquidity_history"

        capped_quantity = requested_quantity if force else apply_liquidity_cap(
            requested_quantity,
            average_volume,
            self.liquidity_cap_pct,
            require_volume=self.liquidity_cap_pct > 0,
        )
        return capped_quantity, average_volume, volume_observations, None

    def _execute_order(self, date, order, price_field="adjusted_open", force=False):
        ticker = order["ticker"]
        side = order["side"]
        price = self._get_execution_price(date, ticker, price_field, fallback=order.get("reference_price"))
        if price is None or price <= 0:
            self.rejected_orders.append({**order, "execution_date": date, "reason": "invalid_price"})
            return False

        quantity, average_volume, volume_observations, reject_reason = self._resolve_quantity(
            date,
            side,
            ticker,
            order["quantity"],
            price,
            force=force,
        )
        if quantity <= 0:
            self.rejected_orders.append({**order, "execution_date": date, "reason": reject_reason or "zero_quantity"})
            return False

        def pricing_for(size):
            priced_fill, priced_slippage = apply_slippage(
                price,
                side,
                size,
                average_volume=average_volume,
                slippage_perc=self.slippage_perc,
                slippage_impact=self.slippage_impact,
            )
            return priced_fill, priced_slippage, self.calculate_commission(size, priced_fill)

        fill_price, slippage_cost, commission = pricing_for(quantity)

        if side == "buy":
            while quantity > 0 and quantity * fill_price + commission > self.cash:
                quantity -= 1
                fill_price, slippage_cost, commission = pricing_for(quantity)
            if quantity <= 0:
                self.rejected_orders.append({**order, "execution_date": date, "reason": "insufficient_cash"})
                return False
            total_cost = quantity * fill_price + commission
            self.cash -= total_cost
            self.portfolio.setdefault(ticker, {"quantity": 0, "price": fill_price})
            self.portfolio[ticker]["quantity"] += quantity
            self.portfolio[ticker]["price"] = fill_price
        else:
            if ticker not in self.portfolio or self.portfolio[ticker]["quantity"] < quantity:
                self.rejected_orders.append({**order, "execution_date": date, "reason": "insufficient_holdings"})
                return False
            self.cash += quantity * fill_price - commission
            self.portfolio[ticker]["quantity"] -= quantity
            if self.portfolio[ticker]["quantity"] == 0:
                del self.portfolio[ticker]

        self.history.append({
            "signal_date": order.get("signal_date"),
            "execution_date": date,
            "ticker": ticker,
            "type": side,
            "reference_price": price,
            "price": fill_price,
            "quantity": quantity,
            "commission": commission,
            "slippage_cost": slippage_cost,
            "average_volume": average_volume,
            "volume_observations": volume_observations,
            "participation_rate": abs(quantity) / average_volume if average_volume else None,
            "liquidity_cap_pct": self.liquidity_cap_pct,
        })
        return True

    def _execute_pending_orders(self, date):
        pending = self.pending_orders
        self.pending_orders = []
        for order in pending:
            self._execute_order(date, order, price_field="adjusted_open")

    def _reject_pending_orders(self, reason):
        for order in self.pending_orders:
            self.rejected_orders.append({**order, "execution_date": None, "reason": reason})
        self.pending_orders = []

    def run(self, strategy, delist_check=True):
        date_range = self.data_loader.get_date_range()
        last_data_date = date_range[-1]
        if delist_check and self.requested_end_date is not None:
            all_expected_trading_days = pd.bdate_range(start=date_range[0], end=self.requested_end_date)
            last_expected_date = all_expected_trading_days[-1].date()

            if last_data_date < (last_expected_date - pd.DateOffset(days=3)).date():
                print(f"Current symbol appears to be delisted on {last_data_date}, adjust the end date for 7 days ahead announcement.")
                date_range = [d for d in date_range if d <= (last_data_date - pd.Timedelta(days=7))] # remove the last 7 days for delisting announcement

        if len(date_range) < 21:
            print(f"Not enough data for backtesting. Only {len(date_range)} days available.")
            return False

        for date in date_range:
            self._execute_pending_orders(date)
            status = strategy.on_data(date, self.data_loader.get_data_by_date(date), self)
            strategy.update_info(date, self.data_loader, self)

            if status == "done":
                break

        # if there are any remaining holdings, sell them all at the last date
        for ticker in list(self.portfolio.keys()):
            quantity = self.portfolio[ticker]['quantity']
            self._execute_order(
                date_range[-1],
                {"signal_date": date_range[-1], "ticker": ticker, "side": "sell", "quantity": quantity},
                price_field="adjusted_close",
                force=True,
            )
        self._reject_pending_orders("no_future_bar")

        return True

    def evaluate(self, strategy):
        final_value = self.cash + sum(
            [self.portfolio[ticker]['quantity'] * self.data_loader.get_ticker_price_by_date(ticker, self.data_loader.get_date_range()[-1])
             for ticker in self.portfolio])

        if not len(strategy.equity) > 1:
            print("No equity data available for evaluation.")
            # return all 0s
            return {
                'final_value': final_value,
                'total_return': 0.0,
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'total_commission': 0.0,
                'total_slippage': 0.0,
                'total_trading_cost': 0.0,
                'max_drawdown': 0.0
            }



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
        total_slippage = sum([trade.get('slippage_cost', 0) for trade in self.history])

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
            'total_slippage': total_slippage,
            'total_trading_cost': total_commission + total_slippage,
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
        self.rejected_orders = []
        self.pending_orders = []
        self.data_loader = None
        self.requested_end_date = None

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
