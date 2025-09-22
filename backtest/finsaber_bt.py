import copy
import logging
import os
import warnings
import pickle

import backtrader as bt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate

from backtest.toolkit.execution_helper import ExecutionContext
from backtest.data_util.market_data_provider import MarketDataProvider
from backtest.toolkit.trade_config import TradeConfig
from backtest.toolkit.operation_utils import add_tickers_data, get_tickers_price
from backtest.strategy.selection import *
from backtest.toolkit import metrics

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


# TODO wandb support
# TODO rolling window backtesting

class FINSABERBt:
    def __init__(
            self,
            config: dict,
    ):
        """
        :param config: The configuration for the trade operator
        """
        self.trade_config = TradeConfig.from_dict(config)

    def run_rolling_window(self, strategy: bt.Strategy, process: callable = None, **kwargs):
        """
        Call run_iterative_tickers or execute_all for each rolling window
        :param strategy: The strategy to execute
        :param process: The function to process the data
        :param kwargs: Additional arguments for the strategy
        """
        # divide the date into rolling windows
        rolling_window_size = self.trade_config.rolling_window_size  # in years
        rolling_window_step = self.trade_config.rolling_window_step  # in years

        # e.g. 2000-01-01 to 2005-01-01, rolling_window_size=2, rolling_window_step=1, then the rolling windows are:
        # 2000-01-01 to 2002-01-01, 2001-01-01 to 2003-01-01, 2002-01-01 to 2004-01-01, 2003-01-01 to 2005-01-01
        date_from = pd.to_datetime(self.trade_config.date_from)
        date_to = pd.to_datetime(self.trade_config.date_to)

        # check selection strategy
        if self.trade_config.setup_name in ["selected_4", "selected_5", "cherry_pick_both_finmem"]:
            stock_selector = FinMemSelector()
        else:
            # TODO: implement other selection strategies
            stock_selector = self.trade_config.selection_strategy

        rolling_windows = []
        while date_from + pd.DateOffset(years=rolling_window_size) <= date_to:
            rolling_windows.append((date_from, date_from + pd.DateOffset(years=rolling_window_size)))
            date_from += pd.DateOffset(years=rolling_window_step)

        eval_metrics = {}
        windows_loop = tqdm(rolling_windows)

        for window in windows_loop:
            windows_loop.set_description(f"Processing window {window[0].strftime('%Y')} to {window[1].strftime('%Y')}")

            self.trade_config.tickers = stock_selector.select(
                self.trade_config.data_loader,
                window[0].strftime("%Y-%m-%d"),
                window[1].strftime("%Y-%m-%d")
            )
            LOGGER.info(
                "Selected tickers for %s-%s: %s",
                window[0].strftime('%Y'),
                window[1].strftime('%Y'),
                self.trade_config.tickers,
            )

            test_config = self.trade_config.to_dict()
            test_config["date_from"] = window[0].strftime("%Y-%m-%d")
            test_config["date_to"] = window[1].strftime("%Y-%m-%d")

            eval_metrics[f"{window[0].strftime('%Y-%m-%d')}_{window[1].strftime('%Y-%m-%d')}"] \
                = self.run_iterative_tickers(strategy, process, test_config=test_config, **kwargs)

        # export the evaluation metrics
        if self.trade_config.save_results:
            output_dir = os.path.join(self.trade_config.log_base_dir, self.trade_config.setup_name.replace(":", "_"),
                                      strategy.__name__)
            filename = f"{self.trade_config.date_from}_{self.trade_config.date_to}.pkl" if self.trade_config.result_filename is None else self.trade_config.result_filename
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, filename), "wb") as f:
                pickle.dump(eval_metrics, f)
            f.close()

        return eval_metrics

    def run_iterative_tickers(self, strategy: bt.Strategy, process: callable = None, test_config: dict = None,
                              **kwargs):
        """
        Execute the strategy
        :param strategy: The strategy to execute
        :param process: The function to process the data
        :param test_config: The configuration if different from the global configuration
        :param kwargs: Additional arguments for the strategy
        """

        if test_config is None:
            test_config = self.trade_config
        else:
            test_config = TradeConfig.from_dict(test_config)

        eval_metrics = {}

        market_data = MarketDataProvider(
            adv_window=test_config.execution.liquidity.adv_window,
        )
        base_strategy_kwargs = dict(kwargs)

        tickers_loop = test_config.tickers if test_config.tickers != "all" else get_tickers_price("all")[
            "symbol"].unique()

        for ticker in tickers_loop:

            # print(f"Processing ticker {ticker}...")

            cerebro = bt.Cerebro()

            # Calculate required warmup days based on strategy parameters
            warmup_days = 30  # Default warmup
            if hasattr(strategy.params, 'long_window'):
                warmup_days = max(warmup_days, strategy.params.long_window + 10)  # Add buffer
            elif hasattr(strategy.params, 'short_window') and hasattr(strategy.params, 'long_window'):
                warmup_days = max(warmup_days, max(strategy.params.short_window, strategy.params.long_window) + 10)

            LOGGER.debug("Warmup for %s: %s days (%s)", ticker, warmup_days, strategy.__name__)
            pd_data = get_tickers_price(ticker, date_from=test_config.date_from, date_to=test_config.date_to,
                                        warmup_days=warmup_days)
            train_data = None

            strategy_kwargs = {}
            for key, value in base_strategy_kwargs.items():
                strategy_kwargs[key] = value(pd_data) if callable(value) else value

            if "prior_period" in vars(strategy.params).keys():
                if not strategy.params.prior_period % 252 == 0:
                    raise ValueError("prior_period must be a multiple of 252")

                prior_year = strategy.params.prior_period // 252
                prior_data = get_tickers_price(
                    ticker,
                    date_from=(pd.to_datetime(test_config.date_from) - pd.DateOffset(years=prior_year)).strftime(
                        "%Y-%m-%d"),
                    date_to=(pd.to_datetime(test_config.date_from) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
                )

                if prior_data is not None:
                    if prior_data.index.min().year > pd.to_datetime(test_config.date_from).year - prior_year:
                        LOGGER.warning(
                            "Insufficient prior data for %s in %s",
                            ticker,
                            pd.to_datetime(test_config.date_from).year,
                        )
                        continue
                else:
                    LOGGER.warning(
                        "No prior data for %s in %s",
                        ticker,
                        pd.to_datetime(test_config.date_from).year,
                    )
                    continue

            # if the model needs to be trained, set the training data that are not used for backtesting
            if "train_period" in vars(strategy.params).keys():
                if not strategy.params.train_period % 252 == 0:
                    raise ValueError("train_period must be a multiple of 252")

                train_year = strategy.params.train_period // 252
                train_data = get_tickers_price(
                    ticker,
                    date_from=(pd.to_datetime(test_config.date_from) - pd.DateOffset(years=train_year)).strftime(
                        "%Y-%m-%d"),
                    date_to=(pd.to_datetime(test_config.date_from) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
                )

                kwargs["train_data"] = train_data

            # skip if no data or data is not start from January
            if (pd_data is None or pd_data.index.min().month != 1 or len(pd_data) < 21) and (
                    "cherry_pick" not in test_config.setup_name):
                # print(f"No data in the period {test_config.date_from} to {test_config.date_to} for {ticker}")
                continue

            # skip if no enough data for training
            if train_data is not None:
                if train_data.index.min().year > pd.to_datetime(test_config.date_from).year - train_year:
                    LOGGER.warning(
                        "Insufficient training data for %s in %s",
                        ticker,
                        pd.to_datetime(test_config.date_from).year,
                    )
                    continue

            # detect if the stock is delisted in the middle of the period, if it is, assign 0 price to the missing dates
            # This indicates a complete loss of the stock
            end_date = pd.to_datetime(test_config.date_to) - pd.DateOffset(days=1)
            all_expected_trading_days = pd.bdate_range(start=pd_data.index.min(), end=end_date)
            last_expected_date = all_expected_trading_days[-1]
            last_data_date = pd_data.index.max()

            if last_data_date < last_expected_date - pd.DateOffset(days=3):
                # If the last data date is more than 3 days before the last expected date (avoid weekend or holidays), we assume the stock is delisted
                print(
                    f"{ticker} appears to be delisted on {last_data_date.strftime('%Y-%m-%d')}, applying 7 days delisting announcement period.")

                # remove the last 7 days of data
                pd_data = pd_data[pd_data.index <= last_data_date - pd.DateOffset(days=7)]

            # check again
            if ((pd_data is None or pd_data.index.min().month != 1 or len(pd_data) < 21) and (
                    "cherry_pick" not in test_config.setup_name)):
                LOGGER.warning(
                    "Skipping %s due to insufficient data between %s and %s",
                    ticker,
                    test_config.date_from,
                    test_config.date_to,
                )
                continue

            add_tickers_data(cerebro, pd_data)

            execution_context = ExecutionContext(
                config=copy.deepcopy(test_config.execution),
                market_data=market_data,
            )

            strategy_kwargs.update(
                {
                    "execution_context": execution_context,
                    "setup_name": test_config.setup_name,
                    "strategy_name": strategy.__name__,
                    "total_days": len(set(pd_data.index.tolist())),
                }
            )

            cerebro.addstrategy(
                strategy,
                **strategy_kwargs,
            )

            # Set our desired cash start
            cerebro.broker.setcash(test_config.cash)

            # #ORIGINAL
            # commission_scheme = USStockCommission()
            # cerebro.broker.addcommissioninfo(commission_scheme)

            cerebro.broker.set_shortcash(False)

            # Add observers
            cerebro.addobserver(bt.observers.Value)

            # Add analyzers for Sharpe Ratio and Drawdown
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe', riskfreerate=test_config.risk_free_rate,
                                timeframe=bt.TimeFrame.Days, annualize=True)
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='mydrawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='myreturns')
            # cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='myannualreturn')
            # cerebro.addanalyzer(bt.analyzers.VWR, _name='myvwr')  # Annualized volatility

            # Run over everything
            results = cerebro.run()
            strat = results[0]

            if not test_config.silence:
                # Print out the final result
                eval_metrics[ticker] = self._analyze_results(
                    strat,
                    test_config=test_config,
                    ticker=ticker,
                    print_trades_table=test_config.print_trades_table,
                )
            else:
                eval_metrics[ticker] = self._analyze_results(
                    strat,
                    test_config=test_config,
                    ticker=ticker,
                    print_trades_table=False,
                    print_annual_metrics=False,
                    print_details=test_config.print_trades_table
                )

            # Obtain the equity curve
            equity_with_time = pd.DataFrame(
                {
                    "datetime": strat.equity_date,
                    "equity": strat.equity
                }
            )

            eval_metrics[ticker]["equity_with_time"] = equity_with_time

            if not test_config.silence:
                # Plot the result
                plt.figure(figsize=(10, 6))
                plt.plot(equity_with_time["datetime"], equity_with_time["equity"], label="Equity Curve")
                plt.title(f"Equity Curve for {ticker}")
                plt.xlabel("Date")
                plt.ylabel("Equity")
                plt.legend()
                plt.show()

        if "cherry_pick" in test_config.setup_name and test_config.save_results:
            # store the results using pickle
            output_dir = os.path.join(test_config.log_base_dir, test_config.setup_name.replace(":", "_"),
                                      strategy.__name__)
            filename = f"{test_config.date_from}_{test_config.date_to}.pkl" if test_config.result_filename is None else test_config.result_filename
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, filename), "wb") as f:
                pickle.dump({f"{test_config.date_from}_{test_config.date_to}": eval_metrics}, f)
            f.close()
        return eval_metrics

    def _analyze_results(self,
                         strategy: bt.Strategy,
                         ticker: str,
                         test_config: TradeConfig,
                         print_details=True,
                         print_annual_metrics=True,
                         print_trades_table=False):

        if strategy is None:
            print("No strategy results to analyze")
            return None

        max_drawdown = strategy.analyzers.mydrawdown.get_analysis().max.drawdown
        total_return = strategy.broker.getvalue() / test_config.cash - 1
        total_return_cash = strategy.broker.getvalue() - test_config.cash
        annual_metrics = self._calculate_annualized_metrics(strategy, test_config=test_config)

        if print_details:
            print("\n" + "=" * 50)
            print(f"Period: {test_config.date_from} to {test_config.date_to}")
            print(f"Ticker: {ticker}")
            print("-" * 50)
            print(f"Initial cash: {test_config.cash}")
            print(f"Final cash: {strategy.broker.getvalue():.2f}")
            print(f"Total return (cash): {total_return_cash:.2f}")
            print(f"Total return (%): {total_return:.2%}")
            print(f"Max drawdown (%): {max_drawdown:.2f}%")
            print(f"Number of trades: {len(strategy.trades)}")

        if print_annual_metrics:
            print("-" * 50)
            print(f"Annual return: {annual_metrics['Annual Return']:.2%}")
            print(f"Annual volatility: {annual_metrics['Annual Volatility']:.2%}")
            print(f"Sharpe ratio: {annual_metrics['Sharpe Ratio']:.4f}")
            print(f"Sortino ratio: {annual_metrics['Sortino Ratio']:.4f}")

        if print_trades_table:
            trades = []
            for trade in strategy.trades:
                trades.append([trade.open_datetime().date(), trade.close_datetime().date(), trade.price, trade.pnl,
                               trade.pnlcomm])
            trades_df = pd.DataFrame(trades, columns=['Open Date', 'Close Date', "Price", 'Profit/Loss',
                                                      'PnL (incl. commission)'])
            print("-" * 50)
            print("Trades:")
            print(tabulate(trades_df, headers='keys', tablefmt='psql'))

        if not test_config.silence:
            print("=" * 50)

        return {
            'sharpe_ratio': annual_metrics['Sharpe Ratio'],
            'annual_return': annual_metrics['Annual Return'],
            'annual_volatility': annual_metrics['Annual Volatility'],
            'sortino_ratio': annual_metrics['Sortino Ratio'],
            'max_drawdown': max_drawdown,
            'total_return': total_return
        }

    def _calculate_annualized_metrics(
            self,
            strategy: bt.Strategy,
            test_config: TradeConfig):

        # Calculate the daily returns from the equity curve
        daily_returns = pd.Series(strategy.equity).pct_change().dropna()

        # average_daily_return = daily_returns.mean()
        # daily_risk_free_rate = (1 + test_config.risk_free_rate) ** (1 / (252)) - 1
        # excess_daily_return = average_daily_return - daily_risk_free_rate
        # self_calculate_sharpe_ratio = excess_daily_return / daily_returns.std() * np.sqrt(252)
        # print("Self calculated Sharpe ratio: ", self_calculate_sharpe_ratio)

        if not daily_returns.empty and daily_returns.any():
            if strategy.broker.getvalue() < 0:
                print("Negative value in equity curve")
                final_value = 0
            else:
                final_value = strategy.broker.getvalue()

            total_return = (final_value / test_config.cash) - 1
            total_periods = len(daily_returns)
            annual_return = (1 + total_return) ** (252 / total_periods) - 1
            # check if annual return is float
            try:
                assert isinstance(annual_return, float), f"Annual return is not float: {annual_return}"
            except AssertionError as e:
                print("value", strategy.broker.getvalue())
                print("cash", test_config.cash)
                print("total return", total_return)
                print("total periods", total_periods)
                print("annual return", annual_return)
                # print stock symbol
                print("stock symbol", strategy.datas[0]._name)
                raise e

            # Calculate annual volatility
            annual_volatility = metrics.calculate_annual_volatility(daily_returns)

            sortino_ratio = metrics.calculate_sortino_ratio(daily_returns, risk_free_rate=test_config.risk_free_rate)

            # Use the analyzer's Sharpe ratio if available
            sharpe_ratio = strategy.analyzers.mysharpe.get_analysis()['sharperatio']
        else:
            annual_return = annual_volatility = sharpe_ratio = sortino_ratio = 0

        return {
            "Annual Return": annual_return,
            "Annual Volatility": annual_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
        }


class USStockCommission(bt.CommInfoBase):
    params = (
        ('commission_per_share', 0.0049),  # $0.0049 per share
        ('min_commission', 0.99),  # Minimum $0.99 per order
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),  # Fixed commission per share
    )

    def _getcommission(self, size, price, pseudoexec):
        # Raw commission per share
        commission = abs(size) * self.p.commission_per_share

        # Transaction amount in $
        txn_amount = abs(size * price)

        # # DEBUG PRINT
        # print(f"[COMMISSION] size={size}, price={price}, raw_commission={commission}, "
        #       f"min={self.p.min_commission}, cap={1 * txn_amount}")

        # Apply both minimum and maximum constraints
        return min(max(commission, self.p.min_commission), 0.01 * txn_amount)  # 0.01 MAX


class USStockCommissionWithSlippage(USStockCommission):
    params = (
        # repeat the parent's items
        ('commission_per_share', 0.0049),
        ('min_commission', 0.99),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
        # add your own
        ('adv', 1.0),
        ('c', 10.0),
        ('a', 0.5),
    )

    def _getcommission(self, size, price, pseudoexec):
        commission = super()._getcommission(size, price, pseudoexec)
        adv = max(self.p.adv, 1.0)
        impact_ratio = abs(size) / adv
        slippage = abs(size) * price * (self.p.c * (impact_ratio ** self.p.a)) / 100.0
        return commission + slippage


class USStockCommissionWithRollingSlippage(bt.CommInfoBase):
    params = (
        ('commission_per_share', 0.0049),
        ('min_commission', 0.99),
        ('adv_lookup', None),  # dict {Timestamp: ADV}
        ('c', 10.0),  # slippage constant
        ('a', 0.5),  # slippage exponent
    )

    # The broker reference will be injected immediately after registration
    broker = None

    def _getcommission(self, size, price, pseudoexec):
        # ---------- base commission (no 1 % cap) ---------------------------
        commission = abs(size) * self.p.commission_per_share
        commission = max(commission, self.p.min_commission)

        # ---------- rolling-ADV slippage -----------------------------------
        if self.broker is None:  # sizing pass may hit first
            return commission  # can't compute slippage yet

        trade_date = pd.to_datetime(self.broker.datetime.date())
        adv = max(self.p.adv_lookup.get(trade_date, 1.0), 1.0)

        impact = abs(size) / adv
        slip_pct = self.p.c * (impact ** self.p.a) / 100.0
        slippage = abs(size) * price * slip_pct

        # Optional debug line
        print(f"[COM] {trade_date.date()} size={size} ADV={adv:.0f} "
              f"slip%={slip_pct * 100:.3f} fee=${commission + slippage:,.2f}")

        return commission + slippage
