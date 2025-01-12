import os
import backtrader as bt
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from backtest.toolkit.trade_config import TradeConfig
from backtest.toolkit.operation_utils import add_tickers_data, get_tickers_price
import warnings
import pickle
import matplotlib.pyplot as plt
from backtest.toolkit import metrics

warnings.filterwarnings("ignore")

# TODO wandb support
# TODO rolling window backtesting

class BacktestingEngine:
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
        Call execute_iter or execute_all for each rolling window
        :param strategy: The strategy to execute
        :param process: The function to process the data
        :param kwargs: Additional arguments for the strategy
        """
        # divide the date into rolling windows
        rolling_window_size = self.trade_config.rolling_window_size # in years
        rolling_window_step = self.trade_config.rolling_window_step # in years


        # e.g. 2000-01-01 to 2005-01-01, rolling_window_size=2, rolling_window_step=1, then the rolling windows are:
        # 2000-01-01 to 2002-01-01, 2001-01-01 to 2003-01-01, 2002-01-01 to 2004-01-01, 2003-01-01 to 2005-01-01
        date_from = pd.to_datetime(self.trade_config.date_from)
        date_to = pd.to_datetime(self.trade_config.date_to)

        # check selection strategy
        if self.trade_config.tickers == "all" and self.trade_config.selection_strategy.startswith("random"):
            num_tickers = int(self.trade_config.selection_strategy.split(":")[1])
            tickers_data = get_tickers_price("all", return_original=True)
            # select tickers that have data for the entire period
            # excluding weekends and holidays
            total_years = (date_to - date_from).days // 365
            tickers_data = tickers_data.groupby("symbol").filter(lambda x: x.shape[0] >= total_years * 252)
            print(f"Number of tickers with data for the entire period: {tickers_data['symbol'].nunique()}")
            # set random seed
            if os.path.exists(os.path.join(self.trade_config.log_base_dir, f"random_{num_tickers}", f"random_{num_tickers}_symbols.txt")):
                with open(os.path.join(self.trade_config.log_base_dir, f"random_{num_tickers}", f"random_{num_tickers}_symbols.txt"), "r") as f:
                    tickers = f.read().splitlines()
            else:
                np.random.seed(42)
                tickers = list(np.random.choice(tickers_data["symbol"].unique(), num_tickers, replace=False))
                os.makedirs(os.path.join(self.trade_config.log_base_dir, f"random_{num_tickers}"), exist_ok=True)
                with open(os.path.join(self.trade_config.log_base_dir, f"random_{num_tickers}", f"random_{num_tickers}_symbols.txt"), "w") as f:
                    f.write("\n".join(tickers))

            # print(f"Selected tickers: {tickers}")
            self.trade_config.tickers = tickers

        rolling_windows = []
        while date_from + pd.DateOffset(years=rolling_window_size) <= date_to:
            rolling_windows.append((date_from, date_from + pd.DateOffset(years=rolling_window_size)))
            date_from += pd.DateOffset(years=rolling_window_step)

        eval_metrics = {}
        windows_loop = tqdm(rolling_windows)

        for window in windows_loop:
            windows_loop.set_description(f"Processing window {window[0].strftime('%Y')} to {window[1].strftime('%Y')}")
            test_config = self.trade_config.to_dict()
            test_config["date_from"] = window[0].strftime("%Y-%m-%d")
            test_config["date_to"] = window[1].strftime("%Y-%m-%d")

            eval_metrics[f"{window[0].strftime('%Y-%m-%d')}_{window[1].strftime('%Y-%m-%d')}"] \
                = self.execute_iter(strategy, process, test_config=test_config, **kwargs)

        # export the evaluation metrics
        if self.trade_config.save_results:
            output_dir = os.path.join(self.trade_config.log_base_dir, self.trade_config.selection_strategy.replace(":", "_"), strategy.__name__)
            filename = f"{self.trade_config.date_from}_{self.trade_config.date_to}.pkl" if self.trade_config.result_filename is None else self.trade_config.result_filename
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, filename), "wb") as f:
                pickle.dump(eval_metrics, f)
            f.close()

        return eval_metrics

    def execute_iter(self, strategy: bt.Strategy, process: callable = None, test_config: dict = None, **kwargs):
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

        tickers_loop = test_config.tickers if test_config.tickers != "all" else get_tickers_price("all")["symbol"].unique()

        for ticker in tickers_loop:
            cerebro = bt.Cerebro()

            pd_data = get_tickers_price(ticker, date_from=test_config.date_from, date_to=test_config.date_to)
            train_data = None

            for additional_arg in kwargs:
                # if it is callable, call it
                if callable(kwargs[additional_arg]):
                    kwargs[additional_arg] = kwargs[additional_arg](pd_data)

            # if the model needs to be trained, set the training data that are not used for backtesting
            if "train_period" in vars(strategy.params).keys():
                if not strategy.params.train_period % 252 == 0:
                    raise ValueError("train_period must be a multiple of 252")

                train_year = strategy.params.train_period // 252
                train_data = get_tickers_price(
                    ticker,
                    date_from=(pd.to_datetime(test_config.date_from) - pd.DateOffset(years=train_year)).strftime("%Y-%m-%d"),
                    date_to=(pd.to_datetime(test_config.date_from) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
                )

                kwargs["train_data"] = train_data

            # skip if no data or data is not start from January
            if (pd_data is None or pd_data.index.min().month != 1) and ("cherry_pick" not in test_config.selection_strategy):
                # print(f"No data in the period {test_config.date_from} to {test_config.date_to} for {ticker}")
                continue

            # skip if no enough data for training
            if train_data is not None:
                if train_data.index.min().year > pd.to_datetime(test_config.date_from).year - train_year:
                    # print(f"Train data for {ticker} is not enough at year {pd.to_datetime(test_config.date_from).year}")
                    continue

            add_tickers_data(cerebro, pd_data)

            # Add a strategy
            cerebro.addstrategy(strategy, total_days=len(set(pd_data.index.tolist())), **kwargs)

            # Set our desired cash start
            cerebro.broker.setcash(test_config.cash)
            commission_scheme = USStockCommission()
            cerebro.broker.addcommissioninfo(commission_scheme)
            cerebro.broker.set_shortcash(False)

            # Add observers
            cerebro.addobserver(bt.observers.Value)

            # Add analyzers for Sharpe Ratio and Drawdown
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe', riskfreerate=test_config.risk_free_rate, timeframe=bt.TimeFrame.Days, annualize=True)
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

        if "cherry_pick" in test_config.selection_strategy and test_config.save_results:
            # store the results using pickle
            output_dir = os.path.join(test_config.log_base_dir, test_config.selection_strategy.replace(":", "_"), strategy.__name__)
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
                trades.append([trade.open_datetime().date(), trade.close_datetime().date(), trade.price, trade.pnl, trade.pnlcomm])
            trades_df = pd.DataFrame(trades, columns=['Open Date', 'Close Date', "Price", 'Profit/Loss',
                                                      'PnL (incl. commission)'])
            print("-" * 50)
            print("Trades:")
            print(tabulate(trades_df, headers='keys', tablefmt='psql'))

        if not test_config.silence:
            print("="*50)

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
        # Calculate commission based on shares
        commission = abs(size) * self.p.commission_per_share

        # Ensure the commission is at least the minimum order commission
        return max(commission, self.p.min_commission)