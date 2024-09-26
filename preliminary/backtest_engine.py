import backtrader as bt
import numpy as np
import pandas as pd
from empyrical import max_drawdown, sharpe_ratio, annual_volatility
from tqdm import tqdm
from tabulate import tabulate
from preliminary.trade_config import TradeConfig
from preliminary.operation_utils import get_tickers_price, process_for_ff
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

class BacktestingEngine:
    def __init__(
            self,
            config: dict,
    ):
        """
        :param config: The configuration for the trade operator
        """
        self.trade_config = TradeConfig.from_dict(config)


    def execute_iter(self, strategy: bt.Strategy, process: callable = None, **kwargs):
        """
        Execute the strategy
        :param strategy: The strategy to execute
        :param process: The function to process the data
        :param kwargs: Additional arguments for the strategy
        """

        tickers_data = get_tickers_price(self.trade_config.tickers, self.trade_config.date_from, self.trade_config.date_to)
        eval_metrics = {}

        tickers_loop = tqdm(self.trade_config.tickers) if self.trade_config.tickers != "all" else tqdm(tickers_data["symbol"].unique(), desc=f"Executing strategy {strategy.__name__}")

        for ticker in tickers_loop:
            cerebro = bt.Cerebro()
            pd_data = tickers_data[tickers_data["symbol"] == ticker]

            if process is not None:
                pd_data = process(pd_data)

            data = bt.feeds.PandasData(
                dataname=pd_data,
                fromdate=pd.to_datetime(self.trade_config.date_from),
                todate=pd.to_datetime(self.trade_config.date_to)
            )

            cerebro.adddata(data)

            for additional_arg in kwargs:
                # if it is callable, call it
                if callable(kwargs[additional_arg]):
                    kwargs[additional_arg] = kwargs[additional_arg](pd_data)

            # Add a strategy
            cerebro.addstrategy(strategy, **kwargs)

            # Set our desired cash start
            cerebro.broker.setcash(self.trade_config.cash)
            cerebro.broker.setcommission(commission=self.trade_config.commission)
            cerebro.broker.set_slippage_perc(perc=self.trade_config.slippage_perc)

            # Add observers
            cerebro.addobserver(bt.observers.Value)

            # Add analyzers for Sharpe Ratio and Drawdown
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe', riskfreerate=self.trade_config.risk_free_rate, annualize=True)
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='mydrawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='myreturns')
            cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
            # cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='myannualreturn')
            # cerebro.addanalyzer(bt.analyzers.VWR, _name='myvwr')  # Annualized volatility

            # Run over everything
            results = cerebro.run()
            strat = results[0]

            # Print out the final result
            eval_metrics[ticker] = self.analyze_results(strat, ticker=ticker, print_trades_table=False)

            # Obtain the equity curve
            equity_with_time = pd.DataFrame(
                {
                    "datetime": strat.equity_date,
                    "equity": strat.equity
                }
            )

            eval_metrics[ticker]["equity_with_time"] = equity_with_time

            # Plot the result
            plt.figure(figsize=(10, 6))
            plt.plot(equity_with_time["datetime"], equity_with_time["equity"], label="Equity Curve")
            plt.title(f"Equity Curve for {ticker}")
            plt.xlabel("Date")
            plt.ylabel("Equity")
            plt.legend()
            plt.show()


        return eval_metrics


    def execute_all(self, strategy: bt.Strategy, process: callable = None, **kwargs):
        """
        Execute the strategy
        :param strategy: The strategy to execute
        :param process: The function to process the data
        :param kwargs: Additional arguments for the strategy
        """

        tickers_data = get_tickers_price(self.trade_config.tickers, self.trade_config.date_from, self.trade_config.date_to)

        pd_data = tickers_data

        if process is not None:
            pd_data = process(pd_data)

        cerebro = bt.Cerebro()

        if type(pd_data) == pd.DataFrame:
            data = bt.feeds.PandasData(
                dataname=pd_data,
                fromdate=pd.to_datetime(self.trade_config.date_from),
                todate=pd.to_datetime(self.trade_config.date_to)
            )
            cerebro.adddata(data)
        elif type(pd_data) == list:
            for df in pd_data:
                # assert unique symbol
                assert len(df["symbol"].unique()) == 1
                data = bt.feeds.PandasData(
                    dataname=df,
                    fromdate=pd.to_datetime(self.trade_config.date_from),
                    todate=pd.to_datetime(self.trade_config.date_to)
                )
                cerebro.adddata(data, name=df["symbol"].unique()[0])
                # print(df["symbol"].unique()[0], df.shape)


        for additional_arg in kwargs:
            # if it is callable, call it
            if callable(kwargs[additional_arg]):
                kwargs[additional_arg] = kwargs[additional_arg](pd_data)

        # Add a strategy
        cerebro.addstrategy(strategy, **kwargs)

        # Set our desired cash start
        cerebro.broker.setcash(self.trade_config.cash)
        cerebro.broker.setcommission(commission=self.trade_config.commission)
        cerebro.broker.set_slippage_perc(perc=self.trade_config.slippage_perc)

        # Add observers
        cerebro.addobserver(bt.observers.Value)

        # Add analyzers for Sharpe Ratio and Drawdown
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe', riskfreerate=self.trade_config.risk_free_rate,
                            annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='mydrawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='myreturns')
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        # cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='myannualreturn')
        # cerebro.addanalyzer(bt.analyzers.VWR, _name='myvwr')  # Annualized volatility

        # Run over everything
        results = cerebro.run()
        strat = results[0]

        # Print out the final result
        eval_metrics = self.analyze_results(strat, ticker="all", print_trades_table=True)

        # Obtain the equity curve
        equity_with_time = pd.DataFrame(
            {
                "datetime": strat.equity_date,
                "equity": strat.equity
            }
        )

        eval_metrics["equity_with_time"] = equity_with_time

        # Plot the result
        plt.figure(figsize=(10, 6))
        plt.plot(equity_with_time["datetime"], equity_with_time["equity"], label="Equity Curve")
        plt.title(f"Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.show()

        return eval_metrics


    def analyze_results(self, strategy, ticker, print_details=True, print_annual_metrics=True,
                        print_trades_table=False):
        if strategy is None:
            print("No strategy results to analyze")
            return None


        max_drawdown = strategy.analyzers.mydrawdown.get_analysis().max.drawdown
        total_return = strategy.broker.getvalue() / self.trade_config.cash - 1
        total_return_cash = strategy.broker.getvalue() - self.trade_config.cash
        annual_metrics = self._calculate_annualized_metrics(strategy)

        if print_details:
            print("\n" + "=" * 50)
            print(f"Backtest Results for {ticker}:")
            print("-" * 50)
            print(f"Initial cash: {self.trade_config.cash}")
            print(f"Final cash: {strategy.broker.getvalue():.2f}")
            print(f"Total return (cash): {total_return_cash:.2f}")
            print(f"Total return (%): {total_return:.2%}")
            print(f"Max drawdown: {max_drawdown:.2f}")
            print(f"Number of trades: {len(strategy.trades)}")

        if print_annual_metrics:
            print("-" * 50)
            print(f"Annual return: {annual_metrics['Annual Return']:.2%}")
            print(f"Annual volatility: {annual_metrics['Annual Volatility']:.2%}")
            print(f"Sharpe ratio: {annual_metrics['Sharpe Ratio']:.4f}")

        if print_trades_table:
            trades = []
            for trade in strategy.trades:
                trades.append([trade.open_datetime().date(), trade.close_datetime().date(), trade.price, trade.pnl, trade.pnlcomm])
            trades_df = pd.DataFrame(trades, columns=['Open Date', 'Close Date', "Price", 'Profit/Loss',
                                                      'PnL (incl. commission)'])
            print("-" * 50)
            print("Trades:")
            print(tabulate(trades_df, headers='keys', tablefmt='psql'))

        print("="*50)

        return {
            'sharpe_ratio': annual_metrics['Sharpe Ratio'],
            'annual_return': annual_metrics['Annual Return'],
            'annual_volatility': annual_metrics['Annual Volatility'],
            'max_drawdown': max_drawdown,
            'total_return': total_return
        }


    def _calculate_annualized_metrics(
            self,
            strategy,):
        # str to datetime
        first_date = pd.to_datetime(self.trade_config.date_from)
        last_date = pd.to_datetime(self.trade_config.date_to)
        total_days = (last_date - first_date).days
        annual_factor = 252 / total_days

        daily_returns = pd.Series(strategy.equity).pct_change().dropna()

        if len(daily_returns) > 0:
            total_return = (strategy.broker.getvalue() / self.trade_config.cash) - 1
            annual_return = (1 + total_return) ** annual_factor - 1


            sharpe_ratio = strategy.analyzers.mysharpe.get_analysis()['sharperatio']

            annual_volatility = (annual_return - self.trade_config.risk_free_rate) / sharpe_ratio

        else:
            annual_return = annual_volatility = sharpe_ratio = 0

        return {
            "Annual Return": annual_return,
            "Annual Volatility": annual_volatility,
            "Sharpe Ratio": sharpe_ratio,
        }
