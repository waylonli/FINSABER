import os

import pandas as pd
import backtrader as bt


class FamaFrench3Factor(bt.Strategy):
    # TODO
    def __init__(self):
        pass


if __name__ == "__main__":
    pd_data = pd.read_csv(os.path.join("data", "SP500", "AAPL.csv"))
    pd_data["Date"] = pd.to_datetime(pd_data["Date"])
    # date as index
    pd_data.set_index("Date", inplace=True)
    data = bt.feeds.PandasData(
        dataname=pd_data,
        fromdate=pd.to_datetime("2019-01-01"),  # 回测开始日期
        todate=pd.to_datetime("2024-01-01")  # 回测结束日期
    )

    cerebro = bt.Cerebro()  # 实例化大脑
    cerebro.addstrategy(FamaFrench3Factor)  # 添加策略
    cerebro.adddata(data)

    cerebro.broker.setcash(1000000.0)  # 初始资金
    cerebro.broker.setcommission(commission=0.0003)  # 佣金，双边各 0.0003
    cerebro.broker.set_slippage_perc(perc=0.0001)  # 滑点：双边各 0.0001

    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())  # 打印初始现金
    thestrats = cerebro.run()
    thestrat = thestrats[0]
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())  # 打印策略运行结束后的现金

    print("---------------------------------")
    print('Sharpe Ratio:', round(thestrat.analyzers.sharperatio.get_analysis()['sharperatio'], 4))
    print('Return:', round(thestrat.analyzers.returns.get_analysis()['rnorm'], 4))
    print("---------------------------------")

    cerebro.plot(style='candlestick', volume=False)
