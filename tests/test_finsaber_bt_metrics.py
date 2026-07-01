from types import SimpleNamespace

import pandas as pd
import pytest

from backtest.finsaber_bt import FINSABERBt
from backtest.toolkit import metrics


class _FakeBroker:
    def __init__(self, value):
        self._value = value

    def getvalue(self):
        return self._value


class _FakeSharpeAnalyzer:
    def get_analysis(self):
        return {"sharperatio": 999.0}


class _FakeStrategy:
    def __init__(self):
        self.equity = [
            100_000.0,
            100_800.0,
            100_500.0,
            101_200.0,
            100_900.0,
            101_500.0,
            101_100.0,
        ]
        self.broker = _FakeBroker(self.equity[-1])
        self.analyzers = SimpleNamespace(mysharpe=_FakeSharpeAnalyzer())
        self.datas = [SimpleNamespace(_name="AAA")]


def test_calculate_annualized_metrics_uses_shared_sharpe_and_sortino_helpers():
    operator = FINSABERBt(
        {
            "tickers": ["AAA"],
            "date_from": "2024-01-02",
            "date_to": "2024-01-31",
            "setup_name": "unit",
            "data_loader": None,
            "silence": True,
        }
    )
    strategy = _FakeStrategy()

    annual_metrics = operator._calculate_annualized_metrics(
        strategy, operator.trade_config
    )
    daily_returns = pd.Series(strategy.equity).pct_change().dropna()

    expected_sharpe = metrics.calculate_sharpe_ratio(
        daily_returns, risk_free_rate=operator.trade_config.risk_free_rate
    )
    expected_sortino = metrics.calculate_sortino_ratio(
        daily_returns, risk_free_rate=operator.trade_config.risk_free_rate
    )

    assert annual_metrics["Sharpe Ratio"] == pytest.approx(expected_sharpe)
    assert annual_metrics["Sortino Ratio"] == pytest.approx(expected_sortino)
    assert annual_metrics["Sharpe Ratio"] != 999.0
