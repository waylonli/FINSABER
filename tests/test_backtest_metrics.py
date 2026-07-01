import numpy as np
import pandas as pd
import pytest

from backtest.toolkit import metrics


def test_calculate_sharpe_ratio_matches_standard_excess_return_formula():
    daily_returns = pd.Series([0.012, -0.008, 0.015, 0.004, -0.003, 0.011])
    risk_free_rate = 0.03

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = daily_returns - daily_rf
    expected = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    assert metrics.calculate_sharpe_ratio(
        daily_returns, risk_free_rate=risk_free_rate
    ) == pytest.approx(expected)


def test_calculate_sortino_ratio_uses_semideviation_not_negative_subset_std():
    daily_returns = pd.Series([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01])

    # Under the legacy implementation this returned 0 because the std of the
    # negative-return subset was zero. Standard Sortino uses semideviation.
    expected = -np.sqrt(252)

    assert metrics.calculate_sortino_ratio(daily_returns) == pytest.approx(expected)


def test_calculate_sortino_ratio_returns_zero_when_no_downside_exists():
    daily_returns = pd.Series([0.01, 0.015, 0.012, 0.013, 0.011, 0.014])

    assert metrics.calculate_sortino_ratio(daily_returns) == 0.0
