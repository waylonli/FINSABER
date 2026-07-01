import numpy as np

_MIN_NON_ZERO_RETURNS = 5


def _daily_risk_free_rate(risk_free_rate):
    return (1 + risk_free_rate) ** (1 / 252) - 1


def _has_enough_non_zero_returns(daily_returns):
    non_zero_daily_returns = daily_returns[daily_returns != 0]
    return len(non_zero_daily_returns) >= _MIN_NON_ZERO_RETURNS


def calculate_sharpe_ratio(daily_returns, risk_free_rate=0):
    """
    Calculate the annualized Sharpe Ratio of a strategy.

    :param daily_returns: pd.Series, daily returns of a strategy
    :param risk_free_rate: float, annualized risk-free rate
    :return: float, Sharpe Ratio
    """
    if not _has_enough_non_zero_returns(daily_returns):
        return 0.0

    daily_rf = _daily_risk_free_rate(risk_free_rate)
    excess_returns = daily_returns - daily_rf
    volatility = excess_returns.std()

    if not np.isfinite(volatility) or volatility == 0:
        return 0.0

    sharpe_ratio = excess_returns.mean() / volatility * np.sqrt(252)
    if not np.isfinite(sharpe_ratio):
        return 0.0

    return float(sharpe_ratio)


def calculate_sortino_ratio(daily_returns, risk_free_rate=0):
    """
    Calculate the annualized Sortino Ratio of a strategy.

    :param daily_returns: pd.Series, daily returns of a strategy
    :param risk_free_rate: float, annualized risk-free rate
    :return: float, Sortino Ratio
    """
    if not _has_enough_non_zero_returns(daily_returns):
        return 0.0

    daily_rf = _daily_risk_free_rate(risk_free_rate)

    excess_returns = daily_returns - daily_rf

    # Standard Sortino uses semideviation around the target return rather than
    # the sample std of the negative-return subset.
    downside_returns = np.minimum(excess_returns.to_numpy(), 0.0)
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))

    if not np.isfinite(downside_deviation) or downside_deviation == 0:
        return 0.0

    sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252)
    if not np.isfinite(sortino_ratio):
        return 0.0

    return float(sortino_ratio)

def calculate_annual_volatility(daily_returns):
    """
    Calculate the annualized volatility of a strategy
    :param daily_returns: pd.Series, daily returns of a strategy
    :return: float, annualized volatility
    """
    return daily_returns.std() * np.sqrt(252)
