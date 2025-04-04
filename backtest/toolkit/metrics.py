import numpy as np

def calculate_sortino_ratio(daily_returns, risk_free_rate=0):
    """
    Calculate the Sortino Ratio of a strategy
    :param daily_returns: pd.Series, daily returns of a strategy
    :param risk_free_rate: float, risk-free rate
    :return: float, Sortino Ratio
    """
    # check if non-zero daily returns are enough for calculation
    non_zero_daily_returns = daily_returns[daily_returns != 0]
    if len(non_zero_daily_returns) < 5:
        return 0

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    excess_returns = daily_returns - daily_rf

    # Calculate downside deviation (for Sortino Ratio)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) < 2:
        return 0

    downside_deviation = downside_returns.std()

    if downside_deviation == 0:
        return 0

    return (excess_returns.mean() ) / downside_deviation * np.sqrt(252)

def calculate_annual_volatility(daily_returns):
    """
    Calculate the annualized volatility of a strategy
    :param daily_returns: pd.Series, daily returns of a strategy
    :return: float, annualized volatility
    """
    return daily_returns.std() * np.sqrt(252)