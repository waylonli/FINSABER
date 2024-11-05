import numpy as np

def calculate_sortino_ratio(daily_returns, risk_free_rate=0):
    """
    Calculate the Sortino Ratio of a strategy
    :param daily_returns: pd.Series, daily returns of a strategy
    :param risk_free_rate: float, risk-free rate
    :return: float, Sortino Ratio
    """
    # Calculate downside deviation (for Sortino Ratio)
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)

    # Calculate Sortino Ratio
    if downside_deviation != 0:
        sortino_ratio = (daily_returns.mean() * 252 - risk_free_rate) / downside_deviation
    else:
        sortino_ratio = 0

    return sortino_ratio

def calculate_annual_volatility(daily_returns):
    """
    Calculate the annualized volatility of a strategy
    :param daily_returns: pd.Series, daily returns of a strategy
    :return: float, annualized volatility
    """
    return daily_returns.std() * np.sqrt(252)