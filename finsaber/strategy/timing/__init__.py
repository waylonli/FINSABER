"""Timing strategy imports for the public ``finsaber`` package."""

from backtest.strategy.timing.atr_band import ATRBandStrategy
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.strategy.timing.bollinger_band import BollingerBandsStrategy
from backtest.strategy.timing.buy_and_hold import BuyAndHoldStrategy
from backtest.strategy.timing.sma_crossover import SMACrossStrategy
from backtest.strategy.timing.trend_following import TrendFollowingStrategy
from backtest.strategy.timing.turn_of_the_month import TurnOfTheMonthStrategy
from backtest.strategy.timing.wma_crossover import WMAStrategy

__all__ = [
    "ARIMAPredictorStrategy",
    "ATRBandStrategy",
    "BaseStrategy",
    "BollingerBandsStrategy",
    "BuyAndHoldStrategy",
    "SMACrossStrategy",
    "TrendFollowingStrategy",
    "TurnOfTheMonthStrategy",
    "WMAStrategy",
    "XGBoostPredictorStrategy",
]


def __getattr__(name):
    if name == "ARIMAPredictorStrategy":
        from backtest.strategy.timing.arima_predictor import ARIMAPredictorStrategy

        return ARIMAPredictorStrategy
    if name == "XGBoostPredictorStrategy":
        from backtest.strategy.timing.xgboost_predictor import XGBoostPredictorStrategy

        return XGBoostPredictorStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
