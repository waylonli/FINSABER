from .atr_band import ATRBandStrategy
from .bollinger_band import BollingerBandsStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .sma_crossover import SMACrossStrategy
from .trend_following import TrendFollowingStrategy
from .turn_of_the_month import TurnOfTheMonthStrategy
from .wma_crossover import WMAStrategy

__all__ = [
    "ARIMAPredictorStrategy",
    "ATRBandStrategy",
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
        from .arima_predictor import ARIMAPredictorStrategy

        return ARIMAPredictorStrategy
    if name == "XGBoostPredictorStrategy":
        from .xgboost_predictor import XGBoostPredictorStrategy

        return XGBoostPredictorStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
