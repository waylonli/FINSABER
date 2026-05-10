from .arima_predictor import ARIMAPredictorStrategy
from .atr_band import ATRBandStrategy
from .bollinger_band import BollingerBandsStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .sma_crossover import SMACrossStrategy
from .trend_following import TrendFollowingStrategy
from .turn_of_the_month import TurnOfTheMonthStrategy
from .wma_crossover import WMAStrategy
from .xgboost_predictor import XGBoostPredictorStrategy

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
