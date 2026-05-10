"""Public interface for the reusable FINSABER backtesting package."""

from backtest.data_util import FinsaberDataset, FinsaberParquetDataset, TradingData
from backtest.toolkit.trade_config import TradeConfig

__all__ = [
    "FINSABER",
    "FINSABERBt",
    "FinsaberDataset",
    "FinsaberParquetDataset",
    "TradeConfig",
    "TradingData",
]


def __getattr__(name):
    if name == "FINSABER":
        from backtest.finsaber import FINSABER

        return FINSABER
    if name == "FINSABERBt":
        from backtest.finsaber_bt import FINSABERBt

        return FINSABERBt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
