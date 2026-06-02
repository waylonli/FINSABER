"""Public FINSABER package facade.

The implementation currently lives in the historical ``backtest`` package.
This facade provides the PyPI-facing import path while keeping existing
``backtest`` imports working for repository code and older experiments.
"""

from backtest.data_util import (
    DEFAULT_FINSABER2_DATA_ROOT,
    FinsaberDataset,
    FinsaberParquetDataset,
    TradingData,
    create_finsaber2_data_loader,
    get_finsaber2_data_root,
    resolve_trading_data,
    trading_data_to_env_dict,
)
from backtest.toolkit.trade_config import TradeConfig

__all__ = [
    "FINSABER",
    "FINSABERBt",
    "DEFAULT_FINSABER2_DATA_ROOT",
    "FinsaberDataset",
    "FinsaberParquetDataset",
    "TradeConfig",
    "TradingData",
    "create_finsaber2_data_loader",
    "get_finsaber2_data_root",
    "resolve_trading_data",
    "trading_data_to_env_dict",
]


def __getattr__(name):
    if name == "FINSABER":
        from backtest.finsaber import FINSABER

        return FINSABER
    if name == "FINSABERBt":
        from backtest.finsaber_bt import FINSABERBt

        return FINSABERBt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
