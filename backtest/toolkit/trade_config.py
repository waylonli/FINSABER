from dataclasses import dataclass, field, fields
from typing import Any, Union, List
import os
from backtest.strategy.selection.base_selector import BaseSelector
import sys

@dataclass
class TradeConfig:
    tickers: Union[List[str], str]  # Now can also be 'all' to indicate all tickers
    date_from: str = "2004-01-01"
    date_to: str = "2024-01-01"
    cash: float = 100000.0
    risk_free_rate: float = 0.03
    commission_per_share: float = 0.0049
    min_commission: float = 0.99
    max_commission_rate: float = 0.01
    execution_timing: str = "next_open"
    slippage_perc: float = 0.0
    slippage_impact: float = 0.0
    liquidity_lookback_days: int = 20
    liquidity_min_history_days: int = 1
    liquidity_cap_pct: float = 0.0
    llm_cost_as_trade_cost: bool = True
    print_trades_table: bool = False
    silence: bool = False
    rolling_window_size: int = 2
    rolling_window_step: int = 1
    training_years: int = None
    selection_strategy: BaseSelector = None
    setup_name: str = None
    result_filename: str = None
    save_results: bool = True
    log_base_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    data_loader: Any = None

    def __post_init__(self):
        # Validate and manage the tickers field
        if isinstance(self.tickers, str):
            if self.tickers.lower() != "all":
                raise ValueError("tickers can either be a list of tickers or the string 'all'")
        elif not isinstance(self.tickers, list) or not all(isinstance(t, str) for t in self.tickers):
            raise ValueError("tickers must be a list of strings")

        # Validate the date_from and date_to fields
        if self.date_from > self.date_to:
            raise ValueError("date_from must be earlier than date_to")

        if self.execution_timing not in {"same_close", "next_open"}:
            raise ValueError("execution_timing must be one of: same_close, next_open")

        if self.slippage_perc < 0 or self.slippage_impact < 0:
            raise ValueError("slippage_perc and slippage_impact must be non-negative")

        if not 0 <= self.liquidity_cap_pct <= 1:
            raise ValueError("liquidity_cap_pct must be between 0 and 1")

        if self.liquidity_lookback_days < 1:
            raise ValueError("liquidity_lookback_days must be at least 1")

        if not 1 <= self.liquidity_min_history_days <= self.liquidity_lookback_days:
            raise ValueError("liquidity_min_history_days must be between 1 and liquidity_lookback_days")


    @classmethod
    def from_dict(cls, config_dict):
        """ Initialize a TradeConfig object from a dictionary """
        config_dict = dict(config_dict)
        if "commission" in config_dict and "commission_per_share" not in config_dict:
            config_dict["commission_per_share"] = config_dict.pop("commission")
        return cls(**config_dict)

    def to_dict(self):
        """ Convert the TradeConfig object to a dictionary """
        return {item.name: getattr(self, item.name) for item in fields(self)}


if __name__ == "__main__":
    # test the TradeConfig class
    # Initialize with a dictionary that includes "all" tickers
    config_dict = {
        "tickers": "all",
        "date_from": "2010-01-01",
        "date_to": "2023-12-31",
        "cash": 150000.0,
        "commission": 0.0002,
        "slippage_perc": 0.0001
    }

    try:
        config = TradeConfig.from_dict(config_dict)
        print(config)
    except ValueError as e:
        print(e)

    # Initialize with a list of tickers
    config_dict_list = {
        "tickers": ["AAPL", "GOOGL"],
        "date_from": "2010-01-01",
        "date_to": "2023-12-31",
        "cash": 150000.0,
        "commission": 0.0002,
        "slippage_perc": 0.0001
    }

    config_list = TradeConfig.from_dict(config_dict_list)
    print(config_list)

    # Convert back to dictionary
    print(config_list.to_dict())
