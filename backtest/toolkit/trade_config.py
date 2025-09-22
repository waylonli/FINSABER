from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Union, List, Any, Dict
import os
from backtest.strategy.selection import BaseSelector
import sys


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


@dataclass(slots=True)
class SlippageConfig:
    """Parameters for the six-factor slippage model."""

    enabled: bool = True
    coef_beta_indexret: float = 0.3
    coef_log_market_cap: float = -0.2
    coef_pct_dtv: float = 0.35
    coef_signed_sqrt_dtv: float = 9.32
    coef_idiosync_vol: float = 0.32
    coef_vix: float = 0.13
    fallback_log_market_cap: float = 2.0

    def update_from_dict(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            if not hasattr(self, key):
                continue
            if key == "enabled":
                setattr(self, key, _coerce_bool(value, self.enabled))
            else:
                setattr(self, key, float(value))


@dataclass(slots=True)
class LiquidityConfig:
    """Parameters for liquidity capping and trade continuation."""

    enabled: bool = True
    cap_fraction_of_adv: float = 0.10
    adv_window: int = 252
    median_lookback_days: int = 21
    max_usd_cap: float = 500_000.0
    continue_after_cap: bool = False

    def update_from_dict(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            if not hasattr(self, key):
                continue
            if key in {"enabled", "continue_after_cap"}:
                setattr(self, key, _coerce_bool(value, getattr(self, key)))
            elif key == "cap_fraction_of_adv":
                setattr(self, key, float(value))
            elif key == "adv_window":
                setattr(self, key, int(value))
            elif key == "median_lookback_days":
                setattr(self, key, int(value))
            elif key == "max_usd_cap":
                setattr(self, key, float(value))


@dataclass(slots=True)
class ExecutionConfig:
    """Top-level configuration for execution costs and instrumentation."""

    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    record_trades: bool = True
    record_sizing_debug: bool = False

    def update_from_dict(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            if key == "slippage" and isinstance(value, dict):
                self.slippage.update_from_dict(value)
            elif key == "liquidity" and isinstance(value, dict):
                self.liquidity.update_from_dict(value)
            elif key in {"record_trades", "record_sizing_debug"}:
                setattr(self, key, _coerce_bool(value, getattr(self, key)))

    @classmethod
    def from_dict(cls, values: Dict[str, Any] | None) -> "ExecutionConfig":
        config = cls()
        if values:
            config.update_from_dict(values)
        return config

@dataclass
class TradeConfig:
    tickers: Union[List[str], str]  # Now can also be 'all' to indicate all tickers
    date_from: str = "2004-01-01"
    date_to: str = "2024-01-01"
    cash: float = 1000000.0
    risk_free_rate: float = 0.03
    print_trades_table: bool = False
    silence: bool = False
    rolling_window_size: int = 2
    rolling_window_step: int = 1
    selection_strategy: BaseSelector = None
    setup_name: str = None
    result_filename: str = None
    save_results: bool = True
    training_years: int = None  # Number of years to use for training in rolling window setups
    log_base_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    data_loader: str = None
    liquidity_cap_pct: float = 0.10  # Maximum trade size as percentage of ADV (default: 10%)
    use_slippage: bool = True  # Toggle to enable/disable slippage costs in sizing
    liquidity_max_usd: float = 500000.0
    liquidity_median_lookback_days: int = 21
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

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

        self.liquidity_max_usd = float(self.liquidity_max_usd)
        self.liquidity_median_lookback_days = int(self.liquidity_median_lookback_days)

        # Synchronise aliases with the execution configuration
        self.execution.liquidity.cap_fraction_of_adv = self.liquidity_cap_pct
        self.execution.liquidity.max_usd_cap = self.liquidity_max_usd
        self.execution.liquidity.median_lookback_days = self.liquidity_median_lookback_days
        self.use_slippage = self.execution.slippage.enabled
        liquidity_active = (
            (self.execution.liquidity.cap_fraction_of_adv > 0)
            or (self.execution.liquidity.max_usd_cap > 0)
        )
        self.execution.liquidity.enabled = liquidity_active


    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Initialize a TradeConfig object from a dictionary."""
        config_copy = dict(config_dict)
        execution_cfg = config_copy.get("execution")
        if isinstance(execution_cfg, dict):
            config_copy["execution"] = ExecutionConfig.from_dict(execution_cfg)
        elif isinstance(execution_cfg, ExecutionConfig):
            pass
        else:
            config_copy["execution"] = ExecutionConfig()
        # Keep aliases in sync if provided
        if "liquidity_cap_pct" in config_copy:
            config_copy["execution"].liquidity.cap_fraction_of_adv = config_copy["liquidity_cap_pct"]

        if "liquidity_max_usd" in config_copy:
            config_copy["execution"].liquidity.max_usd_cap = float(config_copy["liquidity_max_usd"])

        if "liquidity_median_lookback_days" in config_copy:
            config_copy["execution"].liquidity.median_lookback_days = int(config_copy["liquidity_median_lookback_days"])

        liquidity_cfg = config_copy["execution"].liquidity
        liquidity_cfg.enabled = (
            liquidity_cfg.cap_fraction_of_adv > 0
            or liquidity_cfg.max_usd_cap > 0
        )

        if "use_slippage" in config_copy:
            config_copy["execution"].slippage.enabled = bool(config_copy["use_slippage"])

        return cls(**config_copy)

    def to_dict(self):
        """ Convert the TradeConfig object to a dictionary """
        return asdict(self)


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
