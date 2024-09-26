from dataclasses import dataclass, field, asdict
from typing import Union, List

@dataclass
class TradeConfig:
    tickers: Union[List[str], str]  # Now can also be 'all' to indicate all tickers
    date_from: str = "2004-01-01"
    date_to: str = "2024-01-01"
    cash: float = 100000.0
    commission: float = 0.0003
    slippage_perc: float = 0.0001
    risk_free_rate: float = 0.0

    def __post_init__(self):
        # Validate and manage the tickers field
        if isinstance(self.tickers, str):
            if self.tickers.lower() != "all":
                raise ValueError("tickers can either be a list of tickers or the string 'all'")
        elif not isinstance(self.tickers, list) or not all(isinstance(t, str) for t in self.tickers):
            raise ValueError("tickers must be a list of strings")

    @classmethod
    def from_dict(cls, config_dict):
        """ Initialize a TradeConfig object from a dictionary """
        return cls(**config_dict)

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