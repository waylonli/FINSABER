from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest.data_util import FinsaberDataset
from backtest.finsaber_bt import FINSABERBt
from backtest.strategy.timing import BuyAndHoldStrategy


def build_demo_loader():
    data = {}
    for offset, timestamp in enumerate(pd.bdate_range("2021-01-04", "2024-02-05")):
        day = timestamp.date()
        open_price = 100.0 + offset
        close_price = open_price + 1.0
        data[day] = {
            "price": {
                "DEMO": {
                    "open": open_price,
                    "high": max(open_price, close_price),
                    "low": min(open_price, close_price),
                    "close": close_price,
                    "adjusted_close": close_price,
                    "volume": 1_000_000,
                }
            },
            "news": {"DEMO": []},
            "filing_k": {},
            "filing_q": {},
        }
    return FinsaberDataset(data=data)


if __name__ == "__main__":
    config = {
        "data_loader": build_demo_loader(),
        "tickers": ["DEMO"],
        "date_from": "2024-01-02",
        "date_to": "2024-02-05",
        "setup_name": "custom_dataset_example",
        "save_results": False,
        "silence": True,
    }
    result = FINSABERBt(config).run_iterative_tickers(BuyAndHoldStrategy)
    print(result["DEMO"]["total_return"])
