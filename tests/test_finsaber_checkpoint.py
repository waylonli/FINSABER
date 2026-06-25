from datetime import date, timedelta

import pandas as pd

from backtest.data_util import FinsaberDataset
from backtest.finsaber import FINSABER


class ExplodingStrategy:
    def __init__(self, **kwargs):
        raise AssertionError("checkpointed ticker should not instantiate strategy")


def _sample_loader(num_days=25):
    start = date(2024, 1, 2)
    data = {}
    for offset in range(num_days):
        current = start + timedelta(days=offset)
        price = 10.0 + offset
        data[current] = {
            "price": {
                "AAA": {
                    "open": price,
                    "close": price + 0.5,
                    "adjusted_open": price,
                    "adjusted_close": price + 0.5,
                    "volume": 1000,
                }
            }
        }
    return FinsaberDataset(data=data)


def _metrics():
    return {
        "final_value": 101_000.0,
        "total_return": 0.01,
        "annual_return": 0.12,
        "annual_volatility": 0.2,
        "sharpe_ratio": 1.1,
        "sortino_ratio": 1.3,
        "max_drawdown": 2.0,
        "total_commission": 1.0,
        "total_slippage": 0.5,
        "total_external_cost": 0.25,
        "total_llm_cost": 0.25,
        "total_trading_cost": 1.75,
        "equity_with_time": pd.DataFrame(
            {"datetime": [pd.Timestamp("2024-01-02")], "equity": [101_000.0]}
        ),
        "external_costs": pd.DataFrame({"reason": ["llm_training_cost"], "cost": [0.25]}),
        "llm_cost_records": pd.DataFrame({"model": ["gpt-4o-mini"], "cost": [0.25]}),
        "trades": pd.DataFrame({"ticker": ["AAA"], "quantity": [10]}),
        "rejected_orders": pd.DataFrame(),
    }


def test_run_iterative_tickers_skips_completed_checkpoint(tmp_path):
    engine = FINSABER(
        {
            "tickers": ["AAA"],
            "date_from": "2024-01-02",
            "date_to": "2024-01-27",
            "setup_name": "unit",
            "log_base_dir": str(tmp_path),
            "data_loader": _sample_loader(),
            "silence": True,
            "save_results": True,
        }
    )
    engine._save_ticker_checkpoint(ExplodingStrategy, "AAA", _metrics())

    results = engine.run_iterative_tickers(ExplodingStrategy, strat_params={}, tickers=["AAA"])

    window = "2024-01-02_2024-01-27"
    assert results[window]["AAA"]["final_value"] == 101_000.0
    output_dir = tmp_path / "unit" / "ExplodingStrategy"
