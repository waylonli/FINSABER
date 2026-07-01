from datetime import date, timedelta

from backtest.data_util import FinsaberDataset
from backtest.finsaber_bt import FINSABERBt
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.toolkit.trade_config import TradeConfig


class MinimalBTStrategy(BaseStrategy):
    params = (
        ("total_days", 0),
    )

    def next(self):
        if not self.position:
            self.buy(size=1)
        self.post_next_actions()


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
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price + 0.25,
                    "adjusted_open": price,
                    "adjusted_high": price + 0.5,
                    "adjusted_low": price - 0.5,
                    "adjusted_close": price + 0.25,
                    "volume": 1000,
                }
            }
        }
    return FinsaberDataset(data=data)


def test_trade_config_normalizes_empty_result_output_dir():
    trade_config = TradeConfig.from_dict(
        {
            "tickers": ["AAA"],
            "date_from": "2024-01-02",
            "date_to": "2024-02-05",
            "result_output_dir": "",
        }
    )

    assert trade_config.result_output_dir is None


def test_finsaber_bt_respects_result_output_dir_override(tmp_path):
    custom_output_dir = tmp_path / "custom-benchmark-results"
    engine = FINSABERBt(
        {
            "tickers": ["AAA"],
            "date_from": "2024-01-02",
            "date_to": "2024-02-05",
            "setup_name": "cherry_pick_unit",
            "log_base_dir": str(tmp_path),
            "result_output_dir": str(custom_output_dir),
            "data_loader": _sample_loader(),
            "silence": True,
            "save_results": True,
        }
    )

    results = engine.run_iterative_tickers(MinimalBTStrategy)

    assert "AAA" in results
    assert (custom_output_dir / "2024-01-02_2024-02-05.pkl").exists()
    assert (custom_output_dir / "run_summary.csv").exists()
    assert (custom_output_dir / "run_manifest.json").exists()
    assert not (tmp_path / "cherry_pick_unit" / "MinimalBTStrategy" / "run_summary.csv").exists()
