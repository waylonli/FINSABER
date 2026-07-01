from datetime import date

from llm_traders.finagent.environment.trading import EnvironmentTrading
from llm_traders.finsaber_strategies.finagent_data_adapter import (
    FinsaberTradingDataDataset,
)


class FakeTradingData:
    def __init__(self):
        self.data = {
            date(2024, 1, 2): {
                "price": {
                    "TEST": {
                        "open": 50.0,
                        "high": 52.0,
                        "low": 49.0,
                        "close": 51.0,
                        "adjusted_open": 100.0,
                        "adjusted_high": 104.0,
                        "adjusted_low": 98.0,
                        "adjusted_close": 102.0,
                        "volume": 1_000,
                    }
                },
                "news": {"TEST": [{"headline": "News", "content": "Body"}]},
                "filing_q": {"TEST": "Quarterly filing"},
            },
            date(2024, 1, 3): {
                "price": {
                    "TEST": {
                        "open": 51.0,
                        "high": 53.0,
                        "low": 50.0,
                        "close": 52.0,
                        "adjusted_open": 102.0,
                        "adjusted_high": 106.0,
                        "adjusted_low": 100.0,
                        "adjusted_close": 104.0,
                        "volume": 1_100,
                    }
                },
                "news": {"TEST": ["Second day"]},
            },
        }

    def get_date_range(self):
        return sorted(self.data)

    def get_data_by_date(self, value):
        return self.data.get(value, {})


def test_direct_data_reaches_finagent_environment():
    dataset = FinsaberTradingDataDataset(
        asset="TEST",
        data_loader=FakeTradingData(),
    )

    assert dataset.prices["TEST"].iloc[0]["open"] == 100.0
    assert dataset.prices["TEST"].iloc[0]["raw_open"] == 50.0
    assert dataset.news["TEST"].iloc[0]["title"] == "News"
    assert len(dataset.guidances["TEST"]) == 1

    environment = EnvironmentTrading(
        dataset=dataset,
        selected_asset="TEST",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
        look_back_days=1,
        look_forward_days=0,
    )
    state, _ = environment.reset()

    assert environment.get_current_date().date() == date(2024, 1, 2)
    assert len(state["news"]) == 1
    assert len(state["guidance"]) == 1
