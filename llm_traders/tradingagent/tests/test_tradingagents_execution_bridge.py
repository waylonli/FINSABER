import pytest

from datetime import date

from backtest.data_util import FinsaberDataset
from llm_traders.finsaber_strategies.tradingagents import (
    TradingAgentsStrategy,
    build_tradingagents_execution_bridge_payload,
)


class DummyFramework:
    def __init__(self, quantity: int = 0):
        self.portfolio = {}
        if quantity > 0:
            self.portfolio["TSLA"] = {"quantity": quantity}
        self.buy_calls = []
        self.sell_calls = []

    def buy(self, date, ticker, price, quantity):
        self.buy_calls.append((date, ticker, price, quantity))

    def sell(self, date, ticker, price, quantity):
        self.sell_calls.append((date, ticker, price, quantity))


def _build_loader():
    return FinsaberDataset(
        data={
            date(2024, 1, 2): {
                "price": {
                    "TSLA": {
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "adjusted_open": 100.0,
                        "adjusted_high": 101.0,
                        "adjusted_low": 99.0,
                        "adjusted_close": 100.5,
                        "volume": 1_000,
                    },
                    "SPY": {
                        "open": 400.0,
                        "high": 401.0,
                        "low": 399.0,
                        "close": 400.5,
                        "adjusted_open": 400.0,
                        "adjusted_high": 401.0,
                        "adjusted_low": 399.0,
                        "adjusted_close": 400.5,
                        "volume": 10_000,
                    },
                }
            }
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw_rating", "pre_state", "target_state", "action"),
    [
        ("Buy", "flat", "long", "submit_buy_all"),
        ("Buy", "long", "long", "noop"),
        ("Overweight", "flat", "long", "submit_buy_all"),
        ("Overweight", "long", "long", "noop"),
        ("Hold", "flat", "flat", "noop"),
        ("Hold", "long", "long", "noop"),
        ("Underweight", "flat", "flat", "noop"),
        ("Underweight", "long", "flat", "submit_sell_all"),
        ("Sell", "flat", "flat", "noop"),
        ("Sell", "long", "flat", "submit_sell_all"),
    ],
)
def test_execution_bridge_matches_frozen_fold_table(
    raw_rating,
    pre_state,
    target_state,
    action,
):
    payload = build_tradingagents_execution_bridge_payload(
        raw_rating=raw_rating,
        pre_decision_position_state=pre_state,
    )

    assert payload == {
        "raw_rating": raw_rating,
        "pre_decision_position_state": pre_state,
        "mapped_target_state": target_state,
        "executed_action": action,
    }


@pytest.mark.unit
def test_strategy_apply_execution_bridge_submits_buy_all_from_flat(tmp_path):
    strategy = TradingAgentsStrategy(
        symbol="TSLA",
        date_from="2024-01-02",
        date_to="2024-12-31",
        data_loader=_build_loader(),
        artifact_config={
            "enabled": False,
            "root": str(
                tmp_path
                / "TradingAgentsStrategy"
                / "tradingagents_window_2024"
                / "tradingagents_artifacts"
            ),
            "run_key": "run_test",
        },
    )
    framework = DummyFramework(quantity=0)

    payload = strategy._apply_execution_bridge(
        raw_rating="Buy",
        date=strategy.date_from,
        reference_price=123.45,
        framework=framework,
    )

    assert framework.buy_calls == [
        (strategy.date_from, "TSLA", 123.45, -1),
    ]
    assert framework.sell_calls == []
    assert payload["executed_action"] == "submit_buy_all"
    assert payload["reference_price"] == 123.45


@pytest.mark.unit
def test_strategy_apply_execution_bridge_submits_sell_all_from_long(tmp_path):
    strategy = TradingAgentsStrategy(
        symbol="TSLA",
        date_from="2024-01-02",
        date_to="2024-12-31",
        data_loader=_build_loader(),
        artifact_config={
            "enabled": False,
            "root": str(
                tmp_path
                / "TradingAgentsStrategy"
                / "tradingagents_window_2024"
                / "tradingagents_artifacts"
            ),
            "run_key": "run_test",
        },
    )
    framework = DummyFramework(quantity=7)

    payload = strategy._apply_execution_bridge(
        raw_rating="Underweight",
        date=strategy.date_from,
        reference_price=99.0,
        framework=framework,
    )

    assert framework.buy_calls == []
    assert framework.sell_calls == [
        (strategy.date_from, "TSLA", 99.0, 7),
    ]
    assert payload["executed_action"] == "submit_sell_all"


@pytest.mark.unit
def test_strategy_apply_execution_bridge_keeps_hold_as_noop(tmp_path):
    strategy = TradingAgentsStrategy(
        symbol="TSLA",
        date_from="2024-01-02",
        date_to="2024-12-31",
        data_loader=_build_loader(),
        artifact_config={
            "enabled": False,
            "root": str(
                tmp_path
                / "TradingAgentsStrategy"
                / "tradingagents_window_2024"
                / "tradingagents_artifacts"
            ),
            "run_key": "run_test",
        },
    )
    framework = DummyFramework(quantity=7)

    payload = strategy._apply_execution_bridge(
        raw_rating="Hold",
        date=strategy.date_from,
        reference_price=88.0,
        framework=framework,
    )

    assert framework.buy_calls == []
    assert framework.sell_calls == []
    assert payload["executed_action"] == "noop"
