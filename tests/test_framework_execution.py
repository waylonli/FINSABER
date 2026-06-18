from datetime import date, timedelta

import pytest

from backtest.data_util import FinsaberDataset
from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper


class BuyOnFirstDateStrategy:
    def __init__(self):
        self.equity = []
        self._seen = 0

    def on_data(self, current_date, today_data, framework):
        if self._seen == 0:
            framework.buy(current_date, "AAA", price=999.0, quantity=10)
        self._seen += 1

    def update_info(self, current_date, data_loader, framework):
        close = data_loader.get_ticker_price_by_date("AAA", current_date)
        quantity = framework.portfolio.get("AAA", {}).get("quantity", 0)
        self.equity.append(framework.cash + quantity * close)



class ExternalCostStrategy:
    def __init__(self, cost_state, increment):
        self.equity = []
        self._seen = 0
        self._cost_state = cost_state
        self._increment = increment

    def on_data(self, current_date, today_data, framework):
        if self._seen == 0:
            self._cost_state["value"] += self._increment
        self._seen += 1

    def update_info(self, current_date, data_loader, framework):
        close = data_loader.get_ticker_price_by_date("AAA", current_date)
        quantity = framework.portfolio.get("AAA", {}).get("quantity", 0)
        self.equity.append(framework.cash + quantity * close)


def _sample_loader(num_days=22):
    start = date(2024, 1, 2)
    data = {}
    for offset in range(num_days):
        current = start + timedelta(days=offset)
        open_price = 10.0 + offset
        close_price = open_price + 0.5
        data[current] = {
            "price": {
                "AAA": {
                    "open": open_price,
                    "close": close_price,
                    "adjusted_open": open_price,
                    "adjusted_close": close_price,
                    "volume": 1000,
                }
                }
            }
    return FinsaberDataset(data=data), sorted(data)


def test_same_close_executes_immediately_at_adjusted_close():
    loader, dates = _sample_loader()
    framework = FINSABERFrameworkHelper(
        initial_cash=10_000,
        commission_per_share=0.0,
        min_commission=0.0,
        execution_timing="same_close",
    )
    framework.load_backtest_data(loader)

    framework.buy(dates[0], "AAA", price=999.0, quantity=10)

    assert framework.pending_orders == []
    assert framework.history[0]["signal_date"] == dates[0]
    assert framework.history[0]["execution_date"] == dates[0]
    assert framework.history[0]["price"] == pytest.approx(10.5)


def test_next_open_defers_signal_to_next_bar_open():
    loader, dates = _sample_loader()
    framework = FINSABERFrameworkHelper(
        initial_cash=10_000,
        commission_per_share=0.0,
        min_commission=0.0,
        execution_timing="next_open",
    )
    framework.load_backtest_data(loader)
    strategy = BuyOnFirstDateStrategy()

    assert framework.run(strategy, delist_check=False) is True

    first_trade = framework.history[0]
    assert first_trade["signal_date"] == dates[0]
    assert first_trade["execution_date"] == dates[1]
    assert first_trade["price"] == pytest.approx(11.0)


def test_framework_applies_liquidity_cap_from_prior_volume():
    loader, dates = _sample_loader()
    framework = FINSABERFrameworkHelper(
        initial_cash=10_000,
        commission_per_share=0.0,
        min_commission=0.0,
        execution_timing="same_close",
        liquidity_lookback_days=5,
        liquidity_cap_pct=0.1,
    )
    framework.load_backtest_data(loader)

    framework.buy(dates[5], "AAA", price=999.0, quantity=1000)

    assert framework.history[0]["average_volume"] == pytest.approx(1000)
    assert framework.history[0]["volume_observations"] == 5
    assert framework.history[0]["participation_rate"] == pytest.approx(0.1)
    assert framework.history[0]["quantity"] == 100


def test_framework_rejects_cap_enabled_order_without_prior_volume_history():
    loader, dates = _sample_loader()
    framework = FINSABERFrameworkHelper(
        initial_cash=10_000,
        commission_per_share=0.0,
        min_commission=0.0,
        execution_timing="same_close",
        liquidity_lookback_days=5,
        liquidity_min_history_days=1,
        liquidity_cap_pct=0.1,
    )
    framework.load_backtest_data(loader)

    framework.buy(dates[0], "AAA", price=999.0, quantity=100)

    assert framework.history == []
    assert framework.rejected_orders[0]["reason"] == "insufficient_liquidity_history"


def test_forced_final_liquidation_updates_last_equity_after_costs():
    loader, _ = _sample_loader()
    strategy = BuyOnFirstDateStrategy()
    framework = FINSABERFrameworkHelper(
        initial_cash=10_000,
        commission_per_share=0.0,
        min_commission=1.0,
        execution_timing="same_close",
    )
    framework.load_backtest_data(loader)

    assert framework.run(strategy, delist_check=False) is True
    metrics = framework.evaluate(strategy)

    assert framework.history[-1]["type"] == "sell"
    assert framework.history[-1]["execution_date"] == loader.get_date_range()[-1]
    assert strategy.equity[-1] == pytest.approx(metrics["final_value"])
    assert metrics["total_commission"] == pytest.approx(2.0)


def test_framework_external_cost_reduces_cash_equity_and_metrics():
    loader, _ = _sample_loader()
    cost_state = {"value": 0.0}
    strategy = ExternalCostStrategy(cost_state, increment=2.5)
    framework = FINSABERFrameworkHelper(
        initial_cash=10_000,
        commission_per_share=0.0,
        min_commission=0.0,
    )
    framework.load_backtest_data(loader)

    assert framework.run(
        strategy,
        delist_check=False,
        external_cost_getter=lambda: cost_state["value"],
        external_cost_offset=0.0,
        external_cost_reason="llm_inference_cost",
    ) is True
    metrics = framework.evaluate(strategy)

    assert strategy.equity[0] == pytest.approx(9997.5)
    assert metrics["final_value"] == pytest.approx(9997.5)
    assert metrics["total_external_cost"] == pytest.approx(2.5)
    assert metrics["total_trading_cost"] == pytest.approx(2.5)


def test_framework_slippage_uses_final_cash_affordable_quantity():
    loader, dates = _sample_loader()
    framework = FINSABERFrameworkHelper(
        initial_cash=105.0,
        commission_per_share=0.0,
        min_commission=0.0,
        execution_timing="same_close",
        slippage_impact=0.1,
    )
    framework.load_backtest_data(loader)

    framework.buy(dates[5], "AAA", price=999.0, quantity=10)

    trade = framework.history[0]
    assert trade["quantity"] == 6
    assert trade["participation_rate"] == pytest.approx(0.006)
    assert trade["price"] == pytest.approx(15.5 * (1 + 0.1 * 0.006**2))
