import pytest

from backtest.toolkit.execution import (
    adjusted_bar_price,
    apply_liquidity_cap,
    apply_slippage,
    calculate_commission,
)


def test_commission_applies_minimum_and_maximum_rate():
    assert calculate_commission(10, 100.0, commission_per_share=0.0049, min_commission=0.99) == 0.99
    assert calculate_commission(100, 10.0, commission_per_share=1.0, min_commission=0.0, max_commission_rate=0.01) == 10.0
    assert calculate_commission(0, 10.0) == 0.0


def test_adjusted_bar_price_derives_adjusted_open_from_close_ratio():
    bar = {"open": 50.0, "close": 100.0, "adjusted_close": 200.0}

    assert adjusted_bar_price(bar, "adjusted_open") == 100.0
    assert adjusted_bar_price(bar, "adjusted_close") == 200.0


def test_liquidity_cap_uses_average_volume_percentage():
    assert apply_liquidity_cap(1000, average_volume=10_000, cap_pct=0.025) == 250
    assert apply_liquidity_cap(1000, average_volume=None, cap_pct=0.025) == 1000
    assert apply_liquidity_cap(1000, average_volume=None, cap_pct=0.025, require_volume=True) == 0
    assert apply_liquidity_cap(1000, average_volume=10_000, cap_pct=0.0) == 1000


def test_slippage_worsens_fill_price_by_side_and_participation():
    buy_price, buy_cost = apply_slippage(
        100.0,
        "buy",
        quantity=100,
        average_volume=10_000,
        slippage_perc=0.001,
        slippage_impact=0.1,
    )
    sell_price, sell_cost = apply_slippage(
        100.0,
        "sell",
        quantity=100,
        average_volume=10_000,
        slippage_perc=0.001,
        slippage_impact=0.1,
    )

    assert buy_price == pytest.approx(100.101)
    assert sell_price == pytest.approx(99.899)
    assert buy_cost == pytest.approx(10.1)
    assert sell_cost == pytest.approx(10.1)
