from datetime import date

import pytest

from backtest.data_util import FinsaberDataset
from llm_traders.finsaber_strategies.tradingagents import TradingAgentsStrategy
from llm_traders.tradingagent.tradingagents.graph.runtime_wiring import (
    validate_runtime_benchmark_seams,
    build_runtime_tool_surfaces,
    validate_runtime_tool_surfaces,
)


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
                    }
                }
            }
        }
    )


def _dummy_dispatch_tool(method: str, *args, **kwargs) -> str:
    return f"{method}:{args}:{kwargs}"


@pytest.mark.unit
def test_runtime_tool_surfaces_match_batch4_baseline_contract():
    tool_surfaces = build_runtime_tool_surfaces(_dummy_dispatch_tool)

    assert tuple(tool.name for tool in tool_surfaces["market"]) == (
        "get_stock_data",
        "get_indicators",
        "get_verified_market_snapshot",
    )
    assert tuple(tool.name for tool in tool_surfaces["news"]) == (
        "get_news",
        "get_global_news",
    )
    assert tuple(tool.name for tool in tool_surfaces["fundamentals"]) == (
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
    )


@pytest.mark.unit
def test_runtime_tool_surfaces_dispatch_through_local_seam_only():
    calls = []

    def dispatch_tool(method: str, *args, **kwargs) -> str:
        calls.append((method, args, kwargs))
        return f"ok:{method}"

    tool_surfaces = build_runtime_tool_surfaces(dispatch_tool)

    assert tool_surfaces["market"][0].func("TSLA", "2024-01-02", "2024-01-10") == "ok:get_stock_data"
    assert tool_surfaces["market"][2].func("TSLA", "2024-01-10", 30) == "ok:get_verified_market_snapshot"
    assert tool_surfaces["news"][1].func("2024-01-10", 7, 10) == "ok:get_global_news"
    assert tool_surfaces["fundamentals"][0].func("TSLA", "2024-01-10") == "ok:get_fundamentals"

    assert calls == [
        ("get_stock_data", ("TSLA", "2024-01-02", "2024-01-10"), {}),
        ("get_verified_market_snapshot", ("TSLA", "2024-01-10", 30), {}),
        ("get_global_news", ("2024-01-10", 7, 10), {}),
        ("get_fundamentals", ("TSLA", "2024-01-10"), {}),
    ]


@pytest.mark.unit
def test_runtime_news_tool_description_marks_local_window_as_authoritative():
    tool_surfaces = build_runtime_tool_surfaces(_dummy_dispatch_tool)

    news_tool = tool_surfaces["news"][0]

    assert "authoritative" in news_tool.description
    assert "requested window is wider or different" in news_tool.description


@pytest.mark.unit
def test_runtime_tool_surfaces_reject_social_when_disabled():
    tool_surfaces = build_runtime_tool_surfaces(_dummy_dispatch_tool)

    with pytest.raises(ValueError, match="allow_social"):
        validate_runtime_tool_surfaces(
            selected_analysts=["market", "social"],
            tool_surfaces=tool_surfaces,
            data_policy={"allow_social": False},
        )


@pytest.mark.unit
def test_runtime_benchmark_seams_require_instrument_context_and_outcome_resolver():
    tool_surfaces = build_runtime_tool_surfaces(_dummy_dispatch_tool)

    with pytest.raises(ValueError, match="instrument_context_builder"):
        validate_runtime_benchmark_seams(
            tool_surfaces=tool_surfaces,
            instrument_context_builder=None,
            outcome_resolver=lambda **_: (0.01, None, 5),
        )

    with pytest.raises(ValueError, match="outcome_resolver"):
        validate_runtime_benchmark_seams(
            tool_surfaces=tool_surfaces,
            instrument_context_builder=lambda ticker, asset_type="stock": f"{ticker}:{asset_type}",
            outcome_resolver=None,
        )


@pytest.mark.unit
def test_strategy_graph_uses_runtime_tool_surfaces(tmp_path):
    artifact_root = (
        tmp_path
        / "cherry_pick_both_finmem"
        / "TradingAgentsStrategy"
        / "tradingagents_window_2024"
        / "tradingagents_artifacts"
    )
    strategy = TradingAgentsStrategy(
        symbol="TSLA",
        date_from="2024-01-02",
        date_to="2024-12-31",
        data_loader=_build_loader(),
        artifact_config={
            "enabled": True,
            "root": str(artifact_root),
            "run_key": "run_test",
        },
    )

    assert tuple(
        tool.name for tool in strategy.graph.graph_setup.analyst_tool_surfaces["market"]
    ) == (
        "get_stock_data",
        "get_indicators",
        "get_verified_market_snapshot",
    )
    assert tuple(strategy.graph.tool_nodes["news"].tools_by_name.keys()) == (
        "get_news",
        "get_global_news",
    )
    assert "get_insider_transactions" not in strategy.graph.tool_nodes["news"].tools_by_name
    assert strategy.graph.instrument_context_builder is not None
    assert strategy.graph.outcome_resolver is not None
