from __future__ import annotations

from datetime import date
import json

from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper
import pytest
import pandas as pd

from backtest.data_util import FinsaberDataset
import llm_traders.finsaber_strategies.tradingagents as tradingagents_strategy_module
from llm_traders.finsaber_strategies.tradingagents import (
    TAOfflineSessionAdapter,
    TATraceWriter,
    TradingAgentsStrategy,
)
from llm_traders.tradingagent.tradingagents.agents.utils.memory import TradingMemoryLog


def _price_row(open_price: float) -> dict[str, float]:
    return {
        "open": open_price,
        "high": open_price + 1.0,
        "low": open_price - 1.0,
        "close": open_price + 0.5,
        "adjusted_open": open_price,
        "adjusted_high": open_price + 1.0,
        "adjusted_low": open_price - 1.0,
        "adjusted_close": open_price + 0.5,
        "volume": 1_000.0,
    }


def _repeat(text: str, count: int) -> str:
    return " ".join([text] * count)


def _sample_10k() -> str:
    return "\n".join(
        [
            "ITEM 1. BUSINESS",
            _repeat(
                "Tesla operates an integrated electric vehicle and energy platform with global manufacturing and distribution.",
                40,
            ),
            "ITEM 1A. RISK FACTORS",
            _repeat(
                "The company faces execution, supply chain, competition, and regulatory risks across production and demand cycles.",
                35,
            ),
            "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
            _repeat(
                "Management discusses revenue growth, margin pressure, capital expenditure, liquidity, and operating leverage trends.",
                45,
            ),
            "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
            _repeat(
                "The filing includes audited consolidated statements, note disclosures, and supplementary financial schedules.",
                45,
            ),
            "ITEM 9A. CONTROLS AND PROCEDURES",
            _repeat(
                "Management evaluated disclosure controls and internal control over financial reporting.",
                20,
            ),
        ]
    )


def _sample_10q() -> str:
    return "\n".join(
        [
            "PART I",
            "ITEM 1. FINANCIAL STATEMENTS",
            _repeat(
                "The quarterly filing includes condensed consolidated balance sheets, income statements, and cash flow statements.",
                40,
            ),
            "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
            _repeat(
                "Management describes quarter-over-quarter demand, production efficiency, pricing, margins, liquidity, and investment.",
                40,
            ),
            "PART II",
            "ITEM 1A. RISK FACTORS",
            _repeat(
                "Quarterly risk factors emphasize execution variability, macro demand sensitivity, and supplier concentration.",
                25,
            ),
            "ITEM 2. UNREGISTERED SALES OF EQUITY SECURITIES AND USE OF PROCEEDS",
            _repeat(
                "The company reports no material unregistered sales and summarizes repurchase or proceeds usage where applicable.",
                20,
            ),
        ]
    )


def _build_loader(include_benchmark: bool = True) -> FinsaberDataset:
    dates = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
        date(2024, 1, 9),
        date(2024, 1, 10),
    ]
    tsla_opens = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    spy_opens = [400.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0]
    data = {}
    for idx, current_date in enumerate(dates):
        daily = {
            "price": {
                "TSLA": _price_row(tsla_opens[idx]),
                "AAPL": _price_row(200.0 + idx),
            },
            "news": {},
        }
        if include_benchmark:
            daily["price"]["SPY"] = _price_row(spy_opens[idx])
        if current_date == date(2024, 1, 3):
            daily["news"]["TSLA"] = [
                "Tesla announces a production efficiency update for its primary assembly line."
            ]
        if current_date == date(2024, 1, 4):
            daily["filing_k"] = {"TSLA": _sample_10k()}
        if current_date == date(2024, 1, 9):
            daily["filing_q"] = {"TSLA": _sample_10q()}
        data[current_date] = daily
    return FinsaberDataset(data=data)


def _build_backtest_loader(periods: int = 25) -> FinsaberDataset:
    dates = [timestamp.date() for timestamp in pd.bdate_range("2024-01-02", periods=periods)]
    data = {}
    for idx, current_date in enumerate(dates):
        daily = {
            "price": {
                "TSLA": _price_row(100.0 + idx),
                "SPY": _price_row(400.0 + idx),
            },
            "news": {},
        }
        if idx == 2:
            daily["news"]["TSLA"] = [
                "Tesla reiterates production guidance in a local benchmark smoke run."
            ]
        if idx == 3:
            daily["filing_k"] = {"TSLA": _sample_10k()}
        if idx == 10:
            daily["filing_q"] = {"TSLA": _sample_10q()}
        data[current_date] = daily
    return FinsaberDataset(data=data)


def _build_adapter(tmp_path, *, include_benchmark: bool = True) -> TAOfflineSessionAdapter:
    loader = _build_loader(include_benchmark=include_benchmark)
    return _build_adapter_from_loader(
        tmp_path,
        loader=loader,
        date_from="2024-01-02",
        date_to="2024-01-10",
    )


def _build_adapter_from_loader(
    tmp_path,
    *,
    loader: FinsaberDataset,
    date_from: str,
    date_to: str,
) -> TAOfflineSessionAdapter:
    window_loader = loader.get_subset_by_time_range("2024-01-02", "2024-01-10")
    if date_from != "2024-01-02" or date_to != "2024-01-10":
        window_loader = loader.get_subset_by_time_range(date_from, date_to)
    runtime_loader = window_loader.get_ticker_subset_by_time_range(
        "TSLA",
        date_from,
        date_to,
    )
    memory_log = TradingMemoryLog(
        {"memory_log_path": str(tmp_path / "memory.md")}
    )
    return TAOfflineSessionAdapter(
        symbol="TSLA",
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        window_loader=window_loader,
        runtime_loader=runtime_loader,
        memory_log=memory_log,
        default_benchmark_ticker="SPY",
        trace_writer=TATraceWriter(ticker_dir=tmp_path / "tickers" / "TSLA"),
    )


def _build_dense_news_loader() -> FinsaberDataset:
    dates = [timestamp.date() for timestamp in pd.bdate_range("2024-01-02", "2024-01-10")]
    data = {}
    for idx, current_date in enumerate(dates):
        daily = {
            "price": {
                "TSLA": _price_row(100.0 + idx),
                "SPY": _price_row(400.0 + idx),
            },
            "news": {},
        }
        if current_date == date(2024, 1, 8):
            daily["news"]["TSLA"] = [f"TSLA 2024-01-08 item {item_idx}" for item_idx in range(8)]
        if current_date == date(2024, 1, 9):
            daily["news"]["TSLA"] = [f"TSLA 2024-01-09 item {item_idx}" for item_idx in range(8)]
        if current_date == date(2024, 1, 10):
            daily["news"]["TSLA"] = [f"TSLA 2024-01-10 item {item_idx}" for item_idx in range(8)]
        data[current_date] = daily
    return FinsaberDataset(data=data)


class _FrameworkStub:
    def __init__(self, *, quantity: int = 0):
        self.portfolio = {}
        if quantity > 0:
            self.portfolio["TSLA"] = {"quantity": quantity}
        self.buy_calls = []
        self.sell_calls = []

    def buy(self, date, ticker, price, quantity):
        self.buy_calls.append((date, ticker, price, quantity))

    def sell(self, date, ticker, price, quantity):
        self.sell_calls.append((date, ticker, price, quantity))


def _patch_fake_graph(monkeypatch, *, rating: str, final_trade_decision: str):
    class FakeTradingAgentsGraph:
        def __init__(
            self,
            *,
            selected_analysts,
            config,
            analyst_tool_surfaces,
            sentiment_prefetch_loader,
            instrument_context_builder,
            outcome_resolver,
        ):
            self.selected_analysts = selected_analysts
            self.config = config
            self.analyst_tool_surfaces = analyst_tool_surfaces
            self.sentiment_prefetch_loader = sentiment_prefetch_loader
            self.instrument_context_builder = instrument_context_builder
            self.outcome_resolver = outcome_resolver
            self.runtime_adapter = None
            self.propagate_calls = []
            self.memory_log = TradingMemoryLog(
                {"memory_log_path": config["memory_log_path"]}
            )

        def bind_runtime_adapter(self, runtime_adapter):
            self.runtime_adapter = runtime_adapter

        def propagate(self, company_name, trade_date, asset_type="stock"):
            self.propagate_calls.append((company_name, trade_date, asset_type))
            appended = self.memory_log.store_decision(
                ticker=company_name,
                trade_date=str(trade_date),
                final_trade_decision=final_trade_decision,
            )
            if (
                appended
                and self.runtime_adapter is not None
                and hasattr(self.runtime_adapter, "record_memory_write")
            ):
                self.runtime_adapter.record_memory_write(
                    event_type="append_pending_decision",
                    decision_date=str(trade_date),
                    rating=rating,
                    note="decision_logged_for_future_reflection",
                )
            return {"final_trade_decision": final_trade_decision}, rating

    monkeypatch.setattr(
        tradingagents_strategy_module,
        "_get_tradingagents_graph_class",
        lambda: FakeTradingAgentsGraph,
    )
    return FakeTradingAgentsGraph


@pytest.mark.unit
def test_local_trim_cap_keeps_under_cap_text_verbatim():
    text = "alpha\n\nbeta\n\ngamma"

    trimmed, applied = tradingagents_strategy_module._apply_local_trim_cap(  # noqa: SLF001
        text,
        cap=len(text),
    )

    assert applied is False
    assert trimmed == text


@pytest.mark.unit
def test_local_trim_cap_trims_over_cap_with_marker():
    text = "alpha\n\nbeta\n\n" + ("g" * 30)

    trimmed, applied = tradingagents_strategy_module._apply_local_trim_cap(  # noqa: SLF001
        text,
        cap=35,
    )

    assert applied is True
    assert trimmed == "alpha[TRUNCATED: local cap reached]"


@pytest.mark.unit
def test_local_trim_cap_returns_empty_string_when_cap_is_zero():
    trimmed, applied = tradingagents_strategy_module._apply_local_trim_cap(  # noqa: SLF001
        "alpha",
        cap=0,
    )

    assert applied is True
    assert trimmed == ""


@pytest.mark.unit
def test_classify_trace_source_mode_recognizes_partial_truncation_marker():
    partial_marker = tradingagents_strategy_module._TRADINGAGENTS_LOCAL_TRUNCATION_MARKER[:12]  # noqa: SLF001

    assert (
        tradingagents_strategy_module._classify_trace_source_mode(partial_marker)  # noqa: SLF001
        == "local_trimmed"
    )
    assert (
        tradingagents_strategy_module._classify_trace_source_mode(  # noqa: SLF001
            f"local payload {partial_marker}"
        )
        == "local_trimmed"
    )


@pytest.mark.unit
def test_offline_session_adapter_clamps_stock_data_and_disables_global_news(tmp_path):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-04", pre_decision_position_state="flat")

    csv_output = adapter.dispatch_tool(
        "get_stock_data",
        "TSLA",
        "2023-12-15",
        "2024-01-10",
    )
    assert "2024-01-02" in csv_output
    assert "2024-01-04" in csv_output
    assert "2024-01-05" not in csv_output

    instrument_context = adapter.build_instrument_context("TSLA", "stock")
    assert "`TSLA`" in instrument_context
    assert "Resolved identity" not in instrument_context
    assert "GLOBAL_NEWS_DISABLED" in adapter.dispatch_tool(
        "get_global_news",
        "2024-01-04",
        7,
        10,
    )


@pytest.mark.unit
def test_offline_session_adapter_uses_authoritative_news_window_regardless_of_request(
    tmp_path,
):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-10", pre_decision_position_state="flat")

    long_window = adapter.dispatch_tool("get_news", "TSLA", "1900-01-01", "2024-01-10")
    short_window = adapter.dispatch_tool("get_news", "TSLA", "2024-01-09", "2024-01-10")

    assert long_window == short_window
    assert "- Visible window: 2024-01-03 to 2024-01-10" in long_window
    assert "- 2024-01-03: Tesla announces a production efficiency update for its primary assembly line." in long_window

    trace_path = tmp_path / "tickers" / "TSLA" / "analyst_input_trace.jsonl"
    lines = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    news_lines = [line for line in lines if line["input_slot"] == "ticker_news"]

    assert len(news_lines) == 2
    assert {line["visible_window_start"] for line in news_lines} == {"2024-01-03"}
    assert {line["visible_window_end"] for line in news_lines} == {"2024-01-10"}


@pytest.mark.unit
def test_offline_session_adapter_news_window_does_not_leak_pre_window_days_on_first_day(
    tmp_path,
):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-02", pre_decision_position_state="flat")

    output = adapter.dispatch_tool("get_news", "TSLA", "1900-01-01", "2024-01-02")

    assert (
        output
        == "NO_TICKER_NEWS_AVAILABLE: No local ticker news is visible for TSLA "
        "between 2024-01-02 and 2024-01-02."
    )

    trace_path = tmp_path / "tickers" / "TSLA" / "analyst_input_trace.jsonl"
    lines = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert lines[-1]["input_slot"] == "ticker_news"
    assert lines[-1]["visible_window_start"] == "2024-01-02"
    assert lines[-1]["visible_window_end"] == "2024-01-02"
    assert lines[-1]["source_mode"] == "explicit_unavailable"


@pytest.mark.unit
def test_offline_session_adapter_news_window_is_newest_first_and_limited_to_20_items(
    tmp_path,
):
    adapter = _build_adapter_from_loader(
        tmp_path,
        loader=_build_dense_news_loader(),
        date_from="2024-01-02",
        date_to="2024-01-10",
    )
    adapter.bind_decision_day("2024-01-10", pre_decision_position_state="flat")

    output = adapter.dispatch_tool("get_news", "TSLA", "2024-01-02", "2024-01-10")
    article_lines = [line for line in output.splitlines() if line.startswith("- 2024-")]

    assert len(article_lines) == 20
    assert article_lines[:8] == [
        f"- 2024-01-10: TSLA 2024-01-10 item {item_idx}" for item_idx in range(8)
    ]
    assert article_lines[8:16] == [
        f"- 2024-01-09: TSLA 2024-01-09 item {item_idx}" for item_idx in range(8)
    ]
    assert article_lines[16:20] == [
        f"- 2024-01-08: TSLA 2024-01-08 item {item_idx}" for item_idx in range(4)
    ]


@pytest.mark.unit
def test_offline_session_adapter_sentiment_prefetch_uses_trade_date_news_window(
    tmp_path,
):
    adapter = _build_adapter_from_loader(
        tmp_path,
        loader=_build_dense_news_loader(),
        date_from="2024-01-02",
        date_to="2024-01-10",
    )
    adapter.bind_decision_day("2024-01-10", pre_decision_position_state="flat")

    blocks = adapter.load_sentiment_prefetch_blocks("TSLA", "2024-01-08")

    assert "- Visible window: 2024-01-02 to 2024-01-08" in blocks["news_block"]
    assert "- 2024-01-08: TSLA 2024-01-08 item 0" in blocks["news_block"]
    assert "2024-01-09" not in blocks["news_block"]
    assert "2024-01-10" not in blocks["news_block"]


@pytest.mark.unit
def test_offline_session_adapter_resolves_open_to_open_outcome_locally(tmp_path):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-09", pre_decision_position_state="flat")
    assert adapter.resolve_outcome("2024-01-02", 5, "SPY") == (None, None, None)

    adapter.bind_decision_day("2024-01-10", pre_decision_position_state="flat")
    raw_return, alpha_return, holding_days = adapter.resolve_outcome(
        "2024-01-02",
        5,
        "SPY",
    )

    expected_raw = (106.0 - 101.0) / 101.0
    expected_benchmark = (406.0 - 401.0) / 401.0
    assert holding_days == 5
    assert raw_return == pytest.approx(expected_raw)
    assert alpha_return == pytest.approx(expected_raw - expected_benchmark)


@pytest.mark.unit
def test_offline_session_adapter_keeps_raw_return_when_local_benchmark_is_unavailable(
    tmp_path,
):
    adapter = _build_adapter(tmp_path, include_benchmark=False)
    adapter.bind_decision_day("2024-01-10", pre_decision_position_state="flat")

    raw_return, alpha_return, holding_days = adapter.resolve_outcome(
        "2024-01-02",
        5,
        "SPY",
    )

    expected_raw = (106.0 - 101.0) / 101.0
    assert holding_days == 5
    assert raw_return == pytest.approx(expected_raw)
    assert alpha_return is None


@pytest.mark.unit
def test_offline_session_adapter_uses_visible_filings_for_fundamentals_and_statement_proxy(
    tmp_path,
):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-04", pre_decision_position_state="flat")

    fundamentals_before_q = adapter.dispatch_tool("get_fundamentals", "TSLA", "2024-01-04")

    assert "### Annual Business Context" in fundamentals_before_q
    assert "### Current Risk Context" in fundamentals_before_q
    assert "- Source: latest visible 10-K item_1 dated 2024-01-04" in fundamentals_before_q
    assert "- Source: latest visible 10-K item_1a dated 2024-01-04" in fundamentals_before_q
    assert "### Latest Quarterly MD&A Update" not in fundamentals_before_q

    adapter.bind_decision_day("2024-01-10", pre_decision_position_state="flat")

    fundamentals = adapter.dispatch_tool("get_fundamentals", "TSLA", "2024-01-10")
    annual_proxy = adapter.dispatch_tool(
        "get_balance_sheet",
        "TSLA",
        "annual",
        "2024-01-10",
    )
    quarterly_proxy = adapter.dispatch_tool(
        "get_cashflow",
        "TSLA",
        "quarterly",
        "2024-01-10",
    )

    assert "### Annual Business Context" in fundamentals
    assert "### Current Risk Context" in fundamentals
    assert "### Annual MD&A Baseline" in fundamentals
    assert "### Latest Quarterly MD&A Update" in fundamentals
    assert "- Source: latest visible 10-Q part_ii_item_1a dated 2024-01-09" in fundamentals
    assert "- Source: latest visible 10-K item_1a" not in fundamentals
    assert "Proxy section: item_8" in annual_proxy
    assert "Proxy section: part_i_item_1" in quarterly_proxy


@pytest.mark.unit
def test_offline_session_adapter_dedups_same_day_statement_proxy_source_but_resets_next_day(
    tmp_path,
):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-04", pre_decision_position_state="flat")

    first = adapter.dispatch_tool("get_balance_sheet", "TSLA", "annual", "2024-01-04")
    duplicate = adapter.dispatch_tool("get_cashflow", "TSLA", "annual", "2024-01-04")

    assert "Proxy section: item_8" in first
    assert duplicate.startswith("DUPLICATE_LOCAL_FUNDAMENTALS_SOURCE:")
    assert "10-K item_8 filing dated 2024-01-04" in duplicate

    adapter.bind_decision_day("2024-01-10", pre_decision_position_state="flat")
    replay = adapter.dispatch_tool("get_income_statement", "TSLA", "annual", "2024-01-10")

    assert "Proxy section: item_8" in replay


@pytest.mark.unit
def test_offline_session_adapter_traces_duplicate_fundamentals_bundle_note(tmp_path):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-10", pre_decision_position_state="flat")

    first = adapter.dispatch_tool("get_fundamentals", "TSLA", "2024-01-10")
    duplicate = adapter.dispatch_tool("get_fundamentals", "TSLA", "2024-01-10")

    assert "### Annual Business Context" in first
    assert duplicate.startswith("DUPLICATE_LOCAL_FUNDAMENTALS_SOURCE:")
    assert "Annual Business Context=10-K item_1 dated 2024-01-04" in duplicate
    assert "Latest Quarterly MD&A Update=10-Q part_i_item_2 dated 2024-01-09" in duplicate

    trace_path = tmp_path / "tickers" / "TSLA" / "analyst_input_trace.jsonl"
    lines = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fundamentals_lines = [
        line for line in lines if line["input_slot"] == "fundamentals_proxy"
    ]

    assert len(fundamentals_lines) == 2
    assert [line["source_mode"] for line in fundamentals_lines] == [
        "local",
        "duplicate_source_note",
    ]
    assert fundamentals_lines[1]["summary_text"].startswith(
        "DUPLICATE_LOCAL_FUNDAMENTALS_SOURCE:"
    )


@pytest.mark.unit
def test_offline_session_adapter_marks_trimmed_statement_proxy_in_trace(tmp_path, monkeypatch):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-10", pre_decision_position_state="flat")
    monkeypatch.setattr(tradingagents_strategy_module, "_TRADINGAGENTS_STATEMENT_SOURCE_CAP", 80)
    monkeypatch.setattr(
        tradingagents_strategy_module,
        "_TRADINGAGENTS_FUNDAMENTALS_CYCLE_TOTAL_CAP",
        10_000,
    )

    output = adapter.dispatch_tool("get_balance_sheet", "TSLA", "annual", "2024-01-10")

    assert "[TRUNCATED: local cap reached]" in output

    trace_path = tmp_path / "tickers" / "TSLA" / "analyst_input_trace.jsonl"
    lines = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert lines[-1]["input_slot"] == "balance_sheet_proxy"
    assert lines[-1]["source_mode"] == "local_trimmed"
    assert lines[-1]["summary_text"].startswith("LOCAL_TRIMMED:")


@pytest.mark.unit
def test_offline_session_adapter_cycle_cap_can_trim_later_fundamentals_outputs(
    tmp_path,
    monkeypatch,
):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-10", pre_decision_position_state="flat")
    monkeypatch.setattr(
        tradingagents_strategy_module,
        "_TRADINGAGENTS_FUNDAMENTALS_CYCLE_TOTAL_CAP",
        400,
    )
    monkeypatch.setattr(
        tradingagents_strategy_module,
        "_TRADINGAGENTS_STATEMENT_SOURCE_CAP",
        10_000,
    )

    first = adapter.dispatch_tool("get_fundamentals", "TSLA", "2024-01-10")
    second = adapter.dispatch_tool("get_balance_sheet", "TSLA", "annual", "2024-01-10")

    assert "[TRUNCATED: local cap reached]" in first or "[TRUNCATED: local cap reached]" in second
    assert len(first) + len(second) == adapter._fundamentals_cycle_chars_used  # noqa: SLF001
    assert adapter._fundamentals_cycle_chars_used <= 400  # noqa: SLF001


@pytest.mark.unit
def test_offline_session_adapter_duplicate_note_also_respects_cycle_cap(
    tmp_path,
    monkeypatch,
):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-04", pre_decision_position_state="flat")

    first = adapter.dispatch_tool("get_balance_sheet", "TSLA", "annual", "2024-01-04")
    used_after_first = adapter._fundamentals_cycle_chars_used  # noqa: SLF001
    remaining = len(tradingagents_strategy_module._TRADINGAGENTS_LOCAL_TRUNCATION_MARKER) + 8  # noqa: SLF001
    monkeypatch.setattr(
        tradingagents_strategy_module,
        "_TRADINGAGENTS_FUNDAMENTALS_CYCLE_TOTAL_CAP",
        used_after_first + remaining,
    )

    duplicate = adapter.dispatch_tool("get_cashflow", "TSLA", "annual", "2024-01-04")

    assert len(first) == used_after_first
    assert len(duplicate) == remaining
    assert duplicate.endswith(tradingagents_strategy_module._TRADINGAGENTS_LOCAL_TRUNCATION_MARKER)  # noqa: SLF001
    assert adapter._fundamentals_cycle_chars_used == used_after_first + remaining  # noqa: SLF001


@pytest.mark.unit
def test_offline_session_adapter_marks_zero_remaining_fundamentals_budget_as_trimmed(
    tmp_path,
    monkeypatch,
):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-04", pre_decision_position_state="flat")

    first = adapter.dispatch_tool("get_balance_sheet", "TSLA", "annual", "2024-01-04")
    used_after_first = adapter._fundamentals_cycle_chars_used  # noqa: SLF001
    monkeypatch.setattr(
        tradingagents_strategy_module,
        "_TRADINGAGENTS_FUNDAMENTALS_CYCLE_TOTAL_CAP",
        used_after_first,
    )

    duplicate = adapter.dispatch_tool("get_cashflow", "TSLA", "annual", "2024-01-04")

    assert duplicate == ""
    trace_path = tmp_path / "tickers" / "TSLA" / "analyst_input_trace.jsonl"
    lines = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert lines[-1]["source_mode"] == "local_trimmed"
    assert lines[-1]["summary_text"] == "LOCAL_TRIMMED: [TRUNCATED: local cap reached]"


@pytest.mark.unit
def test_offline_session_adapter_writes_analyst_input_trace_for_local_disabled_and_unavailable(
    tmp_path,
):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-04", pre_decision_position_state="flat")

    adapter.dispatch_tool("get_news", "TSLA", "2024-01-02", "2024-01-04")
    adapter.dispatch_tool("get_global_news", "2024-01-04", 7, 10)
    adapter.dispatch_tool("get_fundamentals", "TSLA", "2024-01-03")
    adapter.dispatch_tool("get_indicators", "TSLA", "close_10_ema", "2024-01-04", 2)
    adapter.dispatch_tool("get_verified_market_snapshot", "TSLA", "2024-01-04", 30)

    trace_path = tmp_path / "tickers" / "TSLA" / "analyst_input_trace.jsonl"
    lines = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert [line["input_slot"] for line in lines] == [
        "ticker_news",
        "global_news",
        "fundamentals_proxy",
        "technical_indicators",
        "verified_market_snapshot",
    ]
    assert [line["source_mode"] for line in lines] == [
        "local",
        "disabled_placeholder",
        "explicit_unavailable",
        "local",
        "local",
    ]
    assert all(line["date"] == "2024-01-04" for line in lines)
    assert all(line["ticker"] == "TSLA" for line in lines)
    assert lines[0]["prompt_policy_variant"] == "news_local_macro_guard_v1"
    assert lines[1]["visible_window_end"] == "2024-01-04"
    assert lines[2]["summary_text"].startswith("NO_FUNDAMENTALS_AVAILABLE:")
    assert lines[3]["visible_window_start"] == "2024-01-03"
    assert lines[3]["visible_window_end"] == "2024-01-04"
    assert lines[4]["visible_window_start"] == "2024-01-02"
    assert lines[4]["visible_window_end"] == "2024-01-04"
    assert "Verified local market data snapshot for TSLA" in lines[4]["summary_text"]


@pytest.mark.unit
def test_trace_summary_marks_local_outputs_with_embedded_unavailable_markers(tmp_path):
    adapter = _build_adapter(tmp_path)
    adapter.bind_decision_day("2024-01-04", pre_decision_position_state="flat")

    synthetic_local_snapshot = "\n".join(
        [
            "## Verified local market data snapshot for TSLA",
            "",
            "| Indicator | Value |",
            "|---|---:|",
            "| macd | N/A (IndexError) |",
            "| atr | 2.00 |",
        ]
    )

    summary = adapter._build_trace_summary_text(synthetic_local_snapshot)

    assert summary.startswith("LOCAL_WITH_UNAVAILABLE_MARKERS:")
    assert "macd | N/A (IndexError)" in summary


@pytest.mark.unit
def test_offline_session_adapter_writes_memory_reads_trace_once_per_day(tmp_path):
    adapter = _build_adapter(tmp_path)
    adapter.memory_log.store_decision(
        ticker="TSLA",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Buy\nReason: pending TSLA decision.",
        benchmark_execution_semantic="enter_long",
    )
    adapter.memory_log.update_with_outcome(
        ticker="TSLA",
        trade_date="2024-01-02",
        raw_return=0.05,
        alpha_return=0.01,
        holding_days=5,
        reflection="A compact lesson.",
    )
    adapter.bind_decision_day("2024-01-10", pre_decision_position_state="flat")

    context = adapter.load_past_context("TSLA")
    adapter.load_past_context("TSLA")

    trace_path = tmp_path / "tickers" / "TSLA" / "memory_reads.jsonl"
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert "Past analyses of TSLA" in context
    assert len(records) == 1
    assert records[0]["date"] == "2024-01-10"
    assert records[0]["ticker"] == "TSLA"
    assert records[0]["same_ticker_count"] == 1
    assert records[0]["cross_ticker_count"] == 0
    assert records[0]["lesson_dates"] == ["2024-01-02"]
    assert "BENCHMARK_EXECUTION_SEMANTIC:" not in records[0]["past_context_text"]


@pytest.mark.unit
def test_strategy_on_data_runs_graph_and_submits_buy_all_from_flat(
    tmp_path,
    monkeypatch,
):
    _patch_fake_graph(
        monkeypatch,
        rating="Buy",
        final_trade_decision="Rating: Buy\nReason: synthetic TSLA long thesis.",
    )
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
        date_to="2024-01-10",
        data_loader=_build_loader(),
        artifact_config={
            "enabled": True,
            "root": str(artifact_root),
            "run_key": "run_test",
        },
    )

    framework = _FrameworkStub(quantity=0)
    today_data = strategy.runtime_loader.get_data_by_date(date(2024, 1, 3))

    strategy.on_data(date(2024, 1, 3), today_data, framework)

    assert strategy.runtime_adapter.current_day == date(2024, 1, 3)
    assert strategy.runtime_adapter.pre_decision_position_state == "flat"
    assert strategy.graph.runtime_adapter is strategy.runtime_adapter
    assert strategy.graph.propagate_calls == [("TSLA", "2024-01-03", "stock")]
    assert framework.buy_calls == [(date(2024, 1, 3), "TSLA", 101.5, -1)]
    assert framework.sell_calls == []
    assert strategy.last_execution_bridge_payload == {
        "raw_rating": "Buy",
        "pre_decision_position_state": "flat",
        "mapped_target_state": "long",
        "executed_action": "submit_buy_all",
        "reference_price": 101.5,
    }
    assert "benchmark_execution_semantic" not in strategy.last_execution_bridge_payload
    pending_entries = strategy.graph.memory_log.get_pending_entries()
    assert len(pending_entries) == 1
    assert pending_entries[0]["date"] == "2024-01-03"
    assert pending_entries[0]["ticker"] == "TSLA"
    assert pending_entries[0]["rating"] == "Buy"
    assert pending_entries[0]["benchmark_execution_semantic"] is None


@pytest.mark.unit
def test_strategy_on_data_runs_graph_and_submits_sell_all_from_long(
    tmp_path,
    monkeypatch,
):
    _patch_fake_graph(
        monkeypatch,
        rating="Underweight",
        final_trade_decision="Rating: Underweight\nReason: synthetic TSLA de-risking.",
    )
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
        date_to="2024-01-10",
        data_loader=_build_loader(),
        artifact_config={
            "enabled": True,
            "root": str(artifact_root),
            "run_key": "run_test",
        },
    )

    framework = _FrameworkStub(quantity=7)
    today_data = strategy.runtime_loader.get_data_by_date(date(2024, 1, 3))

    strategy.on_data(date(2024, 1, 3), today_data, framework)

    assert strategy.runtime_adapter.current_day == date(2024, 1, 3)
    assert strategy.runtime_adapter.pre_decision_position_state == "long"
    assert strategy.graph.propagate_calls == [("TSLA", "2024-01-03", "stock")]
    assert framework.buy_calls == []
    assert framework.sell_calls == [(date(2024, 1, 3), "TSLA", 101.5, 7)]
    assert strategy.last_execution_bridge_payload == {
        "raw_rating": "Underweight",
        "pre_decision_position_state": "long",
        "mapped_target_state": "flat",
        "executed_action": "submit_sell_all",
        "reference_price": 101.5,
    }
    assert "benchmark_execution_semantic" not in strategy.last_execution_bridge_payload
    pending_entries = strategy.graph.memory_log.get_pending_entries()
    assert len(pending_entries) == 1
    assert pending_entries[0]["date"] == "2024-01-03"
    assert pending_entries[0]["ticker"] == "TSLA"
    assert pending_entries[0]["rating"] == "Underweight"
    assert pending_entries[0]["benchmark_execution_semantic"] is None


@pytest.mark.unit
def test_strategy_short_window_smoke_runs_framework_and_evaluate_chain(
    tmp_path,
    monkeypatch,
):
    _patch_fake_graph(
        monkeypatch,
        rating="Buy",
        final_trade_decision="Rating: Buy\nReason: synthetic TSLA long thesis.",
    )
    loader = _build_backtest_loader(periods=25)
    strategy = TradingAgentsStrategy(
        symbol="TSLA",
        date_from="2024-01-02",
        date_to="2024-02-05",
        data_loader=loader,
        artifact_config={
            "enabled": False,
            "root": str(
                tmp_path
                / "cherry_pick_both_finmem"
                / "TradingAgentsStrategy"
                / "tradingagents_smoke"
                / "tradingagents_artifacts"
            ),
            "run_key": "run_smoke",
        },
    )
    framework = FINSABERFrameworkHelper()

    assert framework.load_backtest_data_single_ticker(
        loader,
        "TSLA",
        "2024-01-02",
        "2024-02-05",
    )

    status = framework.run(strategy)
    metrics = framework.evaluate(strategy)

    assert status is True
    assert len(strategy.equity) > 1
    assert len(framework.history) >= 2
    assert metrics["final_value"] > 0
    assert "total_return" in metrics
    assert "annual_return" in metrics
    assert "total_trading_cost" in metrics


@pytest.mark.unit
def test_strategy_disables_optional_trace_writes_when_artifacts_disabled(tmp_path):
    strategy = TradingAgentsStrategy(
        symbol="TSLA",
        date_from="2024-01-02",
        date_to="2024-01-10",
        data_loader=_build_loader(),
        artifact_config={
            "enabled": False,
            "root": str(
                tmp_path
                / "cherry_pick_both_finmem"
                / "TradingAgentsStrategy"
                / "tradingagents_window_2024"
                / "tradingagents_artifacts"
            ),
            "run_key": "run_test",
        },
    )

    assert strategy.trace_writer is None
    assert strategy.runtime_adapter.trace_writer is None
