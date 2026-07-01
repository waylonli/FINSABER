from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_traders.finsaber_strategies.tradingagents import TATraceWriter
from llm_traders.tradingagent.tradingagents.agents.utils.memory import TradingMemoryLog
from llm_traders.tradingagent.tradingagents.graph.reflection import Reflector
from llm_traders.tradingagent.tradingagents.graph.trading_graph import TradingAgentsGraph


def _add_business_days(start: date, business_days: int) -> date:
    current = start
    added = 0
    while added < business_days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


class DummyReflector:
    def __init__(self):
        self.calls = []

    def reflect_on_final_decision(
        self,
        *,
        final_decision,
        raw_return,
        alpha_return,
        benchmark_name="SPY",
    ):
        self.calls.append(
            {
                "final_decision": final_decision,
                "raw_return": raw_return,
                "alpha_return": alpha_return,
                "benchmark_name": benchmark_name,
            }
        )
        return "Compact recommendation-space lesson."


@dataclass
class DummyRuntimeAdapter:
    memory_log: TradingMemoryLog
    horizon_days: int = 5
    current_day: date | None = None
    pre_decision_position_state: str | None = None
    trace_writer: TATraceWriter | None = None

    def bind_decision_day(
        self,
        curr_date: str | date,
        pre_decision_position_state: str = "flat",
    ) -> None:
        self.current_day = (
            date.fromisoformat(curr_date) if isinstance(curr_date, str) else curr_date
        )
        self.pre_decision_position_state = pre_decision_position_state

    def is_decision_day_bound(self, trade_date: str) -> bool:
        return self.current_day is not None and self.current_day.isoformat() == trade_date

    def resolve_outcome(self, trade_date: str, holding_days: int, benchmark: str):
        del benchmark
        decision_day = date.fromisoformat(trade_date)
        maturity_day = _add_business_days(decision_day, holding_days + 1)
        if self.current_day is None or self.current_day < maturity_day:
            return None, None, None
        return 0.012, -0.004, holding_days

    def load_past_context(self, ticker: str) -> str:
        return self.memory_log.get_past_context(ticker, n_same=5, n_cross=0)

    def build_instrument_context(self, ticker: str, asset_type: str = "stock") -> str:
        return f"{ticker}:{asset_type}:local"

    def record_memory_write(
        self,
        *,
        event_type: str,
        decision_date: str,
        rating: str,
        note: str,
    ) -> None:
        if self.trace_writer is None:
            return
        assert self.current_day is not None
        self.trace_writer.append_memory_write(
            {
                "date": self.current_day.isoformat(),
                "ticker": "TSLA",
                "event_type": event_type,
                "decision_date": decision_date,
                "rating": rating,
                "note": note,
            }
        )

    def record_reflection_trace(
        self,
        *,
        decision_date: str,
        benchmark_ticker: str,
        raw_return: float,
        alpha_return: float | None,
        actual_holding_days: int,
        final_trade_decision: str,
        reflection_text: str,
    ) -> None:
        if self.trace_writer is None:
            return
        assert self.current_day is not None
        self.trace_writer.append_reflection_trace(
            {
                "ticker": "TSLA",
                "decision_date": decision_date,
                "resolution_date": self.current_day.isoformat(),
                "execution_anchor": "next_open",
                "evaluation_anchor": "open_to_open",
                "horizon_days": self.horizon_days,
                "actual_holding_days": actual_holding_days,
                "benchmark_ticker": benchmark_ticker,
                "raw_return_open_to_open": raw_return,
                "alpha_return_open_to_open": alpha_return,
                "final_trade_decision": final_trade_decision,
                "reflection_text": reflection_text,
            }
        )


class DummyCompiledGraph:
    def __init__(self, final_state):
        self.final_state = final_state

    def invoke(self, init_agent_state, **kwargs):
        del init_agent_state, kwargs
        return self.final_state


class DummyPropagator:
    def create_initial_state(self, *args, **kwargs):
        del args, kwargs
        return {}

    def get_graph_args(self):
        return {}


class DummySignalProcessor:
    def process_signal(self, full_signal):
        del full_signal
        return "Hold"


class CapturingPropagator:
    def __init__(self):
        self.last_call = None

    def create_initial_state(self, *args, **kwargs):
        self.last_call = {"args": args, "kwargs": kwargs}
        return {}

    def get_graph_args(self):
        return {}


def _build_memory_log(tmp_path: Path) -> TradingMemoryLog:
    return TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.md")})


def _build_graph_shell(
    memory_log: TradingMemoryLog,
    *,
    runtime_adapter=None,
    final_state=None,
    instrument_context_builder=None,
    outcome_resolver=None,
    propagator=None,
    use_runtime_instrument_context_method: bool = False,
):
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {
        "benchmark_reflection_contract": {"horizon_days": 5},
        "results_dir": str(Path.cwd() / "tmp_unused_results_dir"),
    }
    graph.memory_log = memory_log
    graph.runtime_adapter = runtime_adapter
    graph.reflector = DummyReflector()
    graph.debug = False
    graph._checkpointer_ctx = None
    graph.propagator = propagator or DummyPropagator()
    graph.graph = DummyCompiledGraph(final_state or _minimal_final_state())
    graph.signal_processor = DummySignalProcessor()
    graph.instrument_context_builder = instrument_context_builder
    graph.outcome_resolver = outcome_resolver
    if runtime_adapter is not None:
        if graph.instrument_context_builder is None:
            graph.instrument_context_builder = runtime_adapter.build_instrument_context
        if graph.outcome_resolver is None:
            graph.outcome_resolver = runtime_adapter.resolve_outcome
    if not use_runtime_instrument_context_method:
        graph.resolve_instrument_context = lambda ticker, asset_type="stock": (
            f"{ticker}:{asset_type}"
        )
    graph._log_state = lambda trade_date, final_state: None
    graph.curr_state = None
    graph.ticker = None
    return graph


def _minimal_final_state():
    return {
        "company_of_interest": "TSLA",
        "trade_date": "2024-01-02",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
        },
        "trader_investment_plan": "",
        "risk_debate_state": {
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "history": "",
            "judge_decision": "",
        },
        "investment_plan": "",
        "final_trade_decision": "Rating: Hold\nReason: test decision.",
    }


@pytest.mark.unit
def test_memory_log_pending_entry_ignores_benchmark_execution_semantic(tmp_path):
    memory_log = _build_memory_log(tmp_path)

    memory_log.store_decision(
        ticker="TSLA",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Hold\nReason: test decision.",
        benchmark_execution_semantic="stay_flat",
    )

    entries = memory_log.load_entries()
    assert len(entries) == 1
    assert entries[0]["pending"] is True
    assert entries[0]["benchmark_execution_semantic"] is None
    assert "BENCHMARK_EXECUTION_SEMANTIC:" not in (
        tmp_path / "memory.md"
    ).read_text(encoding="utf-8")


@pytest.mark.unit
def test_benchmark_runtime_resolve_waits_until_d6_and_past_context_is_same_ticker_only(
    tmp_path,
):
    memory_log = _build_memory_log(tmp_path)
    memory_log.store_decision(
        ticker="TSLA",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Buy\nReason: pending TSLA decision.",
        benchmark_execution_semantic="enter_long",
    )
    memory_log.store_decision(
        ticker="AAPL",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Sell\nReason: resolved AAPL decision.",
        benchmark_execution_semantic="exit_long",
    )
    memory_log.update_with_outcome(
        ticker="AAPL",
        trade_date="2024-01-02",
        raw_return=-0.02,
        alpha_return=-0.01,
        holding_days=5,
        reflection="Resolved AAPL lesson.",
    )

    adapter = DummyRuntimeAdapter(memory_log=memory_log)
    graph = _build_graph_shell(memory_log, runtime_adapter=adapter)

    for curr_day in (
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
        date(2024, 1, 9),
    ):
        adapter.bind_decision_day(curr_day, pre_decision_position_state="flat")
        graph._resolve_pending_entries("TSLA")
        assert graph._load_past_context("TSLA") == ""

    adapter.bind_decision_day(date(2024, 1, 10), pre_decision_position_state="flat")
    graph._resolve_pending_entries("TSLA")
    context = graph._load_past_context("TSLA")
    entries = memory_log.load_entries()
    tsla_entry = next(e for e in entries if e["ticker"] == "TSLA")

    assert tsla_entry["pending"] is False
    assert tsla_entry["holding"] == "5d"
    assert tsla_entry["benchmark_execution_semantic"] is None
    assert "Past analyses of TSLA" in context
    assert "BENCHMARK_EXECUTION_SEMANTIC:" not in context
    assert "AAPL" not in context
    assert "benchmark_execution_semantic" not in graph.reflector.calls[0]


@pytest.mark.unit
def test_batch_update_with_outcomes_returns_only_successfully_applied_entries(tmp_path):
    memory_log = _build_memory_log(tmp_path)
    memory_log.store_decision(
        ticker="TSLA",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Buy\nReason: pending TSLA decision.",
        benchmark_execution_semantic="enter_long",
    )

    updated_entries = memory_log.batch_update_with_outcomes(
        [
            {
                "ticker": "TSLA",
                "trade_date": "2024-01-02",
                "raw_return": 0.012,
                "alpha_return": -0.004,
                "holding_days": 5,
                "reflection": "Resolved TSLA lesson.",
            },
            {
                "ticker": "AAPL",
                "trade_date": "2024-01-02",
                "raw_return": -0.01,
                "alpha_return": 0.0,
                "holding_days": 5,
                "reflection": "Should not apply.",
            },
        ]
    )

    assert updated_entries == [("2024-01-02", "TSLA")]


@pytest.mark.unit
def test_log_state_uses_full_state_log_dir_and_canonical_field_names(tmp_path):
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {
        "results_dir": str(tmp_path / "legacy_results"),
        "full_state_log_dir": str(tmp_path / "tickers" / "TSLA" / "full_state_logs"),
    }
    graph.log_states_dict = {}
    graph.ticker = "TSLA"

    TradingAgentsGraph._log_state(
        graph,
        "2024-01-10",
        {
            **_minimal_final_state(),
            "trade_date": "2024-01-10",
            "instrument_context": "TSLA:stock:local",
            "past_context": "Past analyses of TSLA",
            "trader_investment_plan": "Trader plan",
        },
    )

    log_path = (
        tmp_path
        / "tickers"
        / "TSLA"
        / "full_state_logs"
        / "full_states_log_2024-01-10.json"
    )
    payload = json.loads(log_path.read_text(encoding="utf-8"))

    assert log_path.exists()
    assert payload["instrument_context"] == "TSLA:stock:local"
    assert payload["past_context"] == "Past analyses of TSLA"
    assert payload["trader_investment_plan"] == "Trader plan"
    assert "trader_investment_decision" not in payload


@pytest.mark.unit
def test_reflector_prompt_uses_recommendation_space_and_open_to_open_labels():
    class CapturingLLM:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="Recommendation-space lesson.")

    llm = CapturingLLM()
    reflector = Reflector(llm)

    result = reflector.reflect_on_final_decision(
        final_decision="Rating: Hold\nReason: test decision.",
        raw_return=0.012,
        alpha_return=-0.004,
        benchmark_name="SPY",
    )

    assert result == "Recommendation-space lesson."
    assert "Raw return (open-to-open): +1.2%" in llm.messages[1][1]
    assert "Alpha vs SPY (open-to-open): -0.4%" in llm.messages[1][1]
    assert "Benchmark execution semantic:" not in llm.messages[1][1]


@pytest.mark.unit
def test_reflector_prompt_marks_alpha_unavailable_under_local_benchmark_policy():
    class CapturingLLM:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="Recommendation-space lesson.")

    llm = CapturingLLM()
    reflector = Reflector(llm)

    reflector.reflect_on_final_decision(
        final_decision="Rating: Hold\nReason: test decision.",
        raw_return=0.012,
        alpha_return=None,
        benchmark_name="SPY",
    )

    assert "Alpha vs SPY (open-to-open): unavailable under local benchmark policy" in (
        llm.messages[1][1]
    )


@pytest.mark.unit
def test_reflector_prompt_restores_upstream_alpha_primary_guidance():
    class CapturingLLM:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="Recommendation-space lesson.")

    llm = CapturingLLM()
    reflector = Reflector(llm)

    reflector.reflect_on_final_decision(
        final_decision="Rating: Hold\nReason: test decision.",
        raw_return=-0.012,
        alpha_return=None,
        benchmark_name="SPY",
    )

    system_prompt = llm.messages[0][1]
    assert "Was the directional call correct? (cite the alpha figure)" in system_prompt
    assert "One concrete lesson to apply to the next similar analysis." in system_prompt
    assert "directionally supported or contradicted" not in system_prompt


@pytest.mark.unit
def test_benchmark_runtime_accepts_missing_alpha_and_keeps_reflection_flow(tmp_path):
    class NoAlphaAdapter(DummyRuntimeAdapter):
        def resolve_outcome(self, trade_date: str, holding_days: int, benchmark: str):
            del benchmark
            decision_day = date.fromisoformat(trade_date)
            maturity_day = _add_business_days(decision_day, holding_days + 1)
            if self.current_day is None or self.current_day < maturity_day:
                return None, None, None
            return 0.012, None, holding_days

    memory_log = _build_memory_log(tmp_path)
    memory_log.store_decision(
        ticker="TSLA",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Hold\nReason: pending TSLA decision.",
        benchmark_execution_semantic="stay_flat",
    )
    adapter = NoAlphaAdapter(memory_log=memory_log)
    adapter.bind_decision_day(date(2024, 1, 10), pre_decision_position_state="flat")
    graph = _build_graph_shell(memory_log, runtime_adapter=adapter)

    graph._resolve_pending_entries("TSLA")
    entries = memory_log.load_entries()

    assert entries[0]["pending"] is False
    assert entries[0]["alpha"] == "n/a"
    assert graph.reflector.calls[0]["alpha_return"] is None


@pytest.mark.unit
def test_run_graph_appends_pending_decision_in_benchmark_mode(tmp_path):
    memory_log = _build_memory_log(tmp_path)
    adapter = DummyRuntimeAdapter(memory_log=memory_log)
    adapter.bind_decision_day(date(2024, 1, 2), pre_decision_position_state="flat")
    graph = _build_graph_shell(memory_log, runtime_adapter=adapter)

    final_state, signal = graph._run_graph("TSLA", "2024-01-02")
    entries = memory_log.load_entries()

    assert final_state["final_trade_decision"].startswith("Rating: Hold")
    assert signal == "Hold"
    assert len(entries) == 1
    assert entries[0]["pending"] is True
    assert entries[0]["benchmark_execution_semantic"] is None


@pytest.mark.unit
def test_run_graph_keeps_legacy_graph_owned_pending_append_without_runtime_adapter(
    tmp_path,
):
    memory_log = _build_memory_log(tmp_path)
    graph = _build_graph_shell(memory_log, runtime_adapter=None)

    graph._run_graph("TSLA", "2024-01-02")
    entries = memory_log.load_entries()

    assert len(entries) == 1
    assert entries[0]["pending"] is True
    assert entries[0]["benchmark_execution_semantic"] is None


@pytest.mark.unit
def test_propagate_requires_bound_decision_day_in_benchmark_mode(tmp_path):
    memory_log = _build_memory_log(tmp_path)
    adapter = DummyRuntimeAdapter(memory_log=memory_log)
    graph = _build_graph_shell(memory_log, runtime_adapter=adapter)

    with pytest.raises(RuntimeError, match="must be bound to the current decision day"):
        graph.propagate("TSLA", "2024-01-02")


@pytest.mark.unit
def test_benchmark_runtime_rejects_partial_outcome_bundle(tmp_path):
    class PartialOutcomeAdapter(DummyRuntimeAdapter):
        def resolve_outcome(self, trade_date: str, holding_days: int, benchmark: str):
            del trade_date, benchmark
            return 0.01, -0.002, holding_days - 1

    memory_log = _build_memory_log(tmp_path)
    memory_log.store_decision(
        ticker="TSLA",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Buy\nReason: pending TSLA decision.",
        benchmark_execution_semantic="enter_long",
    )
    adapter = PartialOutcomeAdapter(memory_log=memory_log)
    adapter.bind_decision_day(date(2024, 1, 10), pre_decision_position_state="flat")
    graph = _build_graph_shell(memory_log, runtime_adapter=adapter)

    with pytest.raises(RuntimeError, match="partial outcome bundle"):
        graph._resolve_pending_entries("TSLA")


@pytest.mark.unit
def test_benchmark_runtime_prefers_explicit_outcome_resolver_seam(tmp_path):
    class BlockingAdapter(DummyRuntimeAdapter):
        def resolve_outcome(self, trade_date: str, holding_days: int, benchmark: str):
            raise AssertionError("direct runtime_adapter.resolve_outcome should not be used")

    memory_log = _build_memory_log(tmp_path)
    memory_log.store_decision(
        ticker="TSLA",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Buy\nReason: pending TSLA decision.",
        benchmark_execution_semantic="enter_long",
    )
    adapter = BlockingAdapter(memory_log=memory_log)
    adapter.bind_decision_day(date(2024, 1, 10), pre_decision_position_state="flat")
    graph = _build_graph_shell(
        memory_log,
        runtime_adapter=adapter,
        outcome_resolver=lambda trade_date, holding_days, benchmark: (
            0.012,
            -0.004,
            holding_days,
        ),
    )

    graph._resolve_pending_entries("TSLA")
    entries = memory_log.load_entries()

    assert entries[0]["pending"] is False
    assert graph.reflector.calls[0]["raw_return"] == pytest.approx(0.012)
    assert graph.reflector.calls[0]["alpha_return"] == pytest.approx(-0.004)


@pytest.mark.unit
def test_benchmark_runtime_writes_skip_memory_write_trace_before_maturity(tmp_path):
    memory_log = _build_memory_log(tmp_path)
    memory_log.store_decision(
        ticker="TSLA",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Buy\nReason: pending TSLA decision.",
        benchmark_execution_semantic="enter_long",
    )
    adapter = DummyRuntimeAdapter(
        memory_log=memory_log,
        trace_writer=TATraceWriter(ticker_dir=tmp_path / "tickers" / "TSLA"),
    )
    adapter.bind_decision_day(date(2024, 1, 9), pre_decision_position_state="flat")
    graph = _build_graph_shell(memory_log, runtime_adapter=adapter)

    graph._resolve_pending_entries("TSLA")

    memory_write_path = tmp_path / "tickers" / "TSLA" / "memory_writes.jsonl"
    records = [
        json.loads(line)
        for line in memory_write_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert records == [
        {
            "date": "2024-01-09",
            "ticker": "TSLA",
            "event_type": "skip_unresolved_pending",
            "decision_date": "2024-01-02",
            "rating": "Buy",
            "note": "maturity_not_reached_or_outcome_unavailable",
        }
    ]


@pytest.mark.unit
def test_benchmark_runtime_writes_resolve_memory_write_and_reflection_trace(tmp_path):
    memory_log = _build_memory_log(tmp_path)
    memory_log.store_decision(
        ticker="TSLA",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Buy\nReason: pending TSLA decision.",
        benchmark_execution_semantic="enter_long",
    )
    adapter = DummyRuntimeAdapter(
        memory_log=memory_log,
        trace_writer=TATraceWriter(ticker_dir=tmp_path / "tickers" / "TSLA"),
    )
    adapter.bind_decision_day(date(2024, 1, 10), pre_decision_position_state="flat")
    graph = _build_graph_shell(memory_log, runtime_adapter=adapter)

    graph._resolve_pending_entries("TSLA")

    memory_write_path = tmp_path / "tickers" / "TSLA" / "memory_writes.jsonl"
    reflection_trace_path = tmp_path / "tickers" / "TSLA" / "reflection_trace.jsonl"
    memory_writes = [
        json.loads(line)
        for line in memory_write_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reflection_traces = [
        json.loads(line)
        for line in reflection_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert memory_writes == [
        {
            "date": "2024-01-10",
            "ticker": "TSLA",
            "event_type": "resolve_matured_decision",
            "decision_date": "2024-01-02",
            "rating": "Buy",
            "note": "reflection_resolved_and_written_back",
        }
    ]
    assert reflection_traces == [
        {
            "ticker": "TSLA",
            "decision_date": "2024-01-02",
            "resolution_date": "2024-01-10",
            "execution_anchor": "next_open",
            "evaluation_anchor": "open_to_open",
            "horizon_days": 5,
            "actual_holding_days": 5,
            "benchmark_ticker": "SPY",
            "raw_return_open_to_open": pytest.approx(0.012),
            "alpha_return_open_to_open": pytest.approx(-0.004),
            "final_trade_decision": "Rating: Buy\nReason: pending TSLA decision.",
            "reflection_text": "Compact recommendation-space lesson.",
        }
    ]


@pytest.mark.unit
def test_benchmark_runtime_skips_optional_resolve_traces_when_memory_update_reports_no_match(
    tmp_path,
):
    memory_log = _build_memory_log(tmp_path)
    memory_log.store_decision(
        ticker="TSLA",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Buy\nReason: pending TSLA decision.",
        benchmark_execution_semantic="enter_long",
    )
    adapter = DummyRuntimeAdapter(
        memory_log=memory_log,
        trace_writer=TATraceWriter(ticker_dir=tmp_path / "tickers" / "TSLA"),
    )
    adapter.bind_decision_day(date(2024, 1, 10), pre_decision_position_state="flat")
    graph = _build_graph_shell(memory_log, runtime_adapter=adapter)

    graph.memory_log.batch_update_with_outcomes = lambda updates: []

    graph._resolve_pending_entries("TSLA")

    assert not (tmp_path / "tickers" / "TSLA" / "memory_writes.jsonl").exists()
    assert not (tmp_path / "tickers" / "TSLA" / "reflection_trace.jsonl").exists()


@pytest.mark.unit
def test_benchmark_runtime_prefers_explicit_instrument_context_builder_seam(tmp_path):
    memory_log = _build_memory_log(tmp_path)
    adapter = DummyRuntimeAdapter(memory_log=memory_log)
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.runtime_adapter = adapter
    graph.instrument_context_builder = lambda ticker, asset_type="stock": (
        f"{ticker}:{asset_type}:from-seam"
    )

    context = TradingAgentsGraph.resolve_instrument_context(graph, "TSLA", "stock")

    assert context == "TSLA:stock:from-seam"


@pytest.mark.unit
def test_benchmark_runtime_propagate_smoke_uses_local_seams_only(tmp_path, monkeypatch):
    memory_log = _build_memory_log(tmp_path)
    memory_log.store_decision(
        ticker="TSLA",
        trade_date="2024-01-02",
        final_trade_decision="Rating: Buy\nReason: pending TSLA decision.",
        benchmark_execution_semantic="enter_long",
    )
    adapter = DummyRuntimeAdapter(memory_log=memory_log)
    adapter.bind_decision_day(date(2024, 1, 10), pre_decision_position_state="flat")
    propagator = CapturingPropagator()
    graph = _build_graph_shell(
        memory_log,
        runtime_adapter=adapter,
        instrument_context_builder=lambda ticker, asset_type="stock": (
            f"{ticker}:{asset_type}:from-seam"
        ),
        outcome_resolver=lambda trade_date, holding_days, benchmark: (
            0.012,
            -0.004,
            holding_days,
        ),
        propagator=propagator,
        use_runtime_instrument_context_method=True,
    )

    monkeypatch.setattr(
        "llm_traders.tradingagent.tradingagents.graph.trading_graph.resolve_instrument_identity",
        lambda ticker: (_ for _ in ()).throw(
            AssertionError("resolve_instrument_identity should not be called")
        ),
    )
    monkeypatch.setattr(
        graph,
        "_fetch_returns",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("_fetch_returns should not be called in benchmark mode")
        ),
    )

    final_state, signal = graph.propagate("TSLA", "2024-01-10")
    entries = memory_log.load_entries()

    assert final_state["final_trade_decision"].startswith("Rating: Hold")
    assert signal == "Hold"
    assert propagator.last_call is not None
    assert (
        propagator.last_call["kwargs"]["instrument_context"]
        == "TSLA:stock:from-seam"
    )
    assert "Past analyses of TSLA" in propagator.last_call["kwargs"]["past_context"]
    assert entries[0]["pending"] is False
