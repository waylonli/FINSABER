import pytest
from datetime import date

from backtest.data_util import FinsaberDataset
import llm_traders.finsaber_strategies.tradingagents as tradingagents_module
from llm_traders.finsaber_strategies.tradingagents import (
    TRADINGAGENTS_BASELINE_PROFILE_ID,
    TradingAgentsStrategy,
    build_tradingagents_config_key,
    build_tradingagents_graph_config,
    normalize_tradingagents_artifact_config,
    resolve_tradingagents_run_key,
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
def test_build_graph_config_uses_official_baseline_by_default():
    config = build_tradingagents_graph_config()

    assert config["llm_provider"] == "openai"
    assert config["deep_think_llm"] == "gpt-4o-mini"
    assert config["quick_think_llm"] == "gpt-4o-mini"
    assert config["data_policy"]["allow_online_market_fallback"] is False
    assert config["data_policy"]["allow_social"] is False
    assert config["data_policy"]["filing_mode"] == "raw_filing_extraction"


@pytest.mark.unit
def test_config_key_ignores_paths_and_window_identity():
    base = build_tradingagents_graph_config()
    variant = build_tradingagents_graph_config()

    key_a = build_tradingagents_config_key(base)

    variant["results_dir"] = "/tmp/ta-results"
    variant["memory_log_path"] = "/tmp/ta-memory.md"
    variant["artifact_root"] = "/tmp/artifacts"
    key_b = build_tradingagents_config_key(variant)

    assert key_a == key_b


@pytest.mark.unit
def test_config_key_changes_when_prompt_policy_changes(monkeypatch):
    base = build_tradingagents_graph_config()

    key_a = build_tradingagents_config_key(base)

    monkeypatch.setattr(
        tradingagents_module,
        "_TRADINGAGENTS_PROMPT_POLICY_VARIANTS",
        {
            "market": "market_local_na_guard_v2",
            "news": "news_local_macro_guard_v1",
            "fundamentals": "fundamentals_local_filing_guard_v1",
        },
    )
    key_b = build_tradingagents_config_key(base)

    assert key_a != key_b


@pytest.mark.unit
def test_config_key_changes_when_reflection_prompt_policy_changes(monkeypatch):
    base = build_tradingagents_graph_config()

    key_a = build_tradingagents_config_key(base)

    monkeypatch.setattr(
        tradingagents_module,
        "_TRADINGAGENTS_REFLECTION_PROMPT_POLICY",
        "reflection_upstream_alpha_primary_open_to_open_v3",
    )
    key_b = build_tradingagents_config_key(base)

    assert key_a != key_b


@pytest.mark.unit
def test_config_key_changes_when_fundamentals_payload_contract_changes(monkeypatch):
    base = build_tradingagents_graph_config()

    key_a = build_tradingagents_config_key(base)

    monkeypatch.setattr(tradingagents_module, "_TRADINGAGENTS_STATEMENT_SOURCE_CAP", 130_000)
    key_b = build_tradingagents_config_key(base)

    assert key_a != key_b


@pytest.mark.unit
def test_normalize_artifact_config_keeps_minimal_fields_only():
    normalized = normalize_tradingagents_artifact_config(
        {
            "enabled": True,
            "root": (
                "backtest/output/cherry_pick_both_finmem/TradingAgentsStrategy/"
                "tradingagents_window_2024/tradingagents_artifacts"
            ),
            "run_key": "manual_run",
            "save_memory_reads": False,
        }
    )

    assert normalized == {
        "enabled": True,
        "root": (
            "backtest/output/cherry_pick_both_finmem/TradingAgentsStrategy/"
            "tradingagents_window_2024/tradingagents_artifacts"
        ),
        "run_key": "manual_run",
    }


@pytest.mark.unit
def test_resolve_run_key_prefers_explicit_value():
    artifact_config = {
        "enabled": True,
        "root": (
            "backtest/output/cherry_pick_both_finmem/TradingAgentsStrategy/"
            "tradingagents_window_2024/tradingagents_artifacts"
        ),
        "run_key": "fixed",
    }
    assert resolve_tradingagents_run_key(artifact_config) == "fixed"


@pytest.mark.unit
def test_strategy_initializes_baseline_shell_and_manifest_preview(tmp_path):
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

    assert strategy.baseline_profile_id == TRADINGAGENTS_BASELINE_PROFILE_ID
    assert strategy.selected_analysts == ("market", "news", "fundamentals")
    assert strategy.config_key.startswith("ta_")
    assert strategy.run_key == "run_test"
    assert strategy.manifest_preview["baseline_profile_id"] == TRADINGAGENTS_BASELINE_PROFILE_ID
    assert strategy.manifest_preview["selected_analysts"] == ["market", "news", "fundamentals"]
    assert strategy.manifest_preview["prompt_policy"] == {
        "market": "market_local_na_guard_v1",
        "news": "news_local_macro_guard_v1",
        "fundamentals": "fundamentals_local_filing_guard_v1",
        "reflection": "reflection_upstream_alpha_primary_open_to_open_v2",
    }
    assert strategy.manifest_preview["window_config_input"]["symbol"] == "TSLA"
    assert strategy.manifest_preview["window_config_input"]["date_from"] == "2024-01-02"
    assert strategy.manifest_preview["window_config_input"]["data_loader"] == {
        "kind": "runtime_object",
        "type": "FinsaberDataset",
    }
    assert strategy.manifest_preview["artifact_config"]["root"].endswith(
        "tradingagents_window_2024/tradingagents_artifacts"
    )
    assert strategy.manifest_preview["fundamentals_payload_contract"] == {
        "truncation_marker": "[TRUNCATED: local cap reached]",
        "bucket_caps": {
            "Annual Business Context": 60_000,
            "Current Risk Context": 120_000,
            "Annual MD&A Baseline": 90_000,
            "Latest Quarterly MD&A Update": 80_000,
        },
        "statement_source_cap": 140_000,
        "cycle_total_cap": 420_000,
        "trim_unit": "double_newline_blocks",
        "trim_mode": "under_cap_passthrough__over_cap_block_trim",
    }
    assert strategy.namespace.profile_name == "tradingagents_window_2024"
    assert strategy.runtime_adapter is not None
    assert strategy.graph.runtime_adapter is strategy.runtime_adapter
    assert strategy.manifest_preview["runtime_session_summary"]["runtime_tickers"] == ["TSLA"]
    assert (
        strategy.manifest_preview["runtime_session_summary"]["default_benchmark_available"]
        is True
    )


@pytest.mark.unit
def test_normalize_artifact_config_rejects_non_contract_root():
    with pytest.raises(ValueError, match="tradingagents_artifacts"):
        normalize_tradingagents_artifact_config(
            {
                "enabled": True,
                "root": "backtest/output/cherry_pick_both_finmem/TradingAgentsStrategy",
                "run_key": "manual_run",
            }
        )


@pytest.mark.unit
def test_manifest_is_written_even_when_optional_artifacts_are_disabled(tmp_path):
    strategy = TradingAgentsStrategy(
        symbol="TSLA",
        date_from="2024-01-02",
        date_to="2024-12-31",
        data_loader=_build_loader(),
        artifact_config={
            "enabled": False,
            "root": str(tmp_path / "TradingAgentsStrategy" / "tradingagents_window_2024" / "tradingagents_artifacts"),
            "run_key": "run_test",
        },
    )

    assert strategy.namespace.namespace_meta_path.exists()
    assert strategy.namespace.manifest_path.exists()
