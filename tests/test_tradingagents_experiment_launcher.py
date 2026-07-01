from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pandas as pd
import pytest

from backtest.data_util import FinsaberDataset
from backtest.toolkit.llm_cost_monitor import add_llm_cost
import llm_traders.finsaber_strategies.tradingagents as tradingagents_strategy_module
from llm_traders.tradingagent.tradingagents.agents.utils.memory import TradingMemoryLog


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "examples" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import experiment_runner as experiment_runner_module  # noqa: E402
from experiment_runner import ExperimentRunner  # noqa: E402
import backtest.finsaber as finsaber_module  # noqa: E402


_LAUNCHER_TICKERS = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]
_REPO_ROOT = Path(__file__).resolve().parents[1]


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
        "volume": 10_000.0,
    }


def _build_launcher_loader(periods: int = 25) -> FinsaberDataset:
    dates = [
        timestamp.date() for timestamp in pd.bdate_range("2024-01-02", periods=periods)
    ]
    data = {}
    for idx, current_date in enumerate(dates):
        daily_price = {
            ticker: _price_row(100.0 * (position + 1) + idx)
            for position, ticker in enumerate(_LAUNCHER_TICKERS)
        }
        data[current_date] = {
            "price": daily_price,
            "news": {},
        }
    return FinsaberDataset(data=data)


def _patch_fake_graph(
    monkeypatch,
    *,
    rating: str,
    final_trade_decision: str,
    synthetic_cost: dict[str, object] | None = None,
):
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
            if synthetic_cost is not None:
                add_llm_cost(
                    model=str(synthetic_cost["model"]),
                    prompt_tokens=int(synthetic_cost["prompt_tokens"]),
                    completion_tokens=int(synthetic_cost["completion_tokens"]),
                    provider=str(synthetic_cost.get("provider", "openai")),
                    metadata={
                        "ticker": company_name,
                        "trade_date": str(trade_date),
                    },
                )
            return {"final_trade_decision": final_trade_decision}, rating

    monkeypatch.setattr(
        tradingagents_strategy_module,
        "_get_tradingagents_graph_class",
        lambda: FakeTradingAgentsGraph,
    )


@pytest.mark.unit
def test_experiment_runner_tradingagents_launcher_materializes_outputs(
    tmp_path,
    monkeypatch,
):
    from llm_traders.finsaber_strategies.tradingagents import TradingAgentsStrategy

    loader = _build_launcher_loader(periods=25)
    _patch_fake_graph(
        monkeypatch,
        rating="Buy",
        final_trade_decision="Rating: Buy\nReason: synthetic launcher smoke thesis.",
    )
    monkeypatch.setattr(
        experiment_runner_module,
        "create_finsaber2_data_loader",
        lambda data_root, tickers=None: loader,
    )
    monkeypatch.setattr(finsaber_module.FINSABER, "_plot_equity_curve", lambda *args, **kwargs: None)

    output_dir = tmp_path / "output"
    artifact_root = (
        output_dir
        / "cherry_pick_both_finmem"
        / "TradingAgentsStrategy"
        / "tradingagents_window_2024"
        / "tradingagents_artifacts"
    )
    strat_config_path = tmp_path / "tradingagents_launcher_smoke.json"
    strat_config_path.write_text(
        json.dumps(
            {
                "data_loader": "$data_loader",
                "date_from": "$date_from",
                "date_to": "$date_to",
                "symbol": "$symbol",
                "artifact_config": {
                    "enabled": True,
                    "root": str(artifact_root),
                    "run_key": None,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    runner = ExperimentRunner(output_dir=str(output_dir), data_root=str(tmp_path / "ignored"))
    runner.run(
        setup_name="cherry_pick_both_finmem",
        strategy_class=TradingAgentsStrategy,
        custom_trade_config={
            "date_from": "2024-01-02",
            "date_to": "2024-02-05",
            "silence": True,
        },
        strat_config_path=str(strat_config_path),
    )

    manifest_paths = sorted(artifact_root.glob("ta_*/*/manifest.json"))
    namespace_meta_paths = sorted(artifact_root.glob("ta_*/*/namespace_meta.json"))
    memory_paths = sorted(artifact_root.glob("ta_*/*/tickers/*/memory/trading_memory.md"))
    ticker_namespace_meta_paths = sorted(
        artifact_root.glob("ta_*/*/tickers/*/ticker_namespace_meta.json")
    )

    assert len(manifest_paths) == 1
    assert len(namespace_meta_paths) == 1
    assert len(memory_paths) == 5
    assert len(ticker_namespace_meta_paths) == 5

    run_root = manifest_paths[0].parent
    benchmark_results_dir = run_root / "benchmark_results"
    launcher_dir = run_root / "launcher"
    assert (benchmark_results_dir / "2024-01-02_2024-02-05.pkl").exists()
    assert (benchmark_results_dir / "results.csv").exists()
    assert (
        benchmark_results_dir
        / "checkpoints"
        / "2024-01-02_2024-02-05"
        / "TSLA.pkl"
    ).exists()
    assert not (
        output_dir / "cherry_pick_both_finmem" / "TradingAgentsStrategy" / "results.csv"
    ).exists()
    assert (launcher_dir / "run.sh").exists()
    assert (launcher_dir / "run.log").exists()
    assert (launcher_dir / "strat_config.materialized.json").exists()

    tickers_seen = {path.parents[1].name for path in memory_paths}
    assert tickers_seen == {"TSLA", "NFLX", "AMZN", "MSFT", "COIN"}
    assert all(
        "BENCHMARK_EXECUTION_SEMANTIC:"
        not in path.read_text(encoding="utf-8")
        for path in memory_paths
    )
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "runtime_session_summary" not in manifest
        assert "symbol" not in manifest["namespace"]
        assert "ticker_dir" not in manifest["namespace"]
        assert "memory_log_path" not in manifest["namespace"]
        assert "full_state_log_dir" not in manifest["namespace"]
        assert manifest["namespace"]["benchmark_results_dir"] == str(
            benchmark_results_dir
        )
        assert manifest["namespace"]["launcher_dir"] == str(launcher_dir)

    run_namespace_meta = json.loads(
        namespace_meta_paths[0].read_text(encoding="utf-8")
    )
    assert "symbol" not in run_namespace_meta
    assert "ticker_dir" not in run_namespace_meta
    assert "memory_log_path" not in run_namespace_meta
    assert "full_state_log_dir" not in run_namespace_meta
    assert run_namespace_meta["launcher_dir"] == str(launcher_dir)

    for ticker_namespace_meta_path in ticker_namespace_meta_paths:
        ticker_meta = json.loads(
            ticker_namespace_meta_path.read_text(encoding="utf-8")
        )
        assert ticker_meta["symbol"] in {"TSLA", "NFLX", "AMZN", "MSFT", "COIN"}
        assert "ticker_dir" in ticker_meta
        assert "memory_log_path" in ticker_meta
        assert "full_state_log_dir" in ticker_meta
        runtime_summary = ticker_meta["runtime_session_summary"]
        assert runtime_summary["default_benchmark_ticker"] == "SPY"
        assert runtime_summary["default_benchmark_available"] is False
        assert "SPY" not in runtime_summary["runtime_tickers"]

    launcher_log_lines = [
        json.loads(line)
        for line in (launcher_dir / "run.log").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [entry["status"] for entry in launcher_log_lines] == [
        "started",
        "completed",
    ]
    replay_strat_config_path = launcher_dir / "strat_config.materialized.json"
    materialized_launcher_config = json.loads(
        replay_strat_config_path.read_text(encoding="utf-8")
    )
    assert materialized_launcher_config["artifact_config"]["root"] == str(artifact_root)
    assert (
        materialized_launcher_config["artifact_config"]["run_key"] == run_root.name
    )
    assert all(
        entry["input_strat_config_path"] == str(strat_config_path)
        for entry in launcher_log_lines
    )
    assert all(
        entry["replay_strat_config_path"] == str(replay_strat_config_path)
        for entry in launcher_log_lines
    )
    expected_repo_root = str(_REPO_ROOT)
    assert all(
        entry["working_directory"] == expected_repo_root
        for entry in launcher_log_lines
    )
    launcher_script = (launcher_dir / "run.sh").read_text(encoding="utf-8")
    assert str(sys.executable) in launcher_script
    assert "cd " in launcher_script
    assert expected_repo_root in launcher_script
    assert "run_llm_traders_exp.py" in launcher_script
    assert "--setup cherry_pick_both_finmem" in launcher_script
    assert str(replay_strat_config_path) in launcher_script


@pytest.mark.unit
def test_experiment_runner_tradingagents_launcher_rejects_conflicting_result_output_dir_override(
    tmp_path,
    monkeypatch,
):
    from llm_traders.finsaber_strategies.tradingagents import TradingAgentsStrategy

    loader = _build_launcher_loader(periods=25)
    _patch_fake_graph(
        monkeypatch,
        rating="Buy",
        final_trade_decision="Rating: Buy\nReason: synthetic launcher override smoke thesis.",
    )
    monkeypatch.setattr(
        experiment_runner_module,
        "create_finsaber2_data_loader",
        lambda data_root, tickers=None: loader,
    )
    monkeypatch.setattr(
        finsaber_module.FINSABER,
        "_plot_equity_curve",
        lambda *args, **kwargs: None,
    )

    output_dir = tmp_path / "output"
    artifact_root = (
        output_dir
        / "cherry_pick_both_finmem"
        / "TradingAgentsStrategy"
        / "tradingagents_window_2024"
        / "tradingagents_artifacts"
    )
    strat_config_path = tmp_path / "tradingagents_launcher_override_smoke.json"
    strat_config_path.write_text(
        json.dumps(
            {
                "data_loader": "$data_loader",
                "date_from": "$date_from",
                "date_to": "$date_to",
                "symbol": "$symbol",
                "artifact_config": {
                    "enabled": True,
                    "root": str(artifact_root),
                    "run_key": None,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    runner = ExperimentRunner(output_dir=str(output_dir), data_root=str(tmp_path / "ignored"))
    with pytest.raises(
        ValueError,
        match="result_output_dir must match the materialized benchmark_results directory",
    ):
        runner.run(
            setup_name="cherry_pick_both_finmem",
            strategy_class=TradingAgentsStrategy,
            custom_trade_config={
                "date_from": "2024-01-02",
                "date_to": "2024-02-05",
                "silence": True,
                "result_output_dir": str(tmp_path / "custom-benchmark-results"),
            },
            strat_config_path=str(strat_config_path),
        )


@pytest.mark.unit
def test_experiment_runner_tradingagents_launcher_materializes_cost_metrics(
    tmp_path,
    monkeypatch,
):
    from llm_traders.finsaber_strategies.tradingagents import TradingAgentsStrategy

    loader = _build_launcher_loader(periods=25)
    _patch_fake_graph(
        monkeypatch,
        rating="Buy",
        final_trade_decision="Rating: Buy\nReason: synthetic launcher cost smoke thesis.",
        synthetic_cost={
            "model": "gpt-4o-mini",
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "provider": "openai",
        },
    )
    monkeypatch.setattr(
        experiment_runner_module,
        "create_finsaber2_data_loader",
        lambda data_root, tickers=None: loader,
    )
    monkeypatch.setattr(
        finsaber_module.FINSABER,
        "_plot_equity_curve",
        lambda *args, **kwargs: None,
    )

    output_dir = tmp_path / "output"
    artifact_root = (
        output_dir
        / "cherry_pick_both_finmem"
        / "TradingAgentsStrategy"
        / "tradingagents_window_2024"
        / "tradingagents_artifacts"
    )
    strat_config_path = tmp_path / "tradingagents_launcher_cost_smoke.json"
    strat_config_path.write_text(
        json.dumps(
            {
                "data_loader": "$data_loader",
                "date_from": "$date_from",
                "date_to": "$date_to",
                "symbol": "$symbol",
                "artifact_config": {
                    "enabled": True,
                    "root": str(artifact_root),
                    "run_key": "cost_smoke",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    runner = ExperimentRunner(output_dir=str(output_dir), data_root=str(tmp_path / "ignored"))
    runner.run(
        setup_name="cherry_pick_both_finmem",
        strategy_class=TradingAgentsStrategy,
        custom_trade_config={
            "date_from": "2024-01-02",
            "date_to": "2024-02-05",
            "silence": True,
        },
        strat_config_path=str(strat_config_path),
    )

    manifest_paths = sorted(artifact_root.glob("ta_*/cost_smoke/manifest.json"))
    assert len(manifest_paths) == 1
    run_root = manifest_paths[0].parent
    benchmark_results_dir = run_root / "benchmark_results"
    launcher_dir = run_root / "launcher"

    checkpoint_path = (
        benchmark_results_dir / "checkpoints" / "2024-01-02_2024-02-05" / "TSLA.pkl"
    )
    with open(checkpoint_path, "rb") as file:
        checkpoint_payload = pickle.load(file)
    metrics = checkpoint_payload["metrics"]

    assert metrics["total_llm_cost"] > 0.0
    assert metrics["total_external_cost"] > 0.0
    assert metrics["total_trading_cost"] >= metrics["total_external_cost"]
    assert not metrics["llm_cost_records"].empty
    assert not metrics["external_costs"].empty
    assert "llm_inference_cost" in set(metrics["external_costs"]["reason"])
    assert "gpt-4o-mini" in set(metrics["llm_cost_records"]["model"])

    run_summary_path = benchmark_results_dir / "run_summary.csv"
    run_summary = pd.read_csv(run_summary_path)
    tsla_row = run_summary.loc[run_summary["ticker"] == "TSLA"].iloc[0]
    assert tsla_row["total_llm_cost"] > 0.0
    assert tsla_row["total_external_cost"] > 0.0
    assert tsla_row["total_trading_cost"] >= tsla_row["total_external_cost"]
    assert (launcher_dir / "run.sh").exists()
    assert (launcher_dir / "run.log").exists()


@pytest.mark.unit
def test_run_llm_traders_exp_module_import_sets_default_mpl_backend():
    command = [
        sys.executable,
        "-c",
        (
            "import os, runpy, sys; "
            "os.environ.pop('MPLBACKEND', None); "
            "sys.path.insert(0, 'examples/experiments'); "
            "runpy.run_path('examples/experiments/run_llm_traders_exp.py', "
            "run_name='launcher_probe'); "
            "print(os.environ.get('MPLBACKEND', ''))"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert stdout_lines[-1] == "Agg"


@pytest.mark.unit
def test_run_llm_traders_exp_help_entrypoint_smoke():
    command = [
        sys.executable,
        str(_REPO_ROOT / "examples" / "experiments" / "run_llm_traders_exp.py"),
        "--help",
    ]
    env = os.environ.copy()
    env.pop("MPLBACKEND", None)
    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Run LLM trader experiments" in result.stdout
    assert "--setup" in result.stdout
    assert "--strategy" in result.stdout
