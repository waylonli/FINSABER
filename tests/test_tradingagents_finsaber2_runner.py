from __future__ import annotations

from datetime import datetime
from itertools import count
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "examples" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import run_tradingagents_finsaber2 as runner  # noqa: E402


def _load_default_manifest() -> dict:
    return runner.load_manifest(runner.DEFAULT_MANIFEST)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _build_loader(ticker: str = "COIN"):
    from backtest.data_util import FinsaberDataset

    data = {}
    for idx, timestamp in enumerate(pd.bdate_range("2025-01-02", "2026-01-01")):
        data[timestamp.date()] = {
            "price": {ticker: _price_row(100.0 + idx)},
            "news": {},
            "filing_k": {},
            "filing_q": {},
        }
    return FinsaberDataset(data=data)


def _patch_fake_tradingagents_graph(monkeypatch, *, rating: str = "Hold") -> None:
    import llm_traders.finsaber_strategies.tradingagents as tradingagents_module
    from llm_traders.tradingagent.tradingagents.agents.utils.memory import (
        TradingMemoryLog,
    )

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
            self.runtime_adapter = None
            self.memory_log = TradingMemoryLog(
                {"memory_log_path": config["memory_log_path"]}
            )

        def bind_runtime_adapter(self, runtime_adapter):
            self.runtime_adapter = runtime_adapter

        def propagate(self, company_name, trade_date, asset_type="stock"):
            final_trade_decision = (
                f"Rating: {rating}\n"
                "Reason: synthetic FINSABER-2 TradingAgents worker smoke."
            )
            self.memory_log.store_decision(
                ticker=company_name,
                trade_date=str(trade_date),
                final_trade_decision=final_trade_decision,
            )
            return {"final_trade_decision": final_trade_decision}, rating

    monkeypatch.setattr(
        tradingagents_module,
        "_get_tradingagents_graph_class",
        lambda: FakeTradingAgentsGraph,
    )


def _minimal_metrics(final_value: float = 100000.0) -> dict:
    return {
        "final_value": final_value,
        "total_return": final_value / 100000.0 - 1.0,
        "annual_return": final_value / 100000.0 - 1.0,
        "annual_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "total_commission": 0.0,
        "total_slippage": 0.0,
        "total_llm_cost": 0.0,
        "total_trading_cost": 0.0,
        "total_external_cost": 0.0,
        "equity_with_time": pd.DataFrame(
            {
                "date": ["2025-01-02"],
                "equity": [final_value],
            }
        ),
    }


def test_tradingagents_plan_expands_shared_cohorts() -> None:
    manifest = _load_default_manifest()
    setups = runner.select_setups(manifest, None)
    windows = runner.select_windows(manifest, None)
    jobs = runner.jobs_for(manifest, setups=setups, windows=windows, tickers=None)

    assert len(jobs) == 54
    assert len({(window, ticker) for _, window, ticker in jobs}) == 47
    assert len({ticker for _, _, ticker in jobs}) == 38


def test_default_tradingagents_manifest_matches_shared_baseline_cohorts() -> None:
    manifest = _load_json(runner.DEFAULT_MANIFEST)
    for reference_name in (
        "finagent_finsaber2_2024_2026.json",
        "finmem_finsaber2_2024_2026.json",
    ):
        reference = _load_json(runner.DEFAULT_MANIFEST.with_name(reference_name))
        assert manifest["windows"] == reference["windows"]
        assert manifest["selections"] == reference["selections"]


def test_default_tradingagents_sampling_matches_runtime_profile() -> None:
    from llm_traders.finsaber_strategies.tradingagents import (
        build_tradingagents_graph_config,
    )

    manifest = _load_default_manifest()
    assert manifest["llm_sampling"] == runner.TRADINGAGENTS_LLM_SAMPLING
    assert (
        build_tradingagents_graph_config()["temperature"]
        == manifest["llm_sampling"]["temperature"]
        == 0.0
    )


def test_tradingagents_manifest_rejects_sampling_drift(tmp_path) -> None:
    manifest = _load_json(runner.DEFAULT_MANIFEST)
    manifest["llm_sampling"]["temperature"] = 0.5
    manifest_path = tmp_path / "sampling_drift.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="llm_sampling"):
        runner.load_manifest(manifest_path)


def test_tradingagents_manifest_requires_integer_seed(tmp_path) -> None:
    manifest = _load_json(runner.DEFAULT_MANIFEST)
    manifest["seed"] = "2026"
    manifest_path = tmp_path / "invalid_seed.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TypeError, match="seed must be an integer"):
        runner.load_manifest(manifest_path)


def test_tradingagents_custom_manifest_can_use_short_smoke_window(tmp_path) -> None:
    manifest = _load_json(runner.DEFAULT_MANIFEST)
    manifest["experiment_name"] = "tradingagents_smoke"
    manifest["windows"] = {
        "smoke_2025-01-02_2025-02-03": ["2025-01-02", "2025-02-03"]
    }
    manifest["selections"] = {
        "magnificent_7_smoke": {
            "smoke_2025-01-02_2025-02-03": ["AMZN", "MSFT", "TSLA"]
        }
    }
    manifest_path = tmp_path / "tradingagents_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = runner.load_manifest(manifest_path)
    jobs = runner.jobs_for(
        loaded,
        setups=["magnificent_7_smoke"],
        windows=["smoke_2025-01-02_2025-02-03"],
        tickers=None,
    )

    assert jobs == [
        ("magnificent_7_smoke", "smoke_2025-01-02_2025-02-03", "AMZN"),
        ("magnificent_7_smoke", "smoke_2025-01-02_2025-02-03", "MSFT"),
        ("magnificent_7_smoke", "smoke_2025-01-02_2025-02-03", "TSLA"),
    ]


def test_tradingagents_artifact_plan_matches_strategy_namespace(tmp_path) -> None:
    from llm_traders.finsaber_strategies.tradingagents import (
        build_tradingagents_namespace,
    )

    manifest = _load_default_manifest()
    job = ("selected_4", "2025-01-01_2026-01-01", "COIN")
    output_root = tmp_path / "out"

    plan = runner.artifact_plan_for_job(manifest, output_root, job)
    start, end = manifest["windows"][job[1]]
    namespace = build_tradingagents_namespace(
        strategy_name=runner.STRATEGY_NAME,
        symbol=job[2],
        date_from=datetime.fromisoformat(start).date(),
        date_to=datetime.fromisoformat(end).date(),
        artifact_root=str(plan.artifact_root),
        config_key=plan.config_key,
        run_key=plan.run_key,
    )

    assert set(plan.artifact_config) == {"enabled", "root", "run_key"}
    assert str(plan.artifact_root) == namespace.artifact_root
    assert plan.profile_name == namespace.profile_name
    assert plan.profile_name == manifest["tradingagents"]["profile_id"]
    assert plan.artifact_root == (
        output_root
        / job[0]
        / runner.STRATEGY_NAME
        / manifest["tradingagents"]["profile_id"]
        / runner.ARTIFACT_ROOT_LEAF
    )
    assert plan.config_key == namespace.config_key
    assert plan.run_key == namespace.run_key
    assert plan.window_key == namespace.window_key
    assert plan.base_run_dir == namespace.base_run_dir
    assert plan.benchmark_results_dir == namespace.benchmark_results_dir
    assert plan.launcher_dir == namespace.launcher_dir
    assert plan.ticker_dir == namespace.ticker_dir
    assert plan.results_dir == namespace.results_dir
    assert plan.data_cache_dir == namespace.data_cache_dir
    assert plan.memory_log_path == namespace.memory_log_path
    assert plan.full_state_log_dir == namespace.full_state_log_dir
    assert plan.ticker_namespace_meta_path == namespace.ticker_namespace_meta_path
    assert plan.namespace_meta_path == namespace.namespace_meta_path
    assert plan.manifest_path == namespace.manifest_path


def test_tradingagents_artifact_plan_is_side_effect_free(tmp_path) -> None:
    manifest = _load_default_manifest()
    job = ("selected_4", "2025-01-01_2026-01-01", "COIN")
    output_root = tmp_path / "out"

    runner.validate_artifact_plans(manifest, output_root, [job])

    assert not output_root.exists()


def test_tradingagents_manifest_rejects_unsupported_artifact_config_keys() -> None:
    manifest = _load_default_manifest()
    manifest["tradingagents"]["artifact_config"]["run_key"] = "manual_override"

    with pytest.raises(ValueError, match="unsupported keys"):
        runner.validate_tradingagents_settings(manifest)


def test_tradingagents_artifact_plan_rejects_duplicate_job(tmp_path) -> None:
    manifest = _load_default_manifest()
    job = ("selected_4", "2025-01-01_2026-01-01", "COIN")

    with pytest.raises(ValueError, match="Duplicate standard ticker output directory"):
        runner.validate_artifact_plans(manifest, tmp_path / "out", [job, job])


def test_tradingagents_worker_materializes_standard_and_private_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    import backtest.data_util as data_util_module

    manifest = _load_default_manifest()
    job = ("selected_4", "2025-01-01_2026-01-01", "COIN")
    output_root = tmp_path / "out"
    loader = _build_loader(job[2])

    _patch_fake_tradingagents_graph(monkeypatch, rating="Hold")
    monkeypatch.setattr(
        data_util_module,
        "FinsaberParquetDataset",
        lambda *args, **kwargs: loader,
    )

    return_code = runner.worker(
        manifest,
        data_root=tmp_path / "data",
        output_root=output_root,
        setup=job[0],
        window=job[1],
        ticker=job[2],
    )

    assert return_code == 0
    standard_dir = runner.ticker_output_dir(manifest, output_root, job)
    assert (standard_dir / "metrics.json").exists()
    assert (standard_dir / "metrics.pkl").exists()
    assert (standard_dir / "equity_curve.csv").exists()
    assert (standard_dir / "trades.csv").exists()
    assert (standard_dir / "rejected_orders.csv").exists()
    assert (standard_dir / "llm_costs.csv").exists()
    assert (standard_dir / "external_costs.csv").exists()
    status = json.loads((standard_dir / "job_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["artifact_run_key"] == runner.deterministic_run_key(*job)

    metrics = json.loads((standard_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["final_value"] == 100000.0
    assert metrics["total_return"] == 0.0

    plan = runner.artifact_plan_for_job(manifest, output_root, job)
    assert plan.manifest_path.exists()
    assert plan.namespace_meta_path.exists()
    assert plan.ticker_namespace_meta_path.exists()
    assert plan.memory_log_path.exists()
    assert plan.full_state_log_dir.exists()
    assert plan.profile_name == manifest["tradingagents"]["profile_id"]

    ticker_meta = json.loads(
        plan.ticker_namespace_meta_path.read_text(encoding="utf-8")
    )
    assert ticker_meta["symbol"] == job[2]
    assert ticker_meta["runtime_session_summary"]["runtime_tickers"] == [job[2]]


def test_tradingagents_cleanup_removes_only_incomplete_job_dirs(tmp_path) -> None:
    manifest = _load_default_manifest()
    job = ("selected_4", "2025-01-01_2026-01-01", "COIN")
    output_root = tmp_path / "out"
    standard_dir = runner.ticker_output_dir(manifest, output_root, job)
    plan = runner.artifact_plan_for_job(manifest, output_root, job)

    standard_dir.mkdir(parents=True)
    (standard_dir / "stale.txt").write_text("partial", encoding="utf-8")
    plan.ticker_dir.mkdir(parents=True)
    (plan.ticker_dir / "stale.txt").write_text("partial", encoding="utf-8")

    runner.cleanup_incomplete_job_outputs(manifest, output_root, job)

    assert not standard_dir.exists()
    assert not plan.base_run_dir.exists()
    assert not plan.ticker_dir.exists()


def test_tradingagents_progress_line_reports_job_and_artifact_progress(
    tmp_path,
) -> None:
    manifest = _load_default_manifest()
    completed_job = ("selected_4", "2025-01-01_2026-01-01", "AMZN")
    running_job = ("selected_4", "2025-01-01_2026-01-01", "COIN")
    output_root = tmp_path / "out"
    runner._write_job_artifacts(
        manifest=manifest,
        output_root=output_root,
        job=completed_job,
        metrics=_minimal_metrics(),
    )

    plan = runner.artifact_plan_for_job(manifest, output_root, running_job)
    plan.full_state_log_dir.mkdir(parents=True)
    for idx in range(3):
        (plan.full_state_log_dir / f"full_states_log_2025-01-0{idx + 2}.json").write_text(
            "{}",
            encoding="utf-8",
        )

    line = runner._format_progress_line(
        manifest=manifest,
        output_root=output_root,
        jobs=[completed_job, running_job],
        pending_count=0,
        running_jobs=[(running_job, 1234, 10.0)],
        started_monotonic=0.0,
        now_monotonic=70.0,
    )

    assert line.startswith("PROGRESS ")
    assert "done=1/2" in line
    assert "completed=1 failed=0 running=1 pending=0" in line
    assert "elapsed=01m10s" in line
    assert "COIN(pid=1234 full_states=3 elapsed=01m00s" in line


def test_tradingagents_orchestrate_writes_runner_manifest_and_summary(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = _load_default_manifest()
    jobs = [
        ("selected_4", "2025-01-01_2026-01-01", "AMZN"),
        ("selected_4", "2025-01-01_2026-01-01", "COIN"),
    ]
    output_root = tmp_path / "out"
    data_root = tmp_path / "data"
    data_root.mkdir()

    monkeypatch.setattr(runner, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setenv("PYTHONHASHSEED", "999")

    class FakeProcess:
        _next_pid = 4000
        last_env = None

        def __init__(self, command, *, cwd, stdout, stderr, env, creationflags):
            self.command = command
            self.pid = FakeProcess._next_pid
            FakeProcess._next_pid += 1
            self._completed = False
            FakeProcess.last_env = env

        def _arg(self, name: str) -> str:
            return self.command[self.command.index(name) + 1]

        def poll(self):
            if not self._completed:
                job = (
                    self._arg("--setup"),
                    self._arg("--window"),
                    self._arg("--ticker"),
                )
                final_value = 101000.0 if job[2] == "AMZN" else 99000.0
                runner._write_job_artifacts(
                    manifest=manifest,
                    output_root=output_root,
                    job=job,
                    metrics=_minimal_metrics(final_value),
                )
                runner.atomic_json(
                    runner.status_path(manifest, output_root, job),
                    {
                        **runner._status_base(manifest, output_root, job),
                        "status": "completed",
                        "started_at_utc": runner.utc_now(),
                        "finished_at_utc": runner.utc_now(),
                        "pid": self.pid,
                    },
                )
                self._completed = True
            return 0

        def terminate(self) -> None:
            self._completed = True

        def kill(self) -> None:
            self._completed = True

        def wait(self, timeout=None) -> int:
            self._completed = True
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)

    return_code = runner.orchestrate(
        manifest_path=runner.DEFAULT_MANIFEST,
        manifest=manifest,
        data_root=data_root,
        output_root=output_root,
        model=manifest["model"],
        jobs=jobs,
        max_parallel=2,
        job_timeout_hours=1,
    )

    assert return_code == 0
    runner_manifest = json.loads(
        (output_root / "runner_manifest.json").read_text(encoding="utf-8")
    )
    assert runner_manifest["counts"]["completed"] == 2
    assert runner_manifest["counts"]["failed"] == 0
    assert runner_manifest["max_parallel"] == 2
    assert runner_manifest["job_timeout_hours"] == 1
    assert runner_manifest["llm_sampling"] == manifest["llm_sampling"]
    assert FakeProcess.last_env["PYTHONUNBUFFERED"] == "1"
    assert FakeProcess.last_env["PYTHONHASHSEED"] == str(manifest["seed"])

    strategy_dir = output_root / "selected_4" / runner.STRATEGY_NAME
    assert (output_root / "experiment_config.json").exists()
    experiment_config = json.loads(
        (output_root / "experiment_config.json").read_text(encoding="utf-8")
    )
    assert experiment_config["resolved_max_parallel"] == 2
    assert experiment_config["resolved_job_timeout_hours"] == 1
    run_config = _load_json(strategy_dir / "run_config.json")
    assert run_config["llm_sampling"] == manifest["llm_sampling"]
    assert (strategy_dir / "run_manifest.json").exists()
    summary = pd.read_csv(strategy_dir / "run_summary.csv")
    assert sorted(summary["ticker"].tolist()) == ["AMZN", "COIN"]
    assert (output_root / "logs" / "selected_4" / "2025-01-01_2026-01-01").exists()


def test_tradingagents_orchestrate_skips_completed_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = _load_default_manifest()
    job = ("selected_4", "2025-01-01_2026-01-01", "COIN")
    output_root = tmp_path / "out"
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner._write_job_artifacts(
        manifest=manifest,
        output_root=output_root,
        job=job,
        metrics=_minimal_metrics(),
    )

    def fail_if_started(*args, **kwargs):
        raise AssertionError("completed jobs should be skipped before worker start")

    monkeypatch.setattr(runner, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(runner.subprocess, "Popen", fail_if_started)

    return_code = runner.orchestrate(
        manifest_path=runner.DEFAULT_MANIFEST,
        manifest=manifest,
        data_root=data_root,
        output_root=output_root,
        model=manifest["model"],
        jobs=[job],
        max_parallel=1,
        job_timeout_hours=1,
    )

    assert return_code == 0
    runner_manifest = json.loads(
        (output_root / "runner_manifest.json").read_text(encoding="utf-8")
    )
    assert runner_manifest["counts"]["completed"] == 1
    assert runner_manifest["counts"]["pending"] == 0


def test_tradingagents_orchestrate_marks_worker_nonzero_as_failed(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = _load_default_manifest()
    job = ("selected_4", "2025-01-01_2026-01-01", "COIN")
    output_root = tmp_path / "out"
    data_root = tmp_path / "data"
    data_root.mkdir()

    monkeypatch.setattr(runner, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    class FailingProcess:
        pid = 5000

        def __init__(self, command, *, cwd, stdout, stderr, env, creationflags):
            self.command = command

        def poll(self):
            return 1

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

        def wait(self, timeout=None) -> int:
            return 1

    monkeypatch.setattr(runner.subprocess, "Popen", FailingProcess)

    return_code = runner.orchestrate(
        manifest_path=runner.DEFAULT_MANIFEST,
        manifest=manifest,
        data_root=data_root,
        output_root=output_root,
        model=manifest["model"],
        jobs=[job],
        max_parallel=1,
        job_timeout_hours=1,
    )

    assert return_code == 1
    status = json.loads(
        runner.status_path(manifest, output_root, job).read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert status["error"] == "Worker exited with return code 1"
    runner_manifest = json.loads(
        (output_root / "runner_manifest.json").read_text(encoding="utf-8")
    )
    assert runner_manifest["counts"]["failed"] == 1
    assert runner_manifest["counts"]["pending"] == 0


def test_tradingagents_orchestrate_rejects_missing_data_root_before_output(
    tmp_path,
) -> None:
    manifest = _load_default_manifest()
    job = ("selected_4", "2025-01-01_2026-01-01", "COIN")
    output_root = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="Dataset root does not exist"):
        runner.orchestrate(
            manifest_path=runner.DEFAULT_MANIFEST,
            manifest=manifest,
            data_root=tmp_path / "missing-data",
            output_root=output_root,
            model=manifest["model"],
            jobs=[job],
            max_parallel=1,
            job_timeout_hours=1,
        )

    assert not output_root.exists()


def test_tradingagents_timeout_error_is_not_overwritten_by_return_code(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = _load_default_manifest()
    job = ("selected_4", "2025-01-01_2026-01-01", "COIN")
    output_root = tmp_path / "out"
    data_root = tmp_path / "data"
    data_root.mkdir()

    monkeypatch.setattr(runner, "git_commit", lambda: "test-commit")
    monotonic_values = count(0.0, 1.0)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    class TimeoutProcess:
        pid = 6000

        def __init__(self, command, *, cwd, stdout, stderr, env, creationflags):
            self._terminated = False

        def poll(self):
            return -15 if self._terminated else None

        def terminate(self) -> None:
            self._terminated = True

        def kill(self) -> None:
            self._terminated = True

        def wait(self, timeout=None) -> int:
            return -15

    monkeypatch.setattr(runner.subprocess, "Popen", TimeoutProcess)

    return_code = runner.orchestrate(
        manifest_path=runner.DEFAULT_MANIFEST,
        manifest=manifest,
        data_root=data_root,
        output_root=output_root,
        model=manifest["model"],
        jobs=[job],
        max_parallel=1,
        job_timeout_hours=1e-9,
    )

    assert return_code == 1
    status = json.loads(
        runner.status_path(manifest, output_root, job).read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert status["error"] == "Job exceeded 1e-09-hour timeout"
