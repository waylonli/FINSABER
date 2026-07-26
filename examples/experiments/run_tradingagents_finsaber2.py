from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = (
    REPO_ROOT
    / "examples"
    / "experiments"
    / "manifests"
    / "tradingagents_finsaber2_2024_2026.json"
)
STRATEGY_NAME = "TradingAgentsStrategy"
ARTIFACT_ROOT_LEAF = "tradingagents_artifacts"
PROGRESS_BAR_WIDTH = 24
PROGRESS_INTERVAL_SECONDS = 60.0
Job = tuple[str, str, str]


@dataclass(frozen=True)
class TradingAgentsArtifactPlan:
    job: Job
    enabled: bool
    artifact_config: dict[str, Any]
    profile_name: str
    config_key: str
    run_key: str
    window_key: str
    artifact_root: Path
    base_run_dir: Path
    benchmark_results_dir: Path
    launcher_dir: Path
    ticker_dir: Path
    results_dir: Path
    data_cache_dir: Path
    memory_log_path: Path
    full_state_log_dir: Path
    ticker_namespace_meta_path: Path
    namespace_meta_path: Path
    manifest_path: Path


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported TradingAgents manifest schema_version.")

    windows = manifest.get("windows")
    selections = manifest.get("selections")
    evaluation = manifest.get("evaluation")
    tradingagents = manifest.get("tradingagents")
    if not isinstance(windows, dict) or not windows:
        raise ValueError("Manifest must define a non-empty 'windows' mapping.")
    if not isinstance(selections, dict) or not selections:
        raise ValueError("Manifest must define a non-empty 'selections' mapping.")
    if not isinstance(evaluation, dict):
        raise ValueError("Manifest must define an 'evaluation' mapping.")
    if not isinstance(tradingagents, dict):
        raise ValueError("Manifest must define a 'tradingagents' mapping.")

    for window_name, window_dates in windows.items():
        _parse_date_pair(f"windows[{window_name!r}]", window_dates)
    _validate_selections(windows, selections)
    _validate_evaluation(evaluation)
    validate_tradingagents_settings(manifest)
    return manifest


def _parse_date_pair(field_name: str, value: Any) -> tuple[datetime, datetime]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item date list.")
    try:
        start = datetime.fromisoformat(str(value[0]))
        end = datetime.fromisoformat(str(value[1]))
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain ISO date strings.") from exc
    if start >= end:
        raise ValueError(f"{field_name} start date must be before end date.")
    return start, end


def _validate_selections(
    windows: dict[str, Any],
    selections: dict[str, Any],
) -> None:
    for setup, by_window in selections.items():
        if not isinstance(by_window, dict):
            raise ValueError(f"selections[{setup!r}] must be a mapping.")
        unknown_windows = set(by_window) - set(windows)
        missing_windows = set(windows) - set(by_window)
        if unknown_windows:
            raise ValueError(
                f"selections[{setup!r}] references undefined windows: "
                f"{sorted(unknown_windows)}"
            )
        if missing_windows:
            raise ValueError(
                f"selections[{setup!r}] must define every manifest window; "
                f"missing={sorted(missing_windows)}"
            )
        for window_name, tickers in by_window.items():
            if not isinstance(tickers, list) or not tickers:
                raise ValueError(
                    f"selections[{setup!r}][{window_name!r}] must be a "
                    "non-empty ticker list."
                )
            if not all(isinstance(ticker, str) and ticker for ticker in tickers):
                raise ValueError(
                    f"selections[{setup!r}][{window_name!r}] must contain "
                    "ticker strings."
                )


def _validate_evaluation(evaluation: dict[str, Any]) -> None:
    required = {
        "initial_cash",
        "risk_free_rate",
        "commission_per_share",
        "min_commission",
        "max_commission_rate",
        "execution_timing",
        "slippage_perc",
        "slippage_impact",
        "liquidity_lookback_days",
        "liquidity_min_history_days",
        "liquidity_cap_pct",
        "llm_cost_as_trade_cost",
        "modalities",
    }
    missing = required - set(evaluation)
    if missing:
        raise ValueError(f"Manifest evaluation section is missing: {sorted(missing)}")
    if evaluation["execution_timing"] != "next_open":
        raise ValueError("TradingAgents FINSABER-2 runs must use next_open execution.")
    modalities = evaluation["modalities"]
    if not isinstance(modalities, list) or not modalities:
        raise ValueError("evaluation.modalities must be a non-empty list.")
    missing_modalities = {"price", "news", "filing_k", "filing_q"} - set(modalities)
    if missing_modalities:
        raise ValueError(
            "TradingAgents requires local price, news, and filing modalities; "
            f"missing={sorted(missing_modalities)}"
        )


def validate_tradingagents_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    settings = manifest["tradingagents"]
    required = {
        "strategy",
        "profile_id",
        "selected_analysts",
        "filing_merge_policy",
        "benchmark_ticker",
        "benchmark_policy",
        "artifact_config",
    }
    missing = required - set(settings)
    if missing:
        raise ValueError(f"Manifest tradingagents section is missing: {sorted(missing)}")

    if settings["strategy"] != (
        "llm_traders.finsaber_strategies.tradingagents.TradingAgentsStrategy"
    ):
        raise ValueError("Manifest tradingagents.strategy does not target TradingAgentsStrategy.")
    if manifest.get("model") != "gpt-4o-mini":
        raise ValueError(
            "TradingAgentsStrategy currently materializes gpt-4o-mini in code; "
            "the manifest model must remain gpt-4o-mini until override support exists."
        )
    if settings["profile_id"] != "finsaber_openai_gpt4omini_v1":
        raise ValueError("Unsupported TradingAgents profile_id.")
    if settings["selected_analysts"] != ["market", "news", "fundamentals"]:
        raise ValueError(
            "TradingAgentsStrategy currently fixes selected analysts to "
            "market, news, and fundamentals."
        )
    if settings["filing_merge_policy"] != "latest":
        raise ValueError("TradingAgentsStrategy expects filing_merge_policy='latest'.")
    if settings["benchmark_ticker"] != "SPY":
        raise ValueError(
            "TradingAgentsStrategy currently fixes benchmark_ticker='SPY' in "
            "the graph config; manifest values must match until override "
            "support exists."
        )
    if settings["benchmark_policy"] != "local_only_unavailable_if_absent":
        raise ValueError("Unsupported TradingAgents benchmark_policy.")

    artifact_config = settings["artifact_config"]
    if not isinstance(artifact_config, dict):
        raise ValueError("tradingagents.artifact_config must be a mapping.")
    unsupported_keys = set(artifact_config) - {"enabled", "root", "run_key_policy"}
    if unsupported_keys:
        raise ValueError(
            "tradingagents.artifact_config contains unsupported keys: "
            f"{sorted(unsupported_keys)}"
        )
    if not isinstance(artifact_config.get("enabled", True), bool):
        raise TypeError("tradingagents.artifact_config.enabled must be a boolean.")
    root = artifact_config.get("root")
    if root == "":
        raise ValueError("tradingagents.artifact_config.root must not be empty.")
    if root is not None and Path(str(root)).name != ARTIFACT_ROOT_LEAF:
        raise ValueError(
            "tradingagents.artifact_config.root must point to a "
            "'tradingagents_artifacts' directory."
        )
    if artifact_config.get("run_key_policy") != "job_scoped":
        raise ValueError("tradingagents.artifact_config.run_key_policy must be job_scoped.")
    return settings


def resolved_data_root(manifest: dict[str, Any], data_root: Path | None) -> Path:
    configured = (
        data_root
        or os.environ.get("FINSABER_DATA_ROOT")
        or manifest["data_root_default"]
    )
    return Path(configured).expanduser().resolve()


def resolved_output_root(manifest: dict[str, Any], output_root: Path | None) -> Path:
    configured = output_root or Path(manifest["output_root_default"])
    configured = Path(configured).expanduser()
    if not configured.is_absolute():
        configured = REPO_ROOT / configured
    return configured.resolve()


def select_setups(manifest: dict[str, Any], requested: list[str] | None) -> list[str]:
    selections = manifest["selections"]
    setups = requested or list(selections)
    unknown = set(setups) - set(selections)
    if unknown:
        raise ValueError(
            f"Unknown setup(s) {sorted(unknown)}; choose from {sorted(selections)}"
        )
    return setups


def select_windows(manifest: dict[str, Any], requested: list[str] | None) -> list[str]:
    windows = manifest["windows"]
    selected = requested or list(windows)
    unknown = set(selected) - set(windows)
    if unknown:
        raise ValueError(
            f"Unknown window(s) {sorted(unknown)}; choose from {sorted(windows)}"
        )
    return selected


def jobs_for(
    manifest: dict[str, Any],
    *,
    setups: list[str],
    windows: list[str],
    tickers: list[str] | None,
) -> list[Job]:
    requested_tickers = set(tickers or [])
    seen_requested: set[str] = set()
    jobs: list[Job] = []

    for setup in setups:
        by_window = manifest["selections"][setup]
        for window in windows:
            for ticker in by_window.get(window, []):
                if requested_tickers and ticker not in requested_tickers:
                    continue
                seen_requested.add(ticker)
                jobs.append((setup, window, ticker))

    missing_tickers = requested_tickers - seen_requested
    if missing_tickers:
        raise ValueError(
            f"Requested ticker(s) are not present in the selected plan: "
            f"{sorted(missing_tickers)}"
        )
    return jobs


def _format_job_count_line(jobs: list[Job]) -> str:
    unique_window_tickers = {(window, ticker) for _, window, ticker in jobs}
    unique_tickers = {ticker for _, _, ticker in jobs}
    return (
        f"jobs={len(jobs)} "
        f"unique_window_ticker_pairs={len(unique_window_tickers)} "
        f"unique_tickers={len(unique_tickers)}"
    )


def _window_key(manifest: dict[str, Any], window: str) -> str:
    start, end = manifest["windows"][window]
    return f"{start}_{end}"


def _safe_path_component(value: str) -> str:
    text = str(value).strip()
    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._")
    return text or "unknown"


def _strategy_output_dir(output_root: Path, setup: str) -> Path:
    return output_root / _safe_path_component(setup) / STRATEGY_NAME


def ticker_output_dir(manifest: dict[str, Any], output_root: Path, job: Job) -> Path:
    setup, window, ticker = job
    return _strategy_output_dir(output_root, setup) / _window_key(manifest, window) / ticker


def artifact_root_for_setup(
    manifest: dict[str, Any],
    output_root: Path,
    setup: str,
) -> Path:
    configured_root = manifest["tradingagents"]["artifact_config"].get("root")
    if configured_root is not None:
        return resolve_path(configured_root)
    return (
        _strategy_output_dir(output_root, setup)
        / _safe_path_component(manifest["tradingagents"]["profile_id"])
        / ARTIFACT_ROOT_LEAF
    )


def deterministic_run_key(setup: str, window: str, ticker: str) -> str:
    return (
        "job_"
        f"{_safe_path_component(setup)}__"
        f"{_safe_path_component(window)}__"
        f"{_safe_path_component(ticker)}"
    )


def materialized_artifact_config_for_job(
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
) -> dict[str, Any]:
    setup, window, ticker = job
    artifact_config = manifest["tradingagents"]["artifact_config"]
    return {
        # Keep this object strategy-compatible; manifest-only policy fields stay out.
        "enabled": bool(artifact_config.get("enabled", True)),
        "root": str(artifact_root_for_setup(manifest, output_root, setup)),
        "run_key": deterministic_run_key(setup, window, ticker),
    }


def _artifact_namespace_window_key(manifest: dict[str, Any], window: str) -> str:
    start, end = manifest["windows"][window]
    return f"test_{start}_{end}"


def artifact_plan_for_job(
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
) -> TradingAgentsArtifactPlan:
    from llm_traders.finsaber_strategies.tradingagents import (
        materialize_tradingagents_run_identity,
    )

    setup, window, ticker = job
    artifact_config = materialized_artifact_config_for_job(manifest, output_root, job)
    identity = materialize_tradingagents_run_identity(
        artifact_config=artifact_config,
    )
    ticker_dir = identity.base_run_dir / "tickers" / _safe_path_component(ticker)
    return TradingAgentsArtifactPlan(
        job=job,
        enabled=artifact_config["enabled"],
        artifact_config=artifact_config,
        profile_name=identity.profile_name,
        config_key=identity.config_key,
        run_key=identity.run_key,
        window_key=_artifact_namespace_window_key(manifest, window),
        artifact_root=identity.artifact_root,
        base_run_dir=identity.base_run_dir,
        benchmark_results_dir=identity.benchmark_results_dir,
        launcher_dir=identity.base_run_dir / "launcher",
        ticker_dir=ticker_dir,
        results_dir=identity.base_run_dir / "runtime_results",
        data_cache_dir=identity.base_run_dir / "runtime_cache",
        memory_log_path=ticker_dir / "memory" / "trading_memory.md",
        full_state_log_dir=ticker_dir / "full_state_logs",
        ticker_namespace_meta_path=ticker_dir / "ticker_namespace_meta.json",
        namespace_meta_path=identity.base_run_dir / "namespace_meta.json",
        manifest_path=identity.base_run_dir / "manifest.json",
    )


def _raise_on_duplicate_paths(
    paths_by_job: list[tuple[Job, Path]],
    *,
    label: str,
) -> None:
    seen: dict[Path, Job] = {}
    for job, path in paths_by_job:
        previous = seen.get(path)
        if previous is not None:
            raise ValueError(
                f"Duplicate {label} for jobs {previous!r} and {job!r}: {path}"
            )
        seen[path] = job


def validate_artifact_plans(
    manifest: dict[str, Any],
    output_root: Path,
    jobs: list[Job],
) -> list[TradingAgentsArtifactPlan]:
    plans = [artifact_plan_for_job(manifest, output_root, job) for job in jobs]
    _raise_on_duplicate_paths(
        [(job, ticker_output_dir(manifest, output_root, job)) for job in jobs],
        label="standard ticker output directory",
    )
    _raise_on_duplicate_paths(
        [(plan.job, plan.ticker_dir) for plan in plans],
        label="TradingAgents artifact ticker directory",
    )
    _raise_on_duplicate_paths(
        [(plan.job, plan.base_run_dir) for plan in plans],
        label="TradingAgents artifact run directory",
    )

    for plan in plans:
        expected_root = artifact_root_for_setup(manifest, output_root, plan.job[0])
        if plan.artifact_root != expected_root:
            raise ValueError(
                "TradingAgents artifact root drifted from the manifest policy: "
                f"expected={expected_root} actual={plan.artifact_root}"
            )
        if set(plan.artifact_config) != {"enabled", "root", "run_key"}:
            raise ValueError(
                "TradingAgents strategy artifact_config must only contain "
                "enabled, root, and run_key."
            )
    return plans


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "isoformat") and value.__class__.__name__ == "Timestamp":
        return value.isoformat()
    if np is not None:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return [_json_safe(item) for item in value.tolist()]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value.__class__.__name__


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def metrics_path(manifest: dict[str, Any], output_root: Path, job: Job) -> Path:
    return ticker_output_dir(manifest, output_root, job) / "metrics.json"


def status_path(manifest: dict[str, Any], output_root: Path, job: Job) -> Path:
    return ticker_output_dir(manifest, output_root, job) / "job_status.json"


def _remove_job_dir_if_exists(path: Path, *, expected_leaf_name: str) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    if resolved.name != expected_leaf_name:
        raise ValueError(f"Refusing to remove unexpected job directory: {resolved}")
    shutil.rmtree(resolved)


def cleanup_incomplete_job_outputs(
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
) -> None:
    if metrics_path(manifest, output_root, job).exists():
        return

    _, _, ticker = job
    # Resume is job-level: stale standard outputs and private TA traces for this
    # ticker must not be mixed into the replacement run.
    _remove_job_dir_if_exists(
        ticker_output_dir(manifest, output_root, job),
        expected_leaf_name=ticker,
    )
    plan = artifact_plan_for_job(manifest, output_root, job)
    _remove_job_dir_if_exists(plan.base_run_dir, expected_leaf_name=plan.run_key)


def log_dir(output_root: Path, job: Job) -> Path:
    setup, window, _ = job
    return output_root / "logs" / _safe_path_component(setup) / _safe_path_component(window)


def _status_base(
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
) -> dict[str, Any]:
    setup, window, ticker = job
    plan = artifact_plan_for_job(manifest, output_root, job)
    return {
        "setup": setup,
        "window": window,
        "window_key": _window_key(manifest, window),
        "ticker": ticker,
        "artifact_run_key": plan.run_key,
        "artifact_ticker_dir": str(plan.ticker_dir),
    }


def read_status(
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
) -> dict[str, Any]:
    base = _status_base(manifest, output_root, job)
    path = status_path(manifest, output_root, job)
    stored: dict[str, Any] = {}
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stored = {}
    if metrics_path(manifest, output_root, job).exists():
        return {**base, **stored, "status": "completed"}
    if stored:
        return {**base, **stored}
    return {**base, "status": "pending"}


def _mark_failed_status(
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
    *,
    error: str,
) -> None:
    status = {
        **_status_base(manifest, output_root, job),
        "status": "failed",
        "finished_at_utc": utc_now(),
        "error": error,
    }
    atomic_json(status_path(manifest, output_root, job), status)


def _load_scalar_results_for_jobs(
    manifest: dict[str, Any],
    output_root: Path,
    jobs: list[Job],
) -> dict[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for job in jobs:
        _, window, ticker = job
        path = metrics_path(manifest, output_root, job)
        if not path.exists():
            continue
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        results.setdefault(_window_key(manifest, window), {})[ticker] = metrics
    return results


def _summary_config(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    data_root: Path,
    output_root: Path,
    model: str,
    setup: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": manifest["experiment_name"],
        "strategy": STRATEGY_NAME,
        "setup_name": setup,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "git_commit": git_commit(),
        "repo_root": str(REPO_ROOT),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "model": model,
        "seed": manifest.get("seed"),
        "windows": manifest["windows"],
        "evaluation": manifest["evaluation"],
        "tradingagents": manifest["tradingagents"],
    }


def build_summary(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    data_root: Path,
    output_root: Path,
    model: str,
    setup: str,
    jobs: list[Job],
) -> None:
    from backtest.toolkit.result_writer import write_result_artifacts

    strategy_dir = _strategy_output_dir(output_root, setup)
    setup_jobs = [job for job in jobs if job[0] == setup]
    results = _load_scalar_results_for_jobs(manifest, output_root, setup_jobs)
    if not results:
        return
    write_result_artifacts(
        str(strategy_dir),
        _summary_config(
            manifest_path=manifest_path,
            manifest=manifest,
            data_root=data_root,
            output_root=output_root,
            model=model,
            setup=setup,
        ),
        results,
    )


def write_experiment_config(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    data_root: Path,
    output_root: Path,
    model: str,
    max_parallel: int,
    job_timeout_hours: float,
) -> None:
    resolved = dict(manifest)
    resolved["source_manifest"] = str(manifest_path)
    resolved["source_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    resolved["resolved_data_root"] = str(data_root)
    resolved["resolved_output_root"] = str(output_root)
    resolved["resolved_model"] = model
    resolved["resolved_max_parallel"] = max_parallel
    resolved["resolved_job_timeout_hours"] = job_timeout_hours
    resolved["git_commit"] = git_commit()
    atomic_json(output_root / "experiment_config.json", resolved)


def write_runner_manifest(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    data_root: Path,
    output_root: Path,
    model: str,
    jobs: list[Job],
    max_parallel: int,
    job_timeout_hours: float,
) -> None:
    statuses = [read_status(manifest, output_root, job) for job in jobs]
    counts = {
        state: sum(item.get("status") == state for item in statuses)
        for state in ("pending", "running", "completed", "failed")
    }
    atomic_json(
        output_root / "runner_manifest.json",
        {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "experiment_name": manifest["experiment_name"],
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "git_commit": git_commit(),
            "python_version": sys.version,
            "repo_root": str(REPO_ROOT),
            "data_root": str(data_root),
            "output_root": str(output_root),
            "model": model,
            "seed": manifest.get("seed"),
            "strategy": STRATEGY_NAME,
            "data_feed": "FinsaberParquetDataset -> TradingAgentsStrategy",
            "evaluation": manifest["evaluation"],
            "tradingagents": manifest["tradingagents"],
            "selections": manifest["selections"],
            "jobs": [
                {
                    "setup": setup,
                    "window": window,
                    "window_key": _window_key(manifest, window),
                    "ticker": ticker,
                }
                for setup, window, ticker in jobs
            ],
            "max_parallel": max_parallel,
            "job_timeout_hours": job_timeout_hours,
            "counts": counts,
            "status": statuses,
        },
    )


def prepare_env(seed: int) -> None:
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    os.chdir(REPO_ROOT)
    os.environ.setdefault("MPLBACKEND", "Agg")
    random.seed(seed)
    if not os.environ.get("OA_OPENAI_KEY") and os.environ.get("OPENAI_API_KEY"):
        os.environ["OA_OPENAI_KEY"] = os.environ["OPENAI_API_KEY"]


def worker_env(seed: int) -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env.setdefault("PYTHONHASHSEED", str(seed))
    return env


def _write_job_artifacts(
    *,
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
    metrics: dict[str, Any],
) -> None:
    import pandas as pd
    from backtest.toolkit.result_writer import (
        DATAFRAME_FILENAMES,
        EMPTY_DATAFRAME_COLUMNS,
        METRIC_KEYS,
    )

    output_dir = ticker_output_dir(manifest, output_root, job)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in metrics.items():
        if isinstance(value, pd.DataFrame):
            filename = DATAFRAME_FILENAMES.get(key, f"{key}.csv")
            if value.empty and len(value.columns) == 0 and key in EMPTY_DATAFRAME_COLUMNS:
                # Empty optional artifacts should still be readable by pandas.
                value = pd.DataFrame(columns=EMPTY_DATAFRAME_COLUMNS[key])
            value.to_csv(output_dir / filename, index=False)

    try:
        with (output_dir / "metrics.pkl").open("wb") as file:
            pickle.dump(metrics, file)
    except Exception as exc:
        # metrics.json remains the completion sentinel for resumable orchestration.
        atomic_json(output_dir / "metrics_pickle_error.json", {"error": repr(exc)})

    scalar_metrics = {
        key: value
        for key, value in metrics.items()
        if key in METRIC_KEYS and not isinstance(value, pd.DataFrame)
    }
    atomic_json(output_dir / "metrics.json", scalar_metrics)


def _build_trade_config(
    manifest: dict[str, Any],
    *,
    data_root: Path,
    output_root: Path,
    setup: str,
    window: str,
    ticker: str,
) -> dict[str, Any]:
    from backtest.data_util import FinsaberParquetDataset

    evaluation = manifest["evaluation"]
    settings = manifest["tradingagents"]
    date_from, date_to = manifest["windows"][window]
    data_loader = FinsaberParquetDataset(
        data_root,
        start_date=date_from,
        end_date=date_to,
        tickers=[ticker],
        modalities=tuple(evaluation["modalities"]),
        filing_merge_policy=settings["filing_merge_policy"],
    )
    return {
        "tickers": [ticker],
        "date_from": date_from,
        "date_to": date_to,
        "cash": evaluation["initial_cash"],
        "risk_free_rate": evaluation["risk_free_rate"],
        "commission_per_share": evaluation["commission_per_share"],
        "min_commission": evaluation["min_commission"],
        "max_commission_rate": evaluation["max_commission_rate"],
        "execution_timing": evaluation["execution_timing"],
        "slippage_perc": evaluation["slippage_perc"],
        "slippage_impact": evaluation["slippage_impact"],
        "liquidity_lookback_days": evaluation["liquidity_lookback_days"],
        "liquidity_min_history_days": evaluation["liquidity_min_history_days"],
        "liquidity_cap_pct": evaluation["liquidity_cap_pct"],
        "llm_cost_as_trade_cost": evaluation["llm_cost_as_trade_cost"],
        "print_trades_table": False,
        "silence": True,
        "rolling_window_size": 1,
        "rolling_window_step": 1,
        "training_years": None,
        "selection_strategy": None,
        "setup_name": setup,
        "result_filename": None,
        # The manifest runner writes the canonical per-job files itself.
        "save_results": False,
        "checkpoint_results": False,
        "resume_from_checkpoint": False,
        "log_base_dir": str(output_root),
        "data_loader": data_loader,
    }


def _build_strat_params(
    manifest: dict[str, Any],
    *,
    output_root: Path,
    job: Job,
) -> dict[str, Any]:
    return {
        "data_loader": "$data_loader",
        "date_from": "$date_from",
        "date_to": "$date_to",
        "symbol": "$symbol",
        "artifact_config": materialized_artifact_config_for_job(
            manifest,
            output_root,
            job,
        ),
    }


def run_single_job(
    manifest: dict[str, Any],
    *,
    data_root: Path,
    output_root: Path,
    setup: str,
    window: str,
    ticker: str,
) -> None:
    prepare_env(int(manifest.get("seed", 2026)))

    from backtest.finsaber import FINSABER
    from llm_traders.finsaber_strategies.tradingagents import TradingAgentsStrategy

    job = (setup, window, ticker)
    trade_config = _build_trade_config(
        manifest,
        data_root=data_root,
        output_root=output_root,
        setup=setup,
        window=window,
        ticker=ticker,
    )
    strat_params = _build_strat_params(
        manifest,
        output_root=output_root,
        job=job,
    )
    result = FINSABER(trade_config).run_iterative_tickers(
        TradingAgentsStrategy,
        strat_params=strat_params,
        tickers=[ticker],
        delist_check=True,
    )
    if ticker not in result:
        raise RuntimeError(f"No metrics returned for {setup} {window} {ticker}.")

    _write_job_artifacts(
        manifest=manifest,
        output_root=output_root,
        job=job,
        metrics=result[ticker],
    )
    print(
        "completed "
        f"setup={setup} window={window} ticker={ticker} "
        f"result={ticker_output_dir(manifest, output_root, job)} "
        f"finished_at={utc_now()}"
    )


def worker(
    manifest: dict[str, Any],
    *,
    data_root: Path,
    output_root: Path,
    setup: str,
    window: str,
    ticker: str,
) -> int:
    job = (setup, window, ticker)
    status = {
        **_status_base(manifest, output_root, job),
        "status": "running",
        "started_at_utc": utc_now(),
        "pid": os.getpid(),
    }
    atomic_json(status_path(manifest, output_root, job), status)
    try:
        if metrics_path(manifest, output_root, job).exists():
            status["status"] = "completed"
            status["finished_at_utc"] = utc_now()
            status["skipped_existing_metrics"] = True
            atomic_json(status_path(manifest, output_root, job), status)
            return 0
        run_single_job(
            manifest,
            data_root=data_root,
            output_root=output_root,
            setup=setup,
            window=window,
            ticker=ticker,
        )
        status["status"] = "completed"
        status["finished_at_utc"] = utc_now()
        atomic_json(status_path(manifest, output_root, job), status)
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["finished_at_utc"] = utc_now()
        status["error"] = repr(exc)
        atomic_json(status_path(manifest, output_root, job), status)
        traceback.print_exc()
        return 1


def _mark_timeout(
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
    job_timeout_hours: float,
) -> None:
    _mark_failed_status(
        manifest,
        output_root,
        job,
        error=f"Job exceeded {job_timeout_hours:g}-hour timeout",
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def _progress_bar(done: int, total: int) -> str:
    if total <= 0:
        return "[" + "#" * PROGRESS_BAR_WIDTH + "]"
    filled = round(PROGRESS_BAR_WIDTH * min(done, total) / total)
    return "[" + "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled) + "]"


def _count_full_state_logs(
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
) -> int:
    plan = artifact_plan_for_job(manifest, output_root, job)
    if not plan.full_state_log_dir.is_dir():
        return 0
    return sum(1 for _ in plan.full_state_log_dir.glob("full_states_log_*.json"))


def _format_progress_line(
    *,
    manifest: dict[str, Any],
    output_root: Path,
    jobs: list[Job],
    pending_count: int,
    running_jobs: list[tuple[Job, int | None, float]],
    started_monotonic: float,
    now_monotonic: float | None = None,
) -> str:
    now = time.monotonic() if now_monotonic is None else now_monotonic
    statuses = [read_status(manifest, output_root, job) for job in jobs]
    completed = sum(item.get("status") == "completed" for item in statuses)
    failed = sum(item.get("status") == "failed" for item in statuses)
    done = completed + failed
    running_details = []
    for job, pid, job_started in running_jobs:
        setup, window, ticker = job
        full_states = _count_full_state_logs(manifest, output_root, job)
        running_details.append(
            f"{ticker}(pid={pid or 'n/a'} "
            f"full_states={full_states} "
            f"elapsed={_format_duration(now - job_started)} "
            f"setup={setup} window={window})"
        )
    active = "; ".join(running_details) if running_details else "-"
    return (
        "PROGRESS "
        f"{_progress_bar(done, len(jobs))} "
        f"done={done}/{len(jobs)} "
        f"completed={completed} failed={failed} "
        f"running={len(running_jobs)} pending={pending_count} "
        f"elapsed={_format_duration(now - started_monotonic)} "
        f"active={active}"
    )


def _running_progress_snapshot(
    running: dict[subprocess.Popen, tuple[Job, Any, Any, float]],
) -> list[tuple[Job, int | None, float]]:
    return [
        (job, process.pid, started)
        for process, (job, _stdout, _stderr, started) in running.items()
    ]


def _worker_command(
    *,
    manifest_path: Path,
    data_root: Path,
    output_root: Path,
    model: str,
    job: Job,
) -> list[str]:
    setup, window, ticker = job
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--setup",
        setup,
        "--window",
        window,
        "--ticker",
        ticker,
        "--manifest",
        str(manifest_path),
        "--data-root",
        str(data_root),
        "--model",
        model,
        "--output-root",
        str(output_root),
    ]


def orchestrate(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    data_root: Path,
    output_root: Path,
    model: str,
    jobs: list[Job],
    max_parallel: int,
    job_timeout_hours: float,
) -> int:
    if not jobs:
        raise ValueError("No TradingAgents jobs were selected.")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    validate_artifact_plans(manifest, output_root, jobs)
    output_root.mkdir(parents=True, exist_ok=True)
    write_experiment_config(
        manifest_path=manifest_path,
        manifest=manifest,
        data_root=data_root,
        output_root=output_root,
        model=model,
        max_parallel=max_parallel,
        job_timeout_hours=job_timeout_hours,
    )

    pending = [
        job for job in jobs if not metrics_path(manifest, output_root, job).exists()
    ]
    running: dict[subprocess.Popen, tuple[Job, Any, Any, float]] = {}
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started_monotonic = time.monotonic()
    last_progress_monotonic = started_monotonic
    print(
        f"Total={len(jobs)} complete={len(jobs) - len(pending)} "
        f"pending={len(pending)} parallel={max_parallel}",
        flush=True,
    )
    write_runner_manifest(
        manifest_path=manifest_path,
        manifest=manifest,
        data_root=data_root,
        output_root=output_root,
        model=model,
        jobs=jobs,
        max_parallel=max_parallel,
        job_timeout_hours=job_timeout_hours,
    )
    print(
        _format_progress_line(
            manifest=manifest,
            output_root=output_root,
            jobs=jobs,
            pending_count=len(pending),
            running_jobs=[],
            started_monotonic=started_monotonic,
        ),
        flush=True,
    )

    try:
        while pending or running:
            while pending and len(running) < max_parallel:
                job = pending.pop(0)
                setup, window, ticker = job
                try:
                    cleanup_incomplete_job_outputs(manifest, output_root, job)
                except Exception as exc:
                    _mark_failed_status(
                        manifest,
                        output_root,
                        job,
                        error=(
                            "Failed to clean incomplete job outputs before restart: "
                            f"{exc!r}"
                        ),
                    )
                    write_runner_manifest(
                        manifest_path=manifest_path,
                        manifest=manifest,
                        data_root=data_root,
                        output_root=output_root,
                        model=model,
                        jobs=jobs,
                        max_parallel=max_parallel,
                        job_timeout_hours=job_timeout_hours,
                    )
                    print(f"FAIL cleanup {setup} {window} {ticker}: {exc!r}", flush=True)
                    print(
                        _format_progress_line(
                            manifest=manifest,
                            output_root=output_root,
                            jobs=jobs,
                            pending_count=len(pending),
                            running_jobs=_running_progress_snapshot(running),
                            started_monotonic=started_monotonic,
                        ),
                        flush=True,
                    )
                    continue

                job_log_dir = log_dir(output_root, job)
                job_log_dir.mkdir(parents=True, exist_ok=True)
                stdout = (job_log_dir / f"{ticker}.stdout.log").open(
                    "a",
                    encoding="utf-8",
                )
                stderr = (job_log_dir / f"{ticker}.stderr.log").open(
                    "a",
                    encoding="utf-8",
                )
                try:
                    process = subprocess.Popen(
                        _worker_command(
                            manifest_path=manifest_path,
                            data_root=data_root,
                            output_root=output_root,
                            model=model,
                            job=job,
                        ),
                        cwd=REPO_ROOT,
                        stdout=stdout,
                        stderr=stderr,
                        env=worker_env(int(manifest.get("seed", 2026))),
                        creationflags=creationflags,
                    )
                except Exception as exc:
                    stdout.close()
                    stderr.close()
                    _mark_failed_status(
                        manifest,
                        output_root,
                        job,
                        error=f"Failed to start worker process: {exc!r}",
                    )
                    write_runner_manifest(
                        manifest_path=manifest_path,
                        manifest=manifest,
                        data_root=data_root,
                        output_root=output_root,
                        model=model,
                        jobs=jobs,
                        max_parallel=max_parallel,
                        job_timeout_hours=job_timeout_hours,
                    )
                    print(f"FAIL start {setup} {window} {ticker}: {exc!r}", flush=True)
                    print(
                        _format_progress_line(
                            manifest=manifest,
                            output_root=output_root,
                            jobs=jobs,
                            pending_count=len(pending),
                            running_jobs=_running_progress_snapshot(running),
                            started_monotonic=started_monotonic,
                        ),
                        flush=True,
                    )
                    continue

                running[process] = (job, stdout, stderr, time.monotonic())
                write_runner_manifest(
                    manifest_path=manifest_path,
                    manifest=manifest,
                    data_root=data_root,
                    output_root=output_root,
                    model=model,
                    jobs=jobs,
                    max_parallel=max_parallel,
                    job_timeout_hours=job_timeout_hours,
                )
                print(f"START pid={process.pid} {setup} {window} {ticker}", flush=True)
                print(
                    _format_progress_line(
                        manifest=manifest,
                        output_root=output_root,
                        jobs=jobs,
                        pending_count=len(pending),
                        running_jobs=_running_progress_snapshot(running),
                        started_monotonic=started_monotonic,
                    ),
                    flush=True,
                )

            if not running:
                continue

            time.sleep(2)
            now_monotonic = time.monotonic()
            if now_monotonic - last_progress_monotonic >= PROGRESS_INTERVAL_SECONDS:
                print(
                    _format_progress_line(
                        manifest=manifest,
                        output_root=output_root,
                        jobs=jobs,
                        pending_count=len(pending),
                        running_jobs=_running_progress_snapshot(running),
                        started_monotonic=started_monotonic,
                        now_monotonic=now_monotonic,
                    ),
                    flush=True,
                )
                last_progress_monotonic = now_monotonic
            for process, (job, stdout, stderr, started) in list(running.items()):
                timed_out = False
                if time.monotonic() - started > job_timeout_hours * 3600:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    _mark_timeout(manifest, output_root, job, job_timeout_hours)
                    timed_out = True

                return_code = process.poll()
                if return_code is None:
                    continue
                stdout.close()
                stderr.close()
                del running[process]
                if (
                    return_code != 0
                    and not timed_out
                    and not metrics_path(manifest, output_root, job).exists()
                ):
                    _mark_failed_status(
                        manifest,
                        output_root,
                        job,
                        error=f"Worker exited with return code {return_code}",
                    )
                print(
                    f"{'DONE' if return_code == 0 else 'FAIL'} rc={return_code} "
                    f"{job[0]} {job[1]} {job[2]}",
                    flush=True,
                )
                write_runner_manifest(
                    manifest_path=manifest_path,
                    manifest=manifest,
                    data_root=data_root,
                    output_root=output_root,
                    model=model,
                    jobs=jobs,
                    max_parallel=max_parallel,
                    job_timeout_hours=job_timeout_hours,
                )
                print(
                    _format_progress_line(
                        manifest=manifest,
                        output_root=output_root,
                        jobs=jobs,
                        pending_count=len(pending),
                        running_jobs=_running_progress_snapshot(running),
                        started_monotonic=started_monotonic,
                    ),
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Stopping active workers...", flush=True)
        for process in running:
            process.terminate()
        for process, (_, stdout, stderr, _) in running.items():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stdout.close()
            stderr.close()
        write_runner_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
            data_root=data_root,
            output_root=output_root,
            model=model,
            jobs=jobs,
            max_parallel=max_parallel,
            job_timeout_hours=job_timeout_hours,
        )
        return 130

    for setup in sorted({job[0] for job in jobs}):
        build_summary(
            manifest_path=manifest_path,
            manifest=manifest,
            data_root=data_root,
            output_root=output_root,
            model=model,
            setup=setup,
            jobs=jobs,
        )
    write_runner_manifest(
        manifest_path=manifest_path,
        manifest=manifest,
        data_root=data_root,
        output_root=output_root,
        model=model,
        jobs=jobs,
        max_parallel=max_parallel,
        job_timeout_hours=job_timeout_hours,
    )
    failed = [
        job
        for job in jobs
        if read_status(manifest, output_root, job).get("status") != "completed"
    ]
    print(f"FINISHED completed={len(jobs) - len(failed)} failed={len(failed)}", flush=True)
    return int(bool(failed))


def _print_tradingagents_options(manifest: dict[str, Any]) -> None:
    settings = manifest["tradingagents"]
    artifact_config = settings["artifact_config"]
    print(f"tradingagents_profile_id={settings['profile_id']}")
    print(f"selected_analysts={','.join(settings['selected_analysts'])}")
    print(f"filing_merge_policy={settings['filing_merge_policy']}")
    print(f"benchmark_ticker={settings['benchmark_ticker']}")
    print(f"benchmark_policy={settings['benchmark_policy']}")
    print(f"artifact_enabled={artifact_config.get('enabled', True)}")
    print(f"artifact_root_policy={artifact_config.get('root') or 'per_setup_under_output_root'}")
    print(f"artifact_run_key_policy={artifact_config['run_key_policy']}")


def print_plan(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    data_root: Path,
    output_root: Path,
    model: str,
    setups: list[str],
    windows: list[str],
    jobs: list[Job],
    max_parallel: int,
    job_timeout_hours: float,
) -> None:
    artifact_plans = validate_artifact_plans(manifest, output_root, jobs)
    plans_by_job = {plan.job: plan for plan in artifact_plans}
    config_keys = sorted({plan.config_key for plan in artifact_plans})
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={hashlib.sha256(manifest_path.read_bytes()).hexdigest()}")
    print(f"experiment={manifest['experiment_name']}")
    print(f"schema_version={manifest['schema_version']}")
    print(f"git_commit={git_commit()}")
    print(f"planned_at_utc={utc_now()}")
    print(f"data_root={data_root} exists={data_root.is_dir()}")
    print(f"output_root={output_root}")
    print(f"model={model} seed={manifest.get('seed')}")
    print(f"max_parallel={max_parallel} job_timeout_hours={job_timeout_hours:g}")
    print(f"tradingagents_config_key={','.join(config_keys)}")
    _print_tradingagents_options(manifest)
    print(f"setups={','.join(setups)}")
    print(f"windows={','.join(windows)}")
    print(_format_job_count_line(jobs))
    print(f"artifact_plan_valid=true artifact_jobs={len(artifact_plans)}")

    for setup in setups:
        setup_jobs = [job for job in jobs if job[0] == setup]
        if not setup_jobs:
            continue
        print(f"{setup}: {len(setup_jobs)}")
        print(
            "  "
            f"artifact_root={artifact_root_for_setup(manifest, output_root, setup)}"
        )
        for window in windows:
            tickers_for_window = [job[2] for job in setup_jobs if job[1] == window]
            if not tickers_for_window:
                continue
            start, end = manifest["windows"][window]
            print(
                "  "
                f"{window}: test={start}->{end} "
                f"tickers={','.join(tickers_for_window)}"
            )

    print("job_paths:")
    for job in jobs:
        setup, window, ticker = job
        print(f"  {setup} {window} {ticker}")
        print(f"    result={ticker_output_dir(manifest, output_root, job)}")
        plan = plans_by_job[job]
        print(f"    artifact_run_key={plan.run_key}")
        print(f"    artifact_base_run_dir={plan.base_run_dir}")
        print(f"    artifact_ticker_dir={plan.ticker_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan FINSABER-2 TradingAgents ticker-year jobs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--setups", nargs="+")
    parser.add_argument("--windows", nargs="+")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--job-timeout-hours", type=float, default=12)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--setup", help=argparse.SUPPRESS)
    parser.add_argument("--window", help=argparse.SUPPRESS)
    parser.add_argument("--ticker", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    manifest = load_manifest(manifest_path)
    model = args.model or manifest["model"]
    if model != manifest["model"]:
        raise ValueError(
            "TradingAgents model overrides are not supported yet; "
            f"manifest model is {manifest['model']!r}, requested {model!r}."
        )
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    if args.job_timeout_hours <= 0:
        raise ValueError("--job-timeout-hours must be positive")
    if args.worker and args.plan:
        raise ValueError("--worker cannot be combined with --plan")

    data_root = resolved_data_root(manifest, args.data_root)
    output_root = resolved_output_root(manifest, args.output_root)
    if args.worker:
        if not (args.setup and args.window and args.ticker):
            raise ValueError("Worker mode requires --setup, --window, and --ticker")
        worker_setups = select_setups(manifest, [args.setup])
        worker_windows = select_windows(manifest, [args.window])
        worker_jobs = jobs_for(
            manifest,
            setups=worker_setups,
            windows=worker_windows,
            tickers=[args.ticker],
        )
        if worker_jobs != [(args.setup, args.window, args.ticker)]:
            raise ValueError("Worker setup/window/ticker must resolve to exactly one job.")
        if not data_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
        return worker(
            manifest,
            data_root=data_root,
            output_root=output_root,
            setup=args.setup,
            window=args.window,
            ticker=args.ticker,
        )

    setups = select_setups(manifest, args.setups)
    windows = select_windows(manifest, args.windows)
    jobs = jobs_for(
        manifest,
        setups=setups,
        windows=windows,
        tickers=args.tickers,
    )

    if args.plan:
        print_plan(
            manifest_path,
            manifest,
            data_root=data_root,
            output_root=output_root,
            model=model,
            setups=setups,
            windows=windows,
            jobs=jobs,
            max_parallel=args.max_parallel,
            job_timeout_hours=args.job_timeout_hours,
        )
        return 0

    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    return orchestrate(
        manifest_path=manifest_path,
        manifest=manifest,
        data_root=data_root,
        output_root=output_root,
        model=model,
        jobs=jobs,
        max_parallel=args.max_parallel,
        job_timeout_hours=args.job_timeout_hours,
    )


if __name__ == "__main__":
    raise SystemExit(main())
