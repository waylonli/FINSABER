from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timezone
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
    / "finmem_finsaber2_2024_2026.json"
)
STRATEGY_NAME = "FinMemStrategy"
Job = tuple[str, str, str]


@dataclass(frozen=True)
class FinMemManifestValidation:
    config_path: Path
    toml_model: str
    artifact_root_policy: str


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported FinMem manifest schema_version.")
    manifest_seed = manifest.get("seed")
    if isinstance(manifest_seed, bool) or not isinstance(manifest_seed, int):
        raise TypeError("FinMem manifest seed must be an integer.")

    windows = manifest.get("windows")
    training_windows = manifest.get("training_windows")
    selections = manifest.get("selections")
    finmem = manifest.get("finmem")
    if not isinstance(windows, dict) or not windows:
        raise ValueError("Manifest must define a non-empty 'windows' mapping.")
    if not isinstance(training_windows, dict):
        raise ValueError("Manifest must define 'training_windows'.")
    if not isinstance(selections, dict) or not selections:
        raise ValueError("Manifest must define a non-empty 'selections' mapping.")
    if not isinstance(finmem, dict):
        raise ValueError("Manifest must define a 'finmem' mapping.")

    for window_name, window_dates in windows.items():
        _parse_date_pair(f"windows[{window_name!r}]", window_dates)

    unknown_training = set(training_windows) - set(windows)
    missing_training = set(windows) - set(training_windows)
    if unknown_training or missing_training:
        raise ValueError(
            "training_windows must match windows exactly; "
            f"unknown={sorted(unknown_training)} missing={sorted(missing_training)}"
        )
    for window_name, train_dates in training_windows.items():
        train_start, train_end = _parse_date_pair(
            f"training_windows[{window_name!r}]",
            train_dates,
        )
        test_start, _ = _parse_date_pair(
            f"windows[{window_name!r}]",
            windows[window_name],
        )
        if train_end >= test_start:
            raise ValueError(
                f"training_windows[{window_name!r}] must end before the test window."
            )

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
                    f"selections[{setup!r}][{window_name!r}] must be a non-empty list."
                )
            if not all(isinstance(ticker, str) and ticker for ticker in tickers):
                raise ValueError(
                    f"selections[{setup!r}][{window_name!r}] must contain ticker strings."
                )
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


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib

        with path.open("rb") as file:
            return tomllib.load(file)
    except ModuleNotFoundError:
        import toml

        return toml.load(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def validate_finmem_settings(
    manifest: dict[str, Any],
    *,
    model: str,
) -> FinMemManifestValidation:
    finmem = manifest["finmem"]
    llm_sampling = manifest.get("llm_sampling")
    if not isinstance(llm_sampling, dict):
        raise TypeError("FinMem manifest llm_sampling must be a mapping.")
    if llm_sampling.get("endpoint") != "chat_completions":
        raise ValueError("FinMem must use Chat Completions.")
    if llm_sampling.get("request_seed_status") != "applied":
        raise ValueError("FinMem request_seed_status must be 'applied'.")
    temperature = llm_sampling.get("temperature")
    request_seed = llm_sampling.get("request_seed")
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise TypeError("FinMem temperature must be numeric.")
    if isinstance(request_seed, bool) or not isinstance(request_seed, int):
        raise TypeError("FinMem request_seed must be an integer.")
    required = {
        "config_path",
        "use_filing_sections",
        "filing_section_map",
        "filing_payload_kind",
        "filing_failure_mode",
        "filing_merge_policy",
        "artifact_config",
    }
    missing = required - set(finmem)
    if missing:
        raise ValueError(f"Manifest finmem section is missing: {sorted(missing)}")

    config_path = resolve_path(finmem["config_path"])
    if not config_path.is_file():
        raise FileNotFoundError(f"FinMem TOML config does not exist: {config_path}")
    finmem_toml = _load_toml(config_path)
    toml_model = finmem_toml.get("chat", {}).get("model")
    if not isinstance(toml_model, str) or not toml_model:
        raise ValueError("FinMem TOML config must define [chat].model.")
    if model != toml_model:
        raise ValueError(
            "FinMem model mismatch: manifest/--model resolves to "
            f"{model!r}, but {config_path} defines [chat].model={toml_model!r}. "
            "Until execution supports materializing an overridden TOML config, "
            "these values must match."
        )
    toml_temperature = finmem_toml.get("chat", {}).get("temperature")
    toml_request_seed = finmem_toml.get("chat", {}).get("seed")
    if toml_temperature != temperature:
        raise ValueError(
            "FinMem temperature mismatch: manifest llm_sampling defines "
            f"{temperature!r}, but {config_path} defines "
            f"[chat].temperature={toml_temperature!r}."
        )
    if toml_request_seed != request_seed:
        raise ValueError(
            "FinMem request seed mismatch: manifest llm_sampling defines "
            f"{request_seed!r}, but {config_path} defines "
            f"[chat].seed={toml_request_seed!r}."
        )

    payload_kind = str(finmem["filing_payload_kind"]).strip().lower()
    if payload_kind not in {"auto", "raw_filing", "section_text"}:
        raise ValueError(
            "finmem.filing_payload_kind must be one of: auto, raw_filing, section_text."
        )
    failure_mode = str(finmem["filing_failure_mode"]).strip().lower()
    if failure_mode not in {"empty", "raw", "raise"}:
        raise ValueError("finmem.filing_failure_mode must be one of: empty, raw, raise.")
    merge_policy = str(finmem["filing_merge_policy"]).strip().lower()
    if merge_policy not in {"concat", "latest"}:
        raise ValueError("finmem.filing_merge_policy must be one of: concat, latest.")

    _validate_filing_section_map(finmem["filing_section_map"])
    artifact_root_policy = _validate_artifact_config(finmem["artifact_config"])
    return FinMemManifestValidation(
        config_path=config_path,
        toml_model=toml_model,
        artifact_root_policy=artifact_root_policy,
    )


def _validate_filing_section_map(section_map: Any) -> None:
    if not isinstance(section_map, dict) or not section_map:
        raise ValueError("finmem.filing_section_map must be a non-empty mapping.")

    from backtest.data_util.filing_section_extractor.upstream_extractor import (
        item_specs_for_request,
        resolve_requested_items,
    )

    for modality, request in section_map.items():
        if not isinstance(request, dict):
            raise TypeError(
                f"finmem.filing_section_map[{modality!r}] must be a mapping."
            )
        form = str(request.get("form", "")).strip().upper()
        item_key = str(request.get("item_key", "")).strip().lower()
        if not form or not item_key:
            raise ValueError(
                f"finmem.filing_section_map[{modality!r}] must define form and item_key."
            )
        requested_items = resolve_requested_items([form], [item_key])
        specs = item_specs_for_request([form], requested_items)
        if len(specs) != 1:
            raise ValueError(
                f"finmem.filing_section_map[{modality!r}] must resolve to one item spec."
            )


def _validate_artifact_config(artifact_config: Any) -> str:
    if not isinstance(artifact_config, dict):
        raise ValueError("finmem.artifact_config must be a mapping.")
    root = artifact_config.get("root")
    if root == "":
        raise ValueError(
            "finmem.artifact_config.root must be omitted or a non-empty path. "
            "An empty string can resolve to the repository root."
        )

    for key in (
        "enabled",
        "save_agent_checkpoint",
        "save_environment_checkpoint",
        "save_reflections",
        "save_query_trace",
        "save_llm_trace",
    ):
        if key in artifact_config and not isinstance(artifact_config[key], bool):
            raise TypeError(f"finmem.artifact_config.{key} must be a boolean.")

    from llm_traders.finsaber_strategies.finmem_artifacts import (
        normalize_finmem_artifact_config,
    )

    normalized = normalize_finmem_artifact_config(artifact_config)
    enabled = bool(normalized["enabled"])
    if root not in (None, ""):
        return f"explicit:{resolve_path(root)}"
    if enabled:
        return "runner_materialized_when_execution_is_enabled"
    return "disabled"


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
    jobs: list[Job] = []
    seen_requested: set[str] = set()

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


def _strategy_output_dir(output_root: Path, setup: str) -> Path:
    return output_root / setup.replace(":", "_") / STRATEGY_NAME


def _ticker_output_dir(
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
) -> Path:
    setup, window, ticker = job
    return _strategy_output_dir(output_root, setup) / _window_key(manifest, window) / ticker


def metrics_path(manifest: dict[str, Any], output_root: Path, job: Job) -> Path:
    return _ticker_output_dir(manifest, output_root, job) / "metrics.json"


def status_path(manifest: dict[str, Any], output_root: Path, job: Job) -> Path:
    return _ticker_output_dir(manifest, output_root, job) / "job_status.json"


def _remove_job_dir_if_exists(path: Path, *, expected_leaf_name: str) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    if resolved.name != expected_leaf_name:
        raise ValueError(f"Refusing to remove unexpected job directory: {resolved}")
    shutil.rmtree(resolved)


def _artifact_ticker_dir(
    manifest: dict[str, Any],
    validation: FinMemManifestValidation,
    output_root: Path,
    job: Job,
) -> Path | None:
    setup, window, ticker = job
    finmem = manifest["finmem"]
    artifact_config = _materialized_artifact_config(
        manifest,
        output_root=output_root,
        setup=setup,
        window=window,
    )
    if not bool(artifact_config["enabled"]):
        return None

    import toml
    from llm_traders.finsaber_strategies.finmem_artifacts import (
        materialize_finmem_run_identity,
    )

    finmem_config = toml.load(validation.config_path)
    finmem_config["general"]["trading_symbol"] = ticker
    finmem_config["general"]["character_string"] = ticker
    test_start, test_end = manifest["windows"][window]
    resolved_strategy_params = {
        "symbol": ticker,
        "config_path": str(validation.config_path),
        "date_from": test_start,
        "date_to": test_end,
        "market_data_info_path": None,
        "market_data_root": None,
        "training_period": tuple(manifest["training_windows"][window]),
        "use_filing_sections": bool(finmem["use_filing_sections"]),
        "filing_section_map": finmem["filing_section_map"],
        "filing_payload_kind": finmem["filing_payload_kind"],
        "filing_failure_mode": finmem["filing_failure_mode"],
        "filing_merge_policy": finmem["filing_merge_policy"],
    }
    identity = materialize_finmem_run_identity(
        artifact_config=artifact_config,
        output_root=str(output_root),
        setup_name=setup,
        strategy_name=STRATEGY_NAME,
        config_path=str(validation.config_path),
        finmem_config=finmem_config,
        resolved_strategy_params=resolved_strategy_params,
    )
    return identity.tickers_dir / ticker


def cleanup_incomplete_job_outputs(
    manifest: dict[str, Any],
    validation: FinMemManifestValidation,
    output_root: Path,
    job: Job,
) -> None:
    if metrics_path(manifest, output_root, job).exists():
        return

    _, _, ticker = job
    # Resume is job-level: an incomplete ticker is rerun from scratch, so stale
    # partial CSVs/traces must not be mixed with the replacement run.
    _remove_job_dir_if_exists(
        _ticker_output_dir(manifest, output_root, job),
        expected_leaf_name=ticker,
    )
    artifact_ticker_dir = _artifact_ticker_dir(
        manifest,
        validation,
        output_root,
        job,
    )
    if artifact_ticker_dir is not None:
        _remove_job_dir_if_exists(artifact_ticker_dir, expected_leaf_name=ticker)


def log_dir(output_root: Path, job: Job) -> Path:
    setup, window, _ = job
    return output_root / "logs" / setup.replace(":", "_") / window


def _mark_failed_status(
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
    *,
    error: str,
) -> None:
    setup, window, ticker = job
    atomic_json(
        status_path(manifest, output_root, job),
        {
            "setup": setup,
            "window": window,
            "window_key": _window_key(manifest, window),
            "ticker": ticker,
            "status": "failed",
            "finished_at_utc": utc_now(),
            "error": error,
        },
    )


def _load_scalar_results_for_jobs(
    manifest: dict[str, Any],
    output_root: Path,
    jobs: list[Job],
) -> dict[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for job in jobs:
        setup, window, ticker = job
        path = metrics_path(manifest, output_root, job)
        if not path.exists():
            continue
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        results.setdefault(_window_key(manifest, window), {})[ticker] = metrics
    return results


def _write_job_artifacts(
    *,
    output_root: Path,
    setup: str,
    window_key: str,
    ticker: str,
    metrics: dict[str, Any],
) -> None:
    import pandas as pd
    from backtest.toolkit.result_writer import (
        DATAFRAME_FILENAMES,
        EMPTY_DATAFRAME_COLUMNS,
        METRIC_KEYS,
    )

    output_dir = _strategy_output_dir(output_root, setup) / window_key / ticker
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in metrics.items():
        if isinstance(value, pd.DataFrame):
            filename = DATAFRAME_FILENAMES.get(key, f"{key}.csv")
            if (
                value.empty
                and len(value.columns) == 0
                and key in EMPTY_DATAFRAME_COLUMNS
            ):
                # Empty optional artifacts should still be readable by pandas.
                value = pd.DataFrame(columns=EMPTY_DATAFRAME_COLUMNS[key])
            value.to_csv(output_dir / filename, index=False)

    try:
        with (output_dir / "metrics.pkl").open("wb") as file:
            pickle.dump(metrics, file)
    except Exception as exc:
        # metrics.json remains the completion sentinel; a pickle failure should
        # not invalidate an otherwise completed and auditable backtest.
        atomic_json(output_dir / "metrics_pickle_error.json", {"error": repr(exc)})

    scalar_metrics = {
        key: value
        for key, value in metrics.items()
        if key in METRIC_KEYS and not isinstance(value, pd.DataFrame)
    }
    # The metrics file is the completion sentinel used for resumable runs.
    atomic_json(output_dir / "metrics.json", scalar_metrics)


def _summary_config(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    validation: FinMemManifestValidation,
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
        "seed": manifest["seed"],
        "llm_sampling": manifest["llm_sampling"],
        "windows": manifest["windows"],
        "training_windows": manifest["training_windows"],
        "evaluation": manifest["evaluation"],
        "finmem": {
            "config_path": str(validation.config_path),
            "toml_model": validation.toml_model,
            "use_filing_sections": manifest["finmem"]["use_filing_sections"],
            "filing_section_map": manifest["finmem"]["filing_section_map"],
            "filing_payload_kind": manifest["finmem"]["filing_payload_kind"],
            "filing_failure_mode": manifest["finmem"]["filing_failure_mode"],
            "filing_merge_policy": manifest["finmem"]["filing_merge_policy"],
            "artifact_config": manifest["finmem"]["artifact_config"],
        },
    }


def build_summary(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    validation: FinMemManifestValidation,
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
            validation=validation,
            data_root=data_root,
            output_root=output_root,
            model=model,
            setup=setup,
        ),
        results,
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


def _materialized_artifact_config(
    manifest: dict[str, Any],
    *,
    output_root: Path,
    setup: str,
    window: str,
) -> dict[str, Any]:
    from llm_traders.finsaber_strategies.finmem_artifacts import (
        default_finmem_run_key,
        normalize_finmem_artifact_config,
    )

    artifact_config = normalize_finmem_artifact_config(
        manifest["finmem"]["artifact_config"]
    )
    if bool(artifact_config["enabled"]) and artifact_config.get("root") in (
        None,
        "",
    ):
        # Keep FinMem's strategy-local traces/checkpoints beside the benchmark
        # outputs without changing FINSABER's standard result tree.
        artifact_config["root"] = str(
            _strategy_output_dir(output_root, setup) / "finmem_artifacts"
        )
    if bool(artifact_config["enabled"]) and artifact_config.get("run_key") in (
        None,
        "",
    ):
        test_start, test_end = manifest["windows"][window]
        artifact_config["run_key"] = default_finmem_run_key(
            date_from=test_start,
            date_to=test_end,
        )
    return artifact_config


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
    finmem = manifest["finmem"]
    train_start, _ = manifest["training_windows"][window]
    test_start, test_end = manifest["windows"][window]
    data_loader = FinsaberParquetDataset(
        data_root,
        start_date=train_start,
        end_date=test_end,
        tickers=[ticker],
        modalities=tuple(evaluation["modalities"]),
        filing_merge_policy=finmem["filing_merge_policy"],
    )
    return {
        "tickers": [ticker],
        "date_from": test_start,
        "date_to": test_end,
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
        "training_years": evaluation.get("training_years"),
        "selection_strategy": None,
        "setup_name": setup,
        "result_filename": None,
        # The runner writes standard artifacts itself so legacy pkl aggregation
        # does not pick the wrong file when multiple windows share one folder.
        "save_results": False,
        "checkpoint_results": False,
        "resume_from_checkpoint": False,
        "log_base_dir": str(output_root),
        "data_loader": data_loader,
    }


def _build_strat_params(
    manifest: dict[str, Any],
    *,
    validation: FinMemManifestValidation,
    output_root: Path,
    setup: str,
    window: str,
) -> dict[str, Any]:
    finmem = manifest["finmem"]
    return {
        "config_path": str(validation.config_path),
        "data_loader": "$data_loader",
        "date_from": "$date_from",
        "date_to": "$date_to",
        "symbol": "$symbol",
        "training_period": tuple(manifest["training_windows"][window]),
        "use_filing_sections": bool(finmem["use_filing_sections"]),
        "filing_section_map": finmem["filing_section_map"],
        "filing_payload_kind": finmem["filing_payload_kind"],
        "filing_failure_mode": finmem["filing_failure_mode"],
        "filing_merge_policy": finmem["filing_merge_policy"],
        "artifact_config": _materialized_artifact_config(
            manifest,
            output_root=output_root,
            setup=setup,
            window=window,
        ),
    }


def run_single_job(
    manifest: dict[str, Any],
    *,
    validation: FinMemManifestValidation,
    data_root: Path,
    output_root: Path,
    setup: str,
    window: str,
    ticker: str,
) -> None:
    prepare_env(manifest["seed"])

    from backtest.finsaber import FINSABER
    from llm_traders.finsaber_strategies.finmem import FinMemStrategy

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
        validation=validation,
        output_root=output_root,
        setup=setup,
        window=window,
    )
    result = FINSABER(trade_config).run_iterative_tickers(
        FinMemStrategy,
        strat_params=strat_params,
        tickers=[ticker],
        delist_check=True,
    )
    if ticker not in result:
        raise RuntimeError(f"No metrics returned for {setup} {window} {ticker}.")

    window_key = _window_key(manifest, window)
    _write_job_artifacts(
        output_root=output_root,
        setup=setup,
        window_key=window_key,
        ticker=ticker,
        metrics=result[ticker],
    )
    print(
        "completed "
        f"setup={setup} window={window} ticker={ticker} "
        f"output={_strategy_output_dir(output_root, setup)} "
        f"finished_at={utc_now()}"
    )


def worker(
    manifest: dict[str, Any],
    *,
    validation: FinMemManifestValidation,
    data_root: Path,
    output_root: Path,
    setup: str,
    window: str,
    ticker: str,
) -> int:
    job = (setup, window, ticker)
    status = {
        "setup": setup,
        "window": window,
        "window_key": _window_key(manifest, window),
        "ticker": ticker,
        "status": "running",
        "started_at_utc": utc_now(),
        "pid": os.getpid(),
    }
    atomic_json(status_path(manifest, output_root, job), status)
    try:
        run_single_job(
            manifest,
            validation=validation,
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


def read_status(
    manifest: dict[str, Any],
    output_root: Path,
    job: Job,
) -> dict[str, Any]:
    setup, window, ticker = job
    if metrics_path(manifest, output_root, job).exists():
        return {
            "setup": setup,
            "window": window,
            "window_key": _window_key(manifest, window),
            "ticker": ticker,
            "status": "completed",
        }
    path = status_path(manifest, output_root, job)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {
        "setup": setup,
        "window": window,
        "window_key": _window_key(manifest, window),
        "ticker": ticker,
        "status": "pending",
    }


def write_runner_manifest(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    data_root: Path,
    output_root: Path,
    model: str,
    jobs: list[Job],
    max_parallel: int,
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
            "seed": manifest["seed"],
            "llm_sampling": manifest["llm_sampling"],
            "strategy": STRATEGY_NAME,
            "data_feed": "FinsaberParquetDataset -> FinMemStrategy",
            "evaluation": manifest["evaluation"],
            "finmem": manifest["finmem"],
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
            "counts": counts,
            "status": statuses,
        },
    )


def write_experiment_config(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    data_root: Path,
    output_root: Path,
    model: str,
) -> None:
    resolved = dict(manifest)
    resolved["source_manifest"] = str(manifest_path)
    resolved["source_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    resolved["resolved_data_root"] = str(data_root)
    resolved["resolved_output_root"] = str(output_root)
    resolved["resolved_model"] = model
    resolved["git_commit"] = git_commit()
    atomic_json(output_root / "experiment_config.json", resolved)


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
    validation: FinMemManifestValidation,
    data_root: Path,
    output_root: Path,
    model: str,
    jobs: list[Job],
    max_parallel: int,
    job_timeout_hours: float,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    write_experiment_config(
        manifest_path=manifest_path,
        manifest=manifest,
        data_root=data_root,
        output_root=output_root,
        model=model,
    )

    pending = [
        job for job in jobs if not metrics_path(manifest, output_root, job).exists()
    ]
    running: dict[subprocess.Popen, tuple[Job, Any, Any, float]] = {}
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
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
    )

    try:
        while pending or running:
            while pending and len(running) < max_parallel:
                job = pending.pop(0)
                setup, window, ticker = job
                try:
                    cleanup_incomplete_job_outputs(
                        manifest,
                        validation,
                        output_root,
                        job,
                    )
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
                    )
                    print(
                        f"FAIL cleanup {setup} {window} {ticker}: {exc!r}",
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
                        env={
                            **os.environ,
                            "PYTHONUNBUFFERED": "1",
                            "PYTHONHASHSEED": str(manifest["seed"]),
                        },
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
                    )
                    print(f"FAIL start {setup} {window} {ticker}: {exc!r}", flush=True)
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
                )
                print(f"START pid={process.pid} {setup} {window} {ticker}", flush=True)

            if not running:
                continue

            time.sleep(2)
            for process, (job, stdout, stderr, started) in list(running.items()):
                if time.monotonic() - started > job_timeout_hours * 3600:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    _mark_timeout(manifest, output_root, job, job_timeout_hours)

                return_code = process.poll()
                if return_code is None:
                    continue
                stdout.close()
                stderr.close()
                del running[process]
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
        )
        return 130

    for setup in sorted({job[0] for job in jobs}):
        build_summary(
            manifest_path=manifest_path,
            manifest=manifest,
            validation=validation,
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
    )
    failed = [
        job
        for job in jobs
        if read_status(manifest, output_root, job).get("status") != "completed"
    ]
    print(f"FINISHED completed={len(jobs) - len(failed)} failed={len(failed)}", flush=True)
    return int(bool(failed))


def _print_finmem_options(manifest: dict[str, Any]) -> None:
    finmem = manifest["finmem"]
    print(f"use_filing_sections={finmem['use_filing_sections']}")
    print(f"filing_payload_kind={finmem['filing_payload_kind']}")
    print(f"filing_failure_mode={finmem['filing_failure_mode']}")
    print(f"filing_merge_policy={finmem['filing_merge_policy']}")
    print("filing_section_map:")
    for modality, request in finmem["filing_section_map"].items():
        print(
            "  "
            f"{modality}: form={request['form']} item_key={request['item_key']}"
        )


def _print_artifact_options(manifest: dict[str, Any]) -> None:
    from llm_traders.finsaber_strategies.finmem_artifacts import (
        normalize_finmem_artifact_config,
    )

    artifact_config = normalize_finmem_artifact_config(
        manifest["finmem"]["artifact_config"]
    )
    print(f"artifact_enabled={artifact_config['enabled']}")
    for key in (
        "save_agent_checkpoint",
        "save_environment_checkpoint",
        "save_reflections",
        "save_query_trace",
        "save_llm_trace",
    ):
        if key in artifact_config:
            print(f"artifact_{key}={artifact_config[key]}")


def print_plan(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    validation: FinMemManifestValidation,
    data_root: Path,
    output_root: Path,
    model: str,
    setups: list[str],
    windows: list[str],
    jobs: list[Job],
    max_parallel: int,
    job_timeout_hours: float,
) -> None:
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={hashlib.sha256(manifest_path.read_bytes()).hexdigest()}")
    print(f"experiment={manifest['experiment_name']}")
    print(f"schema_version={manifest['schema_version']}")
    print(f"data_root={data_root} exists={data_root.is_dir()}")
    print(f"output_root={output_root}")
    print(f"model={model} seed={manifest['seed']}")
    sampling = manifest["llm_sampling"]
    print(
        "llm_sampling="
        f"endpoint:{sampling['endpoint']},"
        f"temperature:{sampling['temperature']:g},"
        f"request_seed:{sampling['request_seed']},"
        f"request_seed_status:{sampling['request_seed_status']}"
    )
    print(f"max_parallel={max_parallel} job_timeout_hours={job_timeout_hours:g}")
    print(f"finmem_config={validation.config_path}")
    print(f"finmem_toml_model={validation.toml_model}")
    _print_finmem_options(manifest)
    _print_artifact_options(manifest)
    print(f"artifact_root_policy={validation.artifact_root_policy}")
    print(f"setups={','.join(setups)}")
    print(f"windows={','.join(windows)}")
    print(_format_job_count_line(jobs))
    for setup in setups:
        setup_jobs = [job for job in jobs if job[0] == setup]
        print(f"{setup}: {len(setup_jobs)}")
        for window in windows:
            tickers = [job[2] for job in setup_jobs if job[1] == window]
            if not tickers:
                continue
            train_start, train_end = manifest["training_windows"][window]
            test_start, test_end = manifest["windows"][window]
            print(
                "  "
                f"{window}: train={train_start}->{train_end} "
                f"test={test_start}->{test_end} tickers={','.join(tickers)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run resumable FINSABER-2 FinMem ticker-year jobs."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--setups", nargs="+")
    parser.add_argument("--windows", nargs="+")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--job-timeout-hours", type=float, default=12)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the expanded job plan without running any backtests.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--setup", help=argparse.SUPPRESS)
    parser.add_argument("--window", help=argparse.SUPPRESS)
    parser.add_argument("--ticker", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    manifest = load_manifest(manifest_path)
    data_root = resolved_data_root(manifest, args.data_root)
    output_root = resolved_output_root(manifest, args.output_root)
    model = args.model or manifest["model"]
    validation = validate_finmem_settings(manifest, model=model)

    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1.")
    if args.job_timeout_hours <= 0:
        raise ValueError("--job-timeout-hours must be positive.")
    if args.worker:
        if not (args.setup and args.window and args.ticker):
            raise ValueError("Worker mode requires --setup, --window, and --ticker.")
        if args.setup not in manifest["selections"]:
            raise ValueError(f"Worker setup is not in the manifest: {args.setup}")
        if args.window not in manifest["windows"]:
            raise ValueError(f"Worker window is not in the manifest: {args.window}")
        if args.ticker not in manifest["selections"][args.setup][args.window]:
            raise ValueError(
                "Worker ticker is not present in the selected setup/window: "
                f"{args.setup} {args.window} {args.ticker}"
            )
        if not data_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
        return worker(
            manifest,
            validation=validation,
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
            validation=validation,
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
        validation=validation,
        data_root=data_root,
        output_root=output_root,
        model=model,
        jobs=jobs,
        max_parallel=args.max_parallel,
        job_timeout_hours=args.job_timeout_hours,
    )


if __name__ == "__main__":
    sys.exit(main())
