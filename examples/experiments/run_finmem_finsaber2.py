from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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


def validate_finmem_settings(
    manifest: dict[str, Any],
    *,
    model: str,
) -> FinMemManifestValidation:
    finmem = manifest["finmem"]
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

    enabled = bool(artifact_config.get("enabled", False))
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
) -> list[tuple[str, str, str]]:
    requested_tickers = set(tickers or [])
    jobs: list[tuple[str, str, str]] = []
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


def _format_job_count_line(jobs: list[tuple[str, str, str]]) -> str:
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


def _load_existing_scalar_results(strategy_dir: Path) -> dict[str, dict[str, dict]]:
    results: dict[str, dict[str, dict]] = {}
    if not strategy_dir.exists():
        return results

    for path in strategy_dir.glob("*/*/metrics.json"):
        relative = path.relative_to(strategy_dir)
        if len(relative.parts) != 3:
            continue
        window, ticker, _ = relative.parts
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        results.setdefault(window, {})[ticker] = metrics
    return results


def _write_job_artifacts(
    *,
    output_root: Path,
    setup: str,
    window_key: str,
    ticker: str,
    trade_config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    from backtest.toolkit.result_writer import write_result_artifacts

    strategy_dir = _strategy_output_dir(output_root, setup)
    results = _load_existing_scalar_results(strategy_dir)
    results.setdefault(window_key, {})[ticker] = metrics
    write_result_artifacts(str(strategy_dir), trade_config, results)


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
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    if not os.environ.get("OA_OPENAI_KEY") and os.environ.get("OPENAI_API_KEY"):
        os.environ["OA_OPENAI_KEY"] = os.environ["OPENAI_API_KEY"]


def _materialized_artifact_config(
    manifest: dict[str, Any],
    *,
    output_root: Path,
    setup: str,
) -> dict[str, Any]:
    artifact_config = dict(manifest["finmem"]["artifact_config"])
    if bool(artifact_config.get("enabled", False)) and artifact_config.get("root") in (
        None,
        "",
    ):
        # Keep FinMem's strategy-local traces/checkpoints beside the benchmark
        # outputs without changing FINSABER's standard result tree.
        artifact_config["root"] = str(
            _strategy_output_dir(output_root, setup) / "finmem_artifacts"
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
    prepare_env(int(manifest.get("seed", 2026)))

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
        trade_config=trade_config,
        metrics=result[ticker],
    )
    print(
        "completed "
        f"setup={setup} window={window} ticker={ticker} "
        f"output={_strategy_output_dir(output_root, setup)} "
        f"finished_at={utc_now()}"
    )


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
    artifact_config = manifest["finmem"]["artifact_config"]
    print(f"artifact_enabled={artifact_config.get('enabled', False)}")
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
    jobs: list[tuple[str, str, str]],
) -> None:
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={hashlib.sha256(manifest_path.read_bytes()).hexdigest()}")
    print(f"experiment={manifest['experiment_name']}")
    print(f"schema_version={manifest['schema_version']}")
    print(f"data_root={data_root} exists={data_root.is_dir()}")
    print(f"output_root={output_root}")
    print(f"model={model} seed={manifest.get('seed')}")
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
            "Plan FINSABER-2 FinMem ticker-year jobs, or run one explicit job."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--setups", nargs="+")
    parser.add_argument("--windows", nargs="+")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the expanded job plan without running any backtests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    manifest = load_manifest(manifest_path)
    data_root = resolved_data_root(manifest, args.data_root)
    output_root = resolved_output_root(manifest, args.output_root)
    model = args.model or manifest["model"]
    validation = validate_finmem_settings(manifest, model=model)
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
        )
        return 0

    if len(jobs) != 1:
        raise ValueError(
            "Execution is currently restricted to one explicit job. "
            f"The current filters expand to {len(jobs)} job(s). "
            "Run with --plan first, then provide --setups, --windows, and "
            "--tickers filters that resolve to exactly one job."
        )
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    setup, window, ticker = jobs[0]
    run_single_job(
        manifest,
        validation=validation,
        data_root=data_root,
        output_root=output_root,
        setup=setup,
        window=window,
        ticker=ticker,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
