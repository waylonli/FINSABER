from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "examples"
    / "experiments"
    / "manifests"
    / "finagent_finsaber2_2024_2026.json"
)
MANIFEST_PATH = DEFAULT_MANIFEST
EXPERIMENT_CONFIG: dict = {}
DATA_ROOT = Path()
DEFAULT_OUT_ROOT = REPO_ROOT / "tmp" / "finagent-finsaber2-2024-2026"
MODEL = "gpt-4o-mini"
SEED = 2026
WINDOWS: dict[str, tuple[str, str]] = {}
SELECTIONS: dict[str, dict[str, list[str]]] = {}
EVALUATION: dict = {}

METRIC_KEYS = {
    "final_value",
    "total_return",
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "total_commission",
    "total_slippage",
    "total_llm_cost",
    "total_trading_cost",
}

DATAFRAME_FILENAMES = {
    "equity_with_time": "equity_curve.csv",
    "trades": "trades.csv",
    "rejected_orders": "rejected_orders.csv",
    "llm_cost_records": "llm_costs.csv",
}


def configure_experiment(
    manifest_path: Path,
    data_root: Path | None = None,
    model: str | None = None,
) -> None:
    global MANIFEST_PATH, EXPERIMENT_CONFIG, DATA_ROOT, DEFAULT_OUT_ROOT
    global MODEL, SEED, WINDOWS, SELECTIONS, EVALUATION

    MANIFEST_PATH = manifest_path.resolve()
    EXPERIMENT_CONFIG = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    WINDOWS = {
        name: tuple(dates)
        for name, dates in EXPERIMENT_CONFIG["windows"].items()
    }
    SELECTIONS = EXPERIMENT_CONFIG["selections"]
    EVALUATION = EXPERIMENT_CONFIG["evaluation"]
    MODEL = model or EXPERIMENT_CONFIG["model"]
    SEED = int(EXPERIMENT_CONFIG.get("seed", 2026))

    configured_data_root = (
        data_root
        or os.environ.get("FINSABER_DATA_ROOT")
        or EXPERIMENT_CONFIG["data_root_default"]
    )
    DATA_ROOT = Path(configured_data_root).expanduser().resolve()

    configured_output = Path(EXPERIMENT_CONFIG["output_root_default"])
    DEFAULT_OUT_ROOT = (
        configured_output
        if configured_output.is_absolute()
        else REPO_ROOT / configured_output
    )

    for setup, selections in SELECTIONS.items():
        unknown_windows = set(selections) - set(WINDOWS)
        if unknown_windows:
            raise ValueError(
                f"{setup} references undefined windows: {sorted(unknown_windows)}"
            )


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


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def jobs_for(setups: list[str]) -> list[tuple[str, str, str]]:
    return [
        (setup, window, ticker)
        for setup in setups
        for window, tickers in SELECTIONS[setup].items()
        for ticker in tickers
    ]


def ticker_dir(out_root: Path, setup: str, window: str, ticker: str) -> Path:
    return out_root / setup / "FinAgentStrategy" / window / ticker


def metrics_path(out_root: Path, job: tuple[str, str, str]) -> Path:
    return ticker_dir(out_root, *job) / "metrics.json"


def status_path(out_root: Path, job: tuple[str, str, str]) -> Path:
    return ticker_dir(out_root, *job) / "job_status.json"


def write_artifacts(
    out_root: Path,
    setup: str,
    window: str,
    ticker: str,
    metrics: dict,
) -> None:
    out_dir = ticker_dir(out_root, setup, window, ticker)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in DATAFRAME_FILENAMES.items():
        value = metrics.get(key)
        if isinstance(value, pd.DataFrame):
            value.to_csv(out_dir / filename, index=False)
    with (out_dir / "metrics.pkl").open("wb") as file:
        pickle.dump(metrics, file)
    scalar = {
        key: json_safe(metrics.get(key))
        for key in METRIC_KEYS
        if key in metrics
    }
    # The completion sentinel is written last so interrupted jobs resume safely.
    atomic_json(out_dir / "metrics.json", scalar)


def build_summary(out_root: Path, setup: str) -> None:
    rows = []
    for window, tickers in SELECTIONS[setup].items():
        for ticker in tickers:
            path = metrics_path(out_root, (setup, window, ticker))
            if not path.exists():
                continue
            row = {"path": f"{window}/{ticker}", "ticker": ticker, "window": window}
            row.update(json.loads(path.read_text(encoding="utf-8")))
            rows.append(row)
    if rows:
        strategy_dir = out_root / setup / "FinAgentStrategy"
        pd.DataFrame(rows).sort_values(["window", "ticker"]).to_csv(
            strategy_dir / "run_summary.csv",
            index=False,
        )


def prepare_env() -> None:
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    random.seed(SEED)
    np.random.seed(SEED)
    if not os.environ.get("OA_OPENAI_KEY") and os.environ.get("OPENAI_API_KEY"):
        os.environ["OA_OPENAI_KEY"] = os.environ["OPENAI_API_KEY"]


def run_one(out_root: Path, setup: str, window: str, ticker: str) -> None:
    from backtest.data_util import FinsaberParquetDataset
    from backtest.finsaber import FINSABER
    from llm_traders.finsaber_strategies.finagent import FinAgentStrategy

    date_from, date_to = WINDOWS[window]
    start = pd.Timestamp(date_from)
    training_years = int(EVALUATION["training_years"])
    train_start = start.replace(
        year=start.year - training_years
    ).date().isoformat()
    data_loader = FinsaberParquetDataset(
        DATA_ROOT,
        start_date=train_start,
        end_date=date_to,
        tickers=[ticker],
        modalities=tuple(EVALUATION["modalities"]),
    )
    job_dir = ticker_dir(out_root, setup, window, ticker)
    trade_config = {
        "tickers": [ticker],
        "date_from": date_from,
        "date_to": date_to,
        "cash": EVALUATION["initial_cash"],
        "risk_free_rate": EVALUATION["risk_free_rate"],
        "commission_per_share": EVALUATION["commission_per_share"],
        "min_commission": EVALUATION["min_commission"],
        "max_commission_rate": EVALUATION["max_commission_rate"],
        "execution_timing": EVALUATION["execution_timing"],
        "slippage_perc": EVALUATION["slippage_perc"],
        "slippage_impact": EVALUATION["slippage_impact"],
        "liquidity_lookback_days": EVALUATION["liquidity_lookback_days"],
        "liquidity_min_history_days": EVALUATION["liquidity_min_history_days"],
        "liquidity_cap_pct": EVALUATION["liquidity_cap_pct"],
        "llm_cost_as_trade_cost": EVALUATION["llm_cost_as_trade_cost"],
        "print_trades_table": False,
        "silence": True,
        "rolling_window_size": 1,
        "rolling_window_step": 1,
        "training_years": training_years,
        "selection_strategy": None,
        "setup_name": setup,
        "result_filename": None,
        "save_results": False,
        "log_base_dir": str(job_dir / "framework_logs"),
        "data_loader": data_loader,
    }
    strat_params = {
        "date_from": "$date_from",
        "date_to": "$date_to",
        "symbol": "$symbol",
        "data_loader": "$data_loader",
        "training_period": training_years,
        "llm_model_id": MODEL,
        "workdir": str(job_dir / "finagent_workdir"),
    }
    result = FINSABER(trade_config).run_iterative_tickers(
        FinAgentStrategy,
        strat_params=strat_params,
        tickers=[ticker],
        delist_check=True,
    )
    if ticker not in result:
        raise RuntimeError(f"No metrics returned for {setup} {window} {ticker}")
    write_artifacts(out_root, setup, window, ticker, result[ticker])


def worker(out_root: Path, setup: str, window: str, ticker: str) -> int:
    prepare_env()
    job = (setup, window, ticker)
    status = {
        "setup": setup,
        "window": window,
        "ticker": ticker,
        "status": "running",
        "started_at_utc": utc_now(),
        "pid": os.getpid(),
    }
    atomic_json(status_path(out_root, job), status)
    try:
        run_one(out_root, *job)
        status["status"] = "completed"
        status["finished_at_utc"] = utc_now()
        atomic_json(status_path(out_root, job), status)
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["finished_at_utc"] = utc_now()
        status["error"] = repr(exc)
        atomic_json(status_path(out_root, job), status)
        traceback.print_exc()
        return 1


def read_status(out_root: Path, job: tuple[str, str, str]) -> dict:
    if metrics_path(out_root, job).exists():
        return {
            "setup": job[0],
            "window": job[1],
            "ticker": job[2],
            "status": "completed",
        }
    path = status_path(out_root, job)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {
        "setup": job[0],
        "window": job[1],
        "ticker": job[2],
        "status": "pending",
    }


def write_manifest(
    out_root: Path,
    all_jobs: list[tuple[str, str, str]],
    max_parallel: int,
) -> None:
    statuses = [read_status(out_root, job) for job in all_jobs]
    counts = {
        state: sum(item["status"] == state for item in statuses)
        for state in ("pending", "running", "completed", "failed")
    }
    atomic_json(
        out_root / "runner_manifest.json",
        {
            "generated_at_utc": utc_now(),
            "experiment_name": EXPERIMENT_CONFIG["experiment_name"],
            "source_manifest": str(MANIFEST_PATH),
            "source_manifest_sha256": hashlib.sha256(
                MANIFEST_PATH.read_bytes()
            ).hexdigest(),
            "git_commit": git_commit(),
            "python_version": sys.version,
            "repo_root": str(REPO_ROOT),
            "data_root": str(DATA_ROOT),
            "output_root": str(out_root),
            "model": MODEL,
            "seed": SEED,
            "data_feed": "FinsaberParquetDataset -> FinsaberTradingDataDataset",
            "evaluation": EVALUATION,
            "max_parallel": max_parallel,
            "counts": counts,
            "selections": SELECTIONS,
            "status": statuses,
        },
    )


def orchestrate(
    out_root: Path,
    setups: list[str],
    max_parallel: int,
    job_timeout_hours: float,
) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    resolved_config = dict(EXPERIMENT_CONFIG)
    resolved_config["resolved_data_root"] = str(DATA_ROOT)
    resolved_config["resolved_output_root"] = str(out_root)
    resolved_config["resolved_model"] = MODEL
    resolved_config["git_commit"] = git_commit()
    atomic_json(out_root / "experiment_config.json", resolved_config)
    all_jobs = jobs_for(setups)
    pending = [job for job in all_jobs if not metrics_path(out_root, job).exists()]
    running: dict[
        subprocess.Popen,
        tuple[tuple[str, str, str], object, object, float],
    ] = {}
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    print(
        f"Total={len(all_jobs)} complete={len(all_jobs) - len(pending)} "
        f"pending={len(pending)} parallel={max_parallel}",
        flush=True,
    )
    write_manifest(out_root, all_jobs, max_parallel)
    try:
        while pending or running:
            while pending and len(running) < max_parallel:
                job = pending.pop(0)
                setup, window, ticker = job
                log_dir = out_root / "logs" / setup / window
                log_dir.mkdir(parents=True, exist_ok=True)
                stdout = (log_dir / f"{ticker}.stdout.log").open("a", encoding="utf-8")
                stderr = (log_dir / f"{ticker}.stderr.log").open("a", encoding="utf-8")
                command = [
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
                    str(MANIFEST_PATH),
                    "--data-root",
                    str(DATA_ROOT),
                    "--model",
                    MODEL,
                    "--output-root",
                    str(out_root),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    creationflags=creationflags,
                )
                running[process] = (job, stdout, stderr, time.monotonic())
                atomic_json(
                    status_path(out_root, job),
                    {
                        "setup": setup,
                        "window": window,
                        "ticker": ticker,
                        "status": "running",
                        "started_at_utc": utc_now(),
                        "pid": process.pid,
                    },
                )
                write_manifest(out_root, all_jobs, max_parallel)
                print(f"START pid={process.pid} {setup} {window} {ticker}", flush=True)
            time.sleep(2)
            for process, (job, stdout, stderr, started) in list(running.items()):
                if time.monotonic() - started > job_timeout_hours * 3600:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    atomic_json(
                        status_path(out_root, job),
                        {
                            "setup": job[0],
                            "window": job[1],
                            "ticker": job[2],
                            "status": "failed",
                            "finished_at_utc": utc_now(),
                            "error": f"Job exceeded {job_timeout_hours:g}-hour timeout",
                        },
                    )
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
                write_manifest(out_root, all_jobs, max_parallel)
    except KeyboardInterrupt:
        print("Stopping active workers...", flush=True)
        for process in running:
            process.terminate()
        for process, (_, stdout, stderr, _) in running.items():
            process.wait(timeout=30)
            stdout.close()
            stderr.close()
        write_manifest(out_root, all_jobs, max_parallel)
        return 130

    for setup in setups:
        build_summary(out_root, setup)
    write_manifest(out_root, all_jobs, max_parallel)
    failed = [
        job for job in all_jobs if read_status(out_root, job)["status"] != "completed"
    ]
    print(f"FINISHED completed={len(all_jobs) - len(failed)} failed={len(failed)}", flush=True)
    return int(bool(failed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable FINSABER-2 FinAgent ticker-year jobs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--job-timeout-hours", type=float, default=12)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--setups", nargs="+")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--setup", help=argparse.SUPPRESS)
    parser.add_argument("--window", help=argparse.SUPPRESS)
    parser.add_argument("--ticker", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_experiment(args.manifest, args.data_root, args.model)
    setups = args.setups or list(SELECTIONS)
    unknown_setups = set(setups) - set(SELECTIONS)
    if unknown_setups:
        raise ValueError(
            f"Unknown setups {sorted(unknown_setups)}; "
            f"choose from {sorted(SELECTIONS)}"
        )
    output_root = args.output_root or DEFAULT_OUT_ROOT
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root = output_root.resolve()
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    if args.job_timeout_hours <= 0:
        raise ValueError("--job-timeout-hours must be positive")
    if args.worker:
        if not (args.setup and args.window and args.ticker):
            raise ValueError("Worker mode requires --setup, --window, and --ticker")
        if args.setup not in SELECTIONS or args.window not in WINDOWS:
            raise ValueError("Worker setup or window is not present in the manifest")
        return worker(output_root, args.setup, args.window, args.ticker)
    if args.plan:
        planned = jobs_for(setups)
        print(f"manifest={MANIFEST_PATH}")
        print(f"data_root={DATA_ROOT}")
        print(f"model={MODEL} seed={SEED}")
        print(f"jobs={len(planned)} max_parallel={args.max_parallel}")
        for setup in setups:
            print(f"{setup}: {len(jobs_for([setup]))}")
        return 0
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {DATA_ROOT}")
    return orchestrate(
        output_root,
        setups,
        args.max_parallel,
        args.job_timeout_hours,
    )


if __name__ == "__main__":
    raise SystemExit(main())
