#!/usr/bin/env python
"""Run one FinAgent rolling-window task with isolated IO.

This runner intentionally bypasses ExperimentRunner aggregation. Each Slurm task writes
into a unique directory so parallel windows do not fight over result files, logs, plots,
or FinAgent memory artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from backtest.data_util import FinMemDataset  # noqa: E402
from backtest.data_util.download import ensure_datasets  # noqa: E402
from backtest.finsaber import FINSABER  # noqa: E402
from backtest.strategy.selection import (  # noqa: E402
    FinConSP500Selector,
    LowVolatilitySP500Selector,
    MomentumSP500Selector,
    RandomSP500Selector,
)
from backtest.toolkit.llm_cost_monitor import get_llm_cost  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one FINSABER-1 FinAgent window.")
    parser.add_argument("--setup", required=True, help="Experiment setup, e.g. selected_4 or random_sp500_5.")
    parser.add_argument("--date-from", required=True, help="Window start date, YYYY-MM-DD.")
    parser.add_argument("--date-to", required=True, help="Window end date, YYYY-MM-DD.")
    parser.add_argument("--strat-config-path", required=True, help="FinAgent strategy config JSON.")
    parser.add_argument("--output-root", default="runs/finagent_gpt4o_mini", help="Root for per-job outputs.")
    parser.add_argument("--model-id", default="gpt-4o-mini", help="OpenAI model id passed to FinAgent.")
    parser.add_argument("--tickers", default="AUTO", help="Comma-separated tickers, or AUTO for selector-driven setups.")
    parser.add_argument("--force", action="store_true", help="Rerun even if summary.json says success=true.")
    parser.add_argument("--skip-dataset-check", action="store_true", help="Skip ensure_datasets().")
    return parser.parse_args()


def safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return f"<DataFrame rows={len(value)} cols={list(value.columns)}>"
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_isolated_finagent_strategy() -> type:
    """Load FinAgentStrategy with per-job workdir controlled by env var.

    The reproduce branch keeps the original FinAgent implementation intact. For Slurm
    arrays, we patch the hard-coded workdir at import time so concurrent jobs do not
    share memory, plots, or prompt artifacts.
    """

    source_path = REPO_ROOT / "backtest" / "strategy" / "timing_llm" / "finagent.py"
    source = source_path.read_text(encoding="utf-8")
    source = source.replace(
        'self.workdir = "llm_traders/finagent/workdir/trading"',
        'self.workdir = os.environ.get("FINSABER_FINAGENT_WORKDIR", "llm_traders/finagent/workdir/trading")',
    )
    module_name = f"_finsaber_slurm_finagent_{os.getpid()}"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module.FinAgentStrategy


def build_data_loader(setup: str) -> FinMemDataset:
    if setup in {"selected_4", "selected_5", "cherry_pick_both_finmem", "cherry_pick_both_fincon"}:
        return FinMemDataset(pickle_file="data/finmem_data/stock_data_cherrypick_2000_2024.pkl")
    return FinMemDataset(pickle_file="data/finmem_data/stock_data_sp500_2000_2024.pkl")


def build_selector(setup: str):
    if setup.startswith("random_sp500_"):
        return RandomSP500Selector(num_tickers=int(setup.split("_")[-1]), random_seed_setting="year")
    if setup.startswith("momentum_sp500_"):
        return MomentumSP500Selector(num_tickers=int(setup.split("_")[-1]), momentum_period=100, skip_period=21, training_period=2)
    if setup.startswith("lowvol_sp500_"):
        return LowVolatilitySP500Selector(num_tickers=int(setup.split("_")[-1]), lookback_period=100, training_period=2)
    if setup.startswith("fincon_selector_sp500_"):
        return FinConSP500Selector(num_tickers=int(setup.split("_")[-1]), lookback_years=2, training_period=2)
    return None


def resolve_tickers(args: argparse.Namespace, data_loader: FinMemDataset) -> list[str]:
    if args.tickers and args.tickers.upper() != "AUTO":
        return [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]
    if args.setup == "selected_4":
        return ["TSLA", "NFLX", "AMZN", "MSFT"]
    if args.setup == "selected_5":
        return ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]
    selector = build_selector(args.setup)
    if selector is None:
        raise ValueError(f"Cannot infer tickers for setup {args.setup}; pass --tickers.")
    selected = selector.select(data_loader, args.date_from, args.date_to)
    if not selected:
        raise RuntimeError(f"Selector returned no tickers for {args.setup} {args.date_from} to {args.date_to}.")
    return selected


def load_strategy_params(path: str, model_id: str, date_from: str, date_to: str) -> dict[str, Any]:
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    params["date_from"] = "$date_from"
    params["date_to"] = "$date_to"
    params["symbol"] = "$symbol"
    params["llm_model_id"] = model_id
    if "market_data_info_path" not in params:
        raise ValueError("Strategy config must contain market_data_info_path.")
    return params


def save_outputs(job_dir: Path, args: argparse.Namespace, tickers: list[str], metrics: dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    wrapped = {f"{args.date_from}_{args.date_to}": metrics}
    with (job_dir / "metrics.pkl").open("wb") as f:
        pickle.dump(wrapped, f)

    equity_dir = job_dir / "equity"
    equity_dir.mkdir(exist_ok=True)
    summary_metrics: dict[str, Any] = {}
    for ticker, ticker_metrics in metrics.items():
        summary_metrics[ticker] = {}
        for key, value in ticker_metrics.items():
            if key == "equity_with_time" and isinstance(value, pd.DataFrame):
                value.to_pickle(equity_dir / f"{ticker}.pkl")
            else:
                summary_metrics[ticker][key] = jsonable(value)

    summary = {
        "success": True,
        "setup": args.setup,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "tickers": tickers,
        "model_id": args.model_id,
        "strategy_config_path": args.strat_config_path,
        "llm_cost_estimate_usd": float(get_llm_cost()),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "metrics": summary_metrics,
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    window_name = f"{safe_part(args.date_from)}_{safe_part(args.date_to)}"
    job_dir = Path(args.output_root) / safe_part(args.setup) / window_name / f"job_{safe_part(job_id)}_{safe_part(task_id)}"
    summary_path = job_dir / "summary.json"

    if summary_path.exists() and not args.force:
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("success") is True:
                print(f"Skipping completed task: {summary_path}")
                return 0
        except json.JSONDecodeError:
            pass

    job_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FINSABER_FINAGENT_WORKDIR"] = str((job_dir / "finagent_workdir").resolve())

    if not args.skip_dataset_check:
        ensure_datasets()

    data_loader = build_data_loader(args.setup)
    tickers = resolve_tickers(args, data_loader)
    strat_params = load_strategy_params(args.strat_config_path, args.model_id, args.date_from, args.date_to)
    strategy_class = load_isolated_finagent_strategy()

    trade_config = {
        "data_loader": data_loader,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "tickers": tickers,
        "setup_name": args.setup,
        "silence": True,
        "save_results": False,
        "log_base_dir": str(job_dir / "framework_output"),
    }

    try:
        print(f"Running {args.setup} {args.date_from} to {args.date_to} tickers={tickers} model={args.model_id}")
        engine = FINSABER(trade_config)
        metrics = engine.run_iterative_tickers(strategy_class, strat_params=strat_params, tickers=tickers, delist_check=True)
        save_outputs(job_dir, args, tickers, metrics)
        print(f"Saved result to {job_dir}")
        return 0
    except Exception as exc:
        failure = {
            "success": False,
            "setup": args.setup,
            "date_from": args.date_from,
            "date_to": args.date_to,
            "tickers": tickers if "tickers" in locals() else None,
            "model_id": args.model_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        (job_dir / "FAILED.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
