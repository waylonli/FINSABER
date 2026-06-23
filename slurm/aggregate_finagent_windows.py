#!/usr/bin/env python
"""Aggregate completed one-window FinAgent Slurm outputs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate FINSABER FinAgent Slurm window outputs.")
    parser.add_argument("--output-root", default="runs/finagent_gpt4o_mini")
    parser.add_argument("--output", default=None, help="Combined pickle path. Defaults to output-root/aggregate/combined.pkl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root)
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else aggregate_dir / "combined.pkl"

    combined: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for summary_path in sorted(root.glob("*/20*/job_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        setup = summary["setup"]
        window = f"{summary['date_from']}_{summary['date_to']}"
        metrics_path = summary_path.parent / "metrics.pkl"
        if not metrics_path.exists():
            failed.append({"summary": str(summary_path), "error": "metrics.pkl missing"})
            continue
        with metrics_path.open("rb") as f:
            metrics = pickle.load(f)
        combined.setdefault(setup, {})[window] = metrics[window]
        for ticker, ticker_metrics in summary["metrics"].items():
            rows.append(
                {
                    "setup": setup,
                    "date_from": summary["date_from"],
                    "date_to": summary["date_to"],
                    "ticker": ticker,
                    "model_id": summary["model_id"],
                    "llm_cost_estimate_usd_window": summary.get("llm_cost_estimate_usd"),
                    "final_value": ticker_metrics.get("final_value"),
                    "total_return": ticker_metrics.get("total_return"),
                    "annual_return": ticker_metrics.get("annual_return"),
                    "annual_volatility": ticker_metrics.get("annual_volatility"),
                    "sharpe_ratio": ticker_metrics.get("sharpe_ratio"),
                    "sortino_ratio": ticker_metrics.get("sortino_ratio"),
                    "max_drawdown": ticker_metrics.get("max_drawdown"),
                    "total_commission": ticker_metrics.get("total_commission"),
                    "source_dir": str(summary_path.parent),
                }
            )

    for failure_path in sorted(root.glob("*/20*/job_*/FAILED.json")):
        failed.append(json.loads(failure_path.read_text(encoding="utf-8")))

    with output_path.open("wb") as f:
        pickle.dump(combined, f)

    summary_df = pd.DataFrame(rows)
    summary_csv = aggregate_dir / "summary.csv"
    summary_json = aggregate_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "completed_windows": int(summary_df[["setup", "date_from", "date_to"]].drop_duplicates().shape[0]) if not summary_df.empty else 0,
                "completed_ticker_runs": int(summary_df.shape[0]),
                "failed_count": len(failed),
                "failed": jsonable(failed),
                "combined_pickle": str(output_path),
                "summary_csv": str(summary_csv),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_path}")
    print(f"Wrote {summary_csv}")
    print(f"Completed ticker runs: {summary_df.shape[0]}")
    print(f"Failures: {len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
