#!/usr/bin/env python
"""Generate TSV manifests for FinAgent Slurm array jobs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

NORMAL_CONFIG = "strats_configs/finagent_config_normal.json"
COMPOSITE_CONFIG = "strats_configs/finagent_config_composite.json"
SELECTED4_TICKERS = "TSLA,NFLX,AMZN,MSFT"


def yearly_windows(start_year: int, end_year: int, window_years: int) -> list[tuple[str, str]]:
    """Return [start, end) yearly windows ending no later than end_year-01-01."""

    return [
        (f"{year}-01-01", f"{year + window_years}-01-01")
        for year in range(start_year, end_year - window_years + 1)
    ]


def build_rows(suite: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    if suite in {"paper_rest", "selected4_long"}:
        # Table 3 style long-run selected tickers: 2-year test windows.
        for date_from, date_to in yearly_windows(2004, 2024, 2):
            rows.append(
                {
                    "setup": "selected_4",
                    "strat_config_path": NORMAL_CONFIG,
                    "date_from": date_from,
                    "date_to": date_to,
                    "tickers": SELECTED4_TICKERS,
                }
            )

    if suite in {"paper_rest", "composite"}:
        # Table 4 style composite selectors: 1-year test windows, selector chooses 5 tickers.
        for setup in [
            "random_sp500_5",
            "momentum_sp500_5",
            "lowvol_sp500_5",
            "fincon_selector_sp500_5",
        ]:
            for date_from, date_to in yearly_windows(2004, 2024, 1):
                rows.append(
                    {
                        "setup": setup,
                        "strat_config_path": COMPOSITE_CONFIG,
                        "date_from": date_from,
                        "date_to": date_to,
                        "tickers": "AUTO",
                    }
                )

    if not rows:
        raise ValueError(f"Unknown suite: {suite}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FinAgent Slurm task manifest.")
    parser.add_argument("--suite", choices=["paper_rest", "selected4_long", "composite"], default="paper_rest")
    parser.add_argument("--output", required=True, help="Output TSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args.suite)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["setup", "strat_config_path", "date_from", "date_to", "tickers"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} tasks to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
