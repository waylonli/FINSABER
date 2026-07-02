from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest.toolkit.metrics import (  # noqa: E402
    calculate_annual_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


STRATEGY_FAMILIES = {
    "BuyAndHoldStrategy": "Baseline",
    "ATRBandStrategy": "Traditional",
    "BollingerBandsStrategy": "Traditional",
    "SMACrossStrategy": "Traditional",
    "TrendFollowingStrategy": "Traditional",
    "TurnOfTheMonthStrategy": "Traditional",
    "WMAStrategy": "Traditional",
    "ARIMAPredictorStrategy": "Statistical predictor",
    "XGBoostPredictorStrategy": "ML predictor",
    "FinRLStrategy": "RL",
    "FinAgentStrategy": "LLM",
}

EXPECTED_ROWS = {
    "selected_4": 110,
    "random_sp500_5": 110,
    "momentum_sp500_5": 110,
    "lowvol_sp500_5": 110,
    "magnificent_7": 154,
}


@dataclass(frozen=True)
class ResultTree:
    name: str
    root: Path
    setups: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate FINSABER-2 2024-2026 experiment results."
    )
    parser.add_argument("--tmp-root", type=Path, default=Path("tmp"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("tmp/consolidated-finsaber2-2024-2026-r1"),
    )
    parser.add_argument("--min-annual-volatility", type=float, default=0.005)
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    return parser.parse_args()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_trees(tmp_root: Path) -> list[ResultTree]:
    return [
        ResultTree(
            "selected_traditional",
            tmp_root / "traditional-selected4-2024-2026-r1",
            ("selected_4",),
        ),
        ResultTree(
            "selected_buyhold",
            tmp_root / "buyhold-selected4-2024-2026-r1",
            ("selected_4",),
        ),
        ResultTree(
            "composite_non_llm",
            tmp_root / "composite-2024-2026-official-r3",
            ("random_sp500_5", "momentum_sp500_5", "lowvol_sp500_5"),
        ),
        ResultTree(
            "finagent_four_selectors",
            tmp_root / "finagent-all-2024-2026-official-r1",
            ("selected_4", "random_sp500_5", "momentum_sp500_5", "lowvol_sp500_5"),
        ),
        ResultTree(
            "magnificent_7_non_llm",
            tmp_root / "magnificent7-benchmarks-2024-2026-r1",
            ("magnificent_7",),
        ),
        ResultTree(
            "finagent_magnificent_7",
            tmp_root / "finagent-magnificent7-2024-2026-r1",
            ("magnificent_7",),
        ),
    ]


def number(metrics: dict, key: str, default=np.nan) -> float:
    value = metrics.get(key, default)
    return float(default if value is None else value)


def recompute_risk_metrics(
    metrics_path: Path,
    risk_free_rate: float,
) -> tuple[float, float, float, int]:
    equity_path = metrics_path.with_name("equity_curve.csv")
    if not equity_path.is_file():
        raise FileNotFoundError(f"Missing equity curve: {equity_path}")

    equity_curve = pd.read_csv(equity_path)
    if "equity" not in equity_curve:
        raise ValueError(f"Missing equity column: {equity_path}")
    equity = pd.to_numeric(equity_curve["equity"], errors="coerce")
    daily_returns = equity.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if daily_returns.empty:
        raise ValueError(f"No valid returns in equity curve: {equity_path}")

    return (
        float(calculate_annual_volatility(daily_returns)),
        float(calculate_sharpe_ratio(daily_returns, risk_free_rate)),
        float(calculate_sortino_ratio(daily_returns, risk_free_rate)),
        int(len(daily_returns)),
    )


def load_rows(
    trees: list[ResultTree],
    min_annual_volatility: float,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    inventory = []
    seen = set()

    for tree in trees:
        if not tree.root.is_dir():
            raise FileNotFoundError(f"Missing result source: {tree.root}")
        source_count = 0
        for setup in tree.setups:
            setup_root = tree.root / setup
            for strategy_dir in sorted(setup_root.iterdir()):
                if not strategy_dir.is_dir() or strategy_dir.name not in STRATEGY_FAMILIES:
                    continue
                strategy = strategy_dir.name
                for path in sorted(strategy_dir.glob("*/*/metrics.json")):
                    window = path.parent.parent.name
                    ticker = path.parent.name
                    identity = (setup, strategy, window, ticker)
                    if identity in seen:
                        raise ValueError(f"Duplicate result identity: {identity}")
                    seen.add(identity)

                    metrics = json.loads(path.read_text(encoding="utf-8"))
                    (
                        annual_volatility,
                        recomputed_sharpe,
                        recomputed_sortino,
                        return_observations,
                    ) = recompute_risk_metrics(path, risk_free_rate)
                    sharpe_defined = (
                        np.isfinite(annual_volatility)
                        and annual_volatility >= min_annual_volatility
                    )
                    rows.append(
                        {
                            "selector": setup,
                            "strategy": strategy,
                            "family": STRATEGY_FAMILIES[strategy],
                            "window": window,
                            "ticker": ticker,
                            "final_value": number(metrics, "final_value"),
                            "total_return": number(metrics, "total_return"),
                            "annual_return": number(metrics, "annual_return"),
                            "annual_volatility": annual_volatility,
                            "stored_annual_volatility": number(
                                metrics, "annual_volatility"
                            ),
                            "stored_sharpe_ratio": number(metrics, "sharpe_ratio"),
                            "recomputed_sharpe_ratio": recomputed_sharpe,
                            "reported_sharpe_ratio": (
                                recomputed_sharpe if sharpe_defined else np.nan
                            ),
                            "sharpe_status": "defined" if sharpe_defined else "near_zero_volatility",
                            "stored_sortino_ratio": number(metrics, "sortino_ratio"),
                            "recomputed_sortino_ratio": recomputed_sortino,
                            "return_observations": return_observations,
                            "max_drawdown": number(metrics, "max_drawdown"),
                            "total_commission": number(metrics, "total_commission", 0.0),
                            "total_slippage": number(metrics, "total_slippage", 0.0),
                            "total_llm_cost": number(metrics, "total_llm_cost", 0.0),
                            "total_trading_cost": number(metrics, "total_trading_cost", 0.0),
                            "source": tree.name,
                            "metrics_path": str(path.resolve()),
                        }
                    )
                    source_count += 1
        inventory.append(
            {
                "name": tree.name,
                "root": str(tree.root.resolve()),
                "metric_rows": source_count,
            }
        )

    return pd.DataFrame(rows), inventory


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (selector, strategy, family), group in rows.groupby(
        ["selector", "strategy", "family"],
        sort=True,
    ):
        yearly = group.groupby("window")["total_return"].mean().sort_index()
        summaries.append(
            {
                "selector": selector,
                "strategy": strategy,
                "family": family,
                "ticker_years": len(group),
                "return_2024": yearly.get("2024-01-01_2025-01-01", np.nan),
                "return_2025": yearly.get("2025-01-01_2026-01-01", np.nan),
                "equal_weight_compounded_return": (1.0 + yearly).prod() - 1.0,
                "mean_ticker_year_return": group["total_return"].mean(),
                "median_ticker_year_return": group["total_return"].median(),
                "mean_reported_sharpe": group["reported_sharpe_ratio"].mean(),
                "defined_sharpe_runs": group["reported_sharpe_ratio"].notna().sum(),
                "undefined_sharpe_runs": group["reported_sharpe_ratio"].isna().sum(),
                "mean_max_drawdown": group["max_drawdown"].mean(),
                "total_commission": group["total_commission"].sum(),
                "total_slippage": group["total_slippage"].sum(),
                "total_llm_cost": group["total_llm_cost"].sum(),
                "total_trading_cost": group["total_trading_cost"].sum(),
            }
        )

    result = pd.DataFrame(summaries)
    result["rank"] = (
        result.groupby("selector")["equal_weight_compounded_return"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return result.sort_values(["selector", "rank", "strategy"]).reset_index(drop=True)


def format_percent(value) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def format_ratio(value) -> str:
    return "N/A" if pd.isna(value) else f"{value:.3f}"


def markdown_report(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    inventory: list[dict],
    min_annual_volatility: float,
    risk_free_rate: float,
) -> str:
    lines = [
        "# FINSABER-2 Consolidated Results, 2024-2026",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "All returns use yearly equal-weight ticker averages. The two-year result",
        "compounds the 2024 and 2025 yearly averages. Trading cost includes",
        "commission, slippage, and LLM cost where applicable.",
        "",
        "## Completeness",
        "",
        "| Selector | Ticker-year rows | Strategies |",
        "|---|---:|---:|",
    ]
    for selector, expected in EXPECTED_ROWS.items():
        selector_rows = rows[rows["selector"] == selector]
        lines.append(
            f"| `{selector}` | {len(selector_rows)}/{expected} | "
            f"{selector_rows['strategy'].nunique()} |"
        )

    lines.extend(
        [
            "",
            "## Sharpe Policy",
            "",
            "Sharpe, Sortino, and annualized volatility are recomputed from every",
            "saved equity curve with the current framework metric helpers and an",
            f"annual risk-free rate of {risk_free_rate:.2%}. Stored experiment-time",
            "metrics remain in the detailed CSV for auditability.",
            "",
            f"Runs with annualized volatility below {min_annual_volatility:.2%} "
            "are treated as inactive/near-cash runs. Their recomputed Sharpe is",
            "preserved in `all_ticker_year_results.csv`, but reported Sharpe is",
            "undefined and excluded from strategy averages.",
            "",
        ]
    )

    selector_order = [
        "magnificent_7",
        "selected_4",
        "random_sp500_5",
        "momentum_sp500_5",
        "lowvol_sp500_5",
    ]
    for selector in selector_order:
        table = summary[summary["selector"] == selector]
        lines.extend(
            [
                f"## {selector}",
                "",
                "| Rank | Strategy | Family | 2024 | 2025 | Two-year | "
                "Sharpe | Mean MDD | Total cost |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in table.itertuples(index=False):
            lines.append(
                f"| {row.rank} | {row.strategy} | {row.family} | "
                f"{format_percent(row.return_2024)} | "
                f"{format_percent(row.return_2025)} | "
                f"{format_percent(row.equal_weight_compounded_return)} | "
                f"{format_ratio(row.mean_reported_sharpe)} | "
                f"{row.mean_max_drawdown:.2f}% | "
                f"${row.total_trading_cost:,.2f} |"
            )
        lines.append("")

    lines.extend(["## Source Inventory", ""])
    for item in inventory:
        lines.append(
            f"- `{item['name']}`: {item['metric_rows']} rows from `{item['root']}`"
        )
    lines.extend(
        [
            "",
            "## Known Limitation",
            "",
            "FinRL results remain preliminary. Several runs had near-zero capital",
            "exposure because the exported action was interpreted as a one-share",
            "order. Action scaling and longer training windows require a controlled",
            "rerun before final publication.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_strategy_returns(summary: pd.DataFrame, output_path: Path) -> None:
    selectors = [
        "magnificent_7",
        "random_sp500_5",
        "momentum_sp500_5",
        "lowvol_sp500_5",
    ]
    colors = {
        "Baseline": "#64748b",
        "Traditional": "#0f766e",
        "Statistical predictor": "#0369a1",
        "ML predictor": "#7c3aed",
        "RL": "#c2410c",
        "LLM": "#b91c1c",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=160)
    for axis, selector in zip(axes.flat, selectors):
        data = summary[summary["selector"] == selector].sort_values(
            "equal_weight_compounded_return"
        )
        axis.barh(
            data["strategy"].str.replace("Strategy", "", regex=False),
            data["equal_weight_compounded_return"] * 100,
            color=[colors[family] for family in data["family"]],
        )
        axis.axvline(0, color="#111827", linewidth=0.8)
        axis.set_title(selector)
        axis.set_xlabel("Two-year compounded return (%)")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("FINSABER-2 Strategy Comparison, 2024-2026", fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_finagent_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    data = summary[summary["strategy"] == "FinAgentStrategy"].copy()
    data = data.sort_values("equal_weight_compounded_return", ascending=False)
    labels = data["selector"].str.replace("_", " ").tolist()
    returns = data["equal_weight_compounded_return"] * 100
    sharpes = data["mean_reported_sharpe"]

    fig, (return_axis, sharpe_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        dpi=160,
    )
    return_axis.bar(labels, returns, color="#0f766e")
    return_axis.axhline(0, color="#111827", linewidth=0.8)
    return_axis.set_ylabel("Two-year compounded return (%)")
    return_axis.set_title("FinAgent Return")
    return_axis.tick_params(axis="x", rotation=25)
    return_axis.grid(axis="y", alpha=0.2)

    sharpe_axis.bar(labels, sharpes, color="#0369a1")
    sharpe_axis.axhline(0, color="#111827", linewidth=0.8)
    sharpe_axis.set_ylabel("Mean reported Sharpe")
    sharpe_axis.set_title("FinAgent Risk-Adjusted Performance")
    sharpe_axis.tick_params(axis="x", rotation=25)
    sharpe_axis.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    tmp_root = args.tmp_root
    output_root = args.output_root
    if not tmp_root.is_absolute():
        tmp_root = repo_root / tmp_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows, inventory = load_rows(
        source_trees(tmp_root.resolve()),
        args.min_annual_volatility,
        args.risk_free_rate,
    )
    summary = summarize(rows)

    actual_counts = rows.groupby("selector").size().to_dict()
    if actual_counts != EXPECTED_ROWS:
        raise ValueError(
            f"Incomplete result inventory: expected {EXPECTED_ROWS}, got {actual_counts}"
        )
    if len(rows) != 594:
        raise ValueError(f"Expected 594 ticker-year rows, got {len(rows)}")

    rows.to_csv(output_root / "all_ticker_year_results.csv", index=False)
    summary.to_csv(output_root / "strategy_summary.csv", index=False)
    summary.sort_values(
        ["selector", "rank"]
    ).to_csv(output_root / "selector_rankings.csv", index=False)
    plot_strategy_returns(
        summary,
        output_root / "strategy_returns_by_selector.png",
    )
    plot_finagent_comparison(
        summary,
        output_root / "finagent_selector_comparison.png",
    )
    (output_root / "consolidated_report.md").write_text(
        markdown_report(
            summary,
            rows,
            inventory,
            args.min_annual_volatility,
            args.risk_free_rate,
        ),
        encoding="utf-8",
    )
    (output_root / "experiment_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "git_commit": git_commit(repo_root),
                "min_annual_volatility": args.min_annual_volatility,
                "risk_free_rate": args.risk_free_rate,
                "total_ticker_year_rows": len(rows),
                "sources": inventory,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"rows={len(rows)} summaries={len(summary)} output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
