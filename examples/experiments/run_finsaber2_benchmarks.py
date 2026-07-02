from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest.data_util import create_finsaber2_data_loader  # noqa: E402
from backtest.finsaber_bt import FINSABERBt  # noqa: E402
from backtest.strategy.selection.base_selector import BaseSelector  # noqa: E402
from backtest.strategy.timing import (  # noqa: E402
    ARIMAPredictorStrategy,
    ATRBandStrategy,
    BollingerBandsStrategy,
    BuyAndHoldStrategy,
    SMACrossStrategy,
    TrendFollowingStrategy,
    TurnOfTheMonthStrategy,
    WMAStrategy,
    XGBoostPredictorStrategy,
)
from backtest.toolkit.operation_utils import (  # noqa: E402
    aggregate_results_one_strategy,
)


DEFAULT_STRATEGIES = (
    "BuyAndHoldStrategy",
    "ATRBandStrategy",
    "BollingerBandsStrategy",
    "SMACrossStrategy",
    "TrendFollowingStrategy",
    "TurnOfTheMonthStrategy",
    "WMAStrategy",
    "ARIMAPredictorStrategy",
    "XGBoostPredictorStrategy",
    "FinRLStrategy",
)

STRATEGIES = {
    strategy.__name__: strategy
    for strategy in (
        BuyAndHoldStrategy,
        ATRBandStrategy,
        BollingerBandsStrategy,
        SMACrossStrategy,
        TrendFollowingStrategy,
        TurnOfTheMonthStrategy,
        WMAStrategy,
        ARIMAPredictorStrategy,
        XGBoostPredictorStrategy,
    )
}


class FixedTickerSelector(BaseSelector):
    def __init__(self, tickers: list[str]):
        self.tickers = list(tickers)

    def select(self, *args, **kwargs) -> list[str]:
        return list(self.tickers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible FINSABER-2 non-LLM benchmark experiments."
    )
    parser.add_argument("--setup", default="magnificent_7")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "examples/experiments/manifests/finagent_finsaber2_2024_2026.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("tmp/magnificent7-benchmarks-2024-2026-r1"),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated strategy class names.",
    )
    parser.add_argument("--training-years", type=int, default=3)
    parser.add_argument("--finrl-training-years", type=int, default=10)
    parser.add_argument("--finrl-total-timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_manifest(args: argparse.Namespace) -> tuple[dict, list[str]]:
    manifest_path = resolve_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selections = manifest["selections"].get(args.setup)
    if not selections:
        raise ValueError(f"Setup {args.setup!r} is not present in {manifest_path}")

    ticker_sets = {tuple(tickers) for tickers in selections.values()}
    if len(ticker_sets) != 1:
        raise ValueError(
            "This runner requires one fixed ticker universe across all windows."
        )
    return manifest, list(ticker_sets.pop())


def expected_result_count(manifest: dict, setup: str, ticker_count: int) -> int:
    return len(manifest["selections"][setup]) * ticker_count


def completed_result_count(
    output_root: Path,
    setup: str,
    strategy_name: str,
) -> int:
    strategy_root = output_root / setup / strategy_name
    return sum(1 for _ in strategy_root.glob("*/*/metrics.json"))


def result_inventory(
    output_root: Path,
    setup: str,
    expected: int,
) -> list[dict]:
    inventory = []
    for strategy_name in DEFAULT_STRATEGIES:
        count = completed_result_count(output_root, setup, strategy_name)
        if count == expected:
            status = "complete"
        elif count:
            status = "partial"
        else:
            status = "missing"
        inventory.append(
            {
                "strategy": strategy_name,
                "status": status,
                "metric_rows": count,
            }
        )
    return inventory


def load_strategy(name: str):
    if name == "FinRLStrategy":
        from rl_traders.finrl_strategy import FinRLStrategy

        return FinRLStrategy
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        choices = ", ".join(DEFAULT_STRATEGIES)
        raise ValueError(f"Unknown strategy {name!r}; choose from {choices}") from exc


def write_runner_manifest(
    output_root: Path,
    args: argparse.Namespace,
    source_manifest: Path,
    tickers: list[str],
    statuses: list[dict],
) -> None:
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_manifest": str(source_manifest.resolve()),
        "setup": args.setup,
        "tickers": tickers,
        "data_root": str(args.data_root),
        "output_root": str(output_root),
        "date_from": "2024-01-01",
        "date_to": "2026-01-01",
        "training_years": args.training_years,
        "finrl_training_years": args.finrl_training_years,
        "finrl_total_timesteps": args.finrl_total_timesteps,
        "seed": args.seed,
        "execution_timing": "next_open",
        "risk_free_rate": 0.03,
        "commission_per_share": 0.0049,
        "slippage_perc": 0.0,
        "liquidity_cap_pct": 0.0,
        "strategies": statuses,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "benchmark_runner_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    source_manifest = resolve_path(args.manifest)
    manifest, tickers = load_manifest(args)
    if args.data_root is None:
        args.data_root = Path(manifest["data_root_default"])
    args.data_root = args.data_root.resolve()
    output_root = resolve_path(args.output_root).resolve()
    strategy_names = [name.strip() for name in args.strategies.split(",") if name.strip()]
    expected = expected_result_count(manifest, args.setup, len(tickers))
    statuses: list[dict] = []

    for strategy_name in strategy_names:
        existing = completed_result_count(output_root, args.setup, strategy_name)
        if existing == expected and not args.force:
            statuses.append(
                {
                    "strategy": strategy_name,
                    "status": "skipped_complete",
                    "metric_rows": existing,
                }
            )
            continue
        if existing and not args.force:
            raise RuntimeError(
                f"{strategy_name} has {existing}/{expected} results. "
                "Remove the incomplete strategy directory or rerun with --force."
            )

        strategy = load_strategy(strategy_name)
        training_years = (
            args.finrl_training_years
            if strategy_name == "FinRLStrategy"
            else args.training_years
        )
        config = {
            "tickers": tickers,
            "date_from": "2024-01-01",
            "date_to": "2026-01-01",
            "cash": 100000.0,
            "risk_free_rate": 0.03,
            "commission_per_share": 0.0049,
            "min_commission": 0.99,
            "max_commission_rate": 0.01,
            "execution_timing": "next_open",
            "slippage_perc": 0.0,
            "slippage_impact": 0.0,
            "liquidity_lookback_days": 20,
            "liquidity_min_history_days": 1,
            "liquidity_cap_pct": 0.0,
            "llm_cost_as_trade_cost": True,
            "print_trades_table": False,
            "silence": True,
            "rolling_window_size": 1,
            "rolling_window_step": 1,
            "training_years": training_years,
            "selection_strategy": FixedTickerSelector(tickers),
            "setup_name": args.setup,
            "save_results": True,
            "log_base_dir": str(output_root),
            "data_loader": create_finsaber2_data_loader(
                args.data_root,
                tickers=tickers,
            ),
        }
        strategy_kwargs = {}
        if strategy_name == "FinRLStrategy":
            strategy_kwargs["total_timesteps"] = args.finrl_total_timesteps
            strategy_kwargs["seed"] = args.seed

        print(f"Running {strategy_name}: {len(tickers)} tickers, 2 windows")
        FINSABERBt(config).run_rolling_window(strategy, **strategy_kwargs)
        aggregate_results_one_strategy(
            args.setup,
            strategy_name,
            output_dir=str(output_root),
        )
        actual = completed_result_count(output_root, args.setup, strategy_name)
        if actual != expected:
            raise RuntimeError(
                f"{strategy_name} produced {actual}/{expected} metric rows."
            )
        statuses.append(
            {
                "strategy": strategy_name,
                "status": "completed",
                "metric_rows": actual,
            }
        )
        write_runner_manifest(
            output_root,
            args,
            source_manifest,
            tickers,
            result_inventory(output_root, args.setup, expected),
        )

    write_runner_manifest(
        output_root,
        args,
        source_manifest,
        tickers,
        result_inventory(output_root, args.setup, expected),
    )
    print(f"Completed {len(statuses)} strategies in {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
