import copy
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import sys

from backtest.data_util import create_finsaber2_data_loader
from backtest.finsaber import FINSABER
from backtest.finsaber_bt import FINSABERBt
from backtest.strategy.selection import (
    LowVolatilitySP500Selector,
    MomentumSP500Selector,
    RandomSP500Selector,
)
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.strategy.timing_llm.base_strategy_iso import BaseStrategyIso
from backtest.toolkit.operation_utils import aggregate_results_one_strategy


class ExperimentRunner:
    def __init__(
        self,
        output_dir: str = os.path.join("backtest", "output"),
        data_root: str | None = None,
    ):
        self.output_dir = output_dir
        self.data_root = data_root
        self.mode = None

    def _data_loader(self, tickers=None):
        return create_finsaber2_data_loader(self.data_root, tickers=tickers)

    def _build_launcher_command(
        self,
        *,
        setup_name: str,
        strategy_class,
        trade_config: dict,
        strat_config_path: str | None,
    ) -> str:
        launcher_path = Path(__file__).resolve().parent / "run_llm_traders_exp.py"
        command_parts = [
            sys.executable,
            str(launcher_path),
            "--setup",
            str(setup_name),
            "--strategy",
            str(strategy_class.__name__),
        ]
        if strat_config_path:
            command_parts.extend(["--strat_config_path", str(strat_config_path)])
        if self.output_dir:
            command_parts.extend(["--output_dir", str(self.output_dir)])
        if trade_config.get("date_from"):
            command_parts.extend(["--date_from", str(trade_config["date_from"])])
        if trade_config.get("date_to"):
            command_parts.extend(["--date_to", str(trade_config["date_to"])])
        if trade_config.get("rolling_window_size") is not None:
            command_parts.extend(
                ["--rolling_window_size", str(trade_config["rolling_window_size"])]
            )
        if trade_config.get("rolling_window_step") is not None:
            command_parts.extend(
                ["--rolling_window_step", str(trade_config["rolling_window_step"])]
            )
        if self.data_root:
            command_parts.extend(["--data_root", str(self.data_root)])
        return " ".join(shlex.quote(part) for part in command_parts)

    def _write_tradingagents_launcher_strat_config_snapshot(
        self,
        *,
        launcher_dir: Path,
        materialized_strat_config: dict,
    ) -> Path:
        snapshot_path = launcher_dir / "strat_config.materialized.json"
        snapshot_path.write_text(
            json.dumps(materialized_strat_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return snapshot_path.resolve()

    def _write_tradingagents_launcher_artifacts(
        self,
        *,
        run_identity,
        setup_name: str,
        strategy_class,
        trade_config: dict,
        strat_config_path: str | None,
        materialized_strat_config: dict | None,
        status: str,
        error: str | None = None,
    ) -> None:
        launcher_dir = run_identity.base_run_dir / "launcher"
        launcher_dir.mkdir(parents=True, exist_ok=True)

        replay_strat_config_path = strat_config_path
        if materialized_strat_config is not None:
            # Replay must target the run-local frozen config snapshot so the
            # launcher cannot drift to a fresh run_key or a different root.
            replay_strat_config_path = str(
                self._write_tradingagents_launcher_strat_config_snapshot(
                    launcher_dir=launcher_dir,
                    materialized_strat_config=materialized_strat_config,
                )
            )

        command_text = self._build_launcher_command(
            setup_name=setup_name,
            strategy_class=strategy_class,
            trade_config=trade_config,
            strat_config_path=replay_strat_config_path,
        )
        repo_root = Path(__file__).resolve().parents[2]
        run_script_path = launcher_dir / "run.sh"
        run_script_path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            "# Replays this run with the frozen run-local strat config snapshot.\n"
            f"cd {shlex.quote(str(repo_root))}\n"
            f"{command_text}\n",
            encoding="utf-8",
        )
        os.chmod(run_script_path, 0o755)

        log_payload = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": status,
            "setup_name": setup_name,
            "strategy_name": strategy_class.__name__,
            "config_key": run_identity.config_key,
            "run_key": run_identity.run_key,
            "base_run_dir": str(run_identity.base_run_dir),
            "benchmark_results_dir": str(run_identity.benchmark_results_dir),
            "working_directory": str(repo_root),
            "command": command_text,
        }
        if strat_config_path is not None:
            log_payload["input_strat_config_path"] = str(strat_config_path)
        if replay_strat_config_path is not None:
            log_payload["replay_strat_config_path"] = str(replay_strat_config_path)
        if error is not None:
            log_payload["error"] = error

        with (launcher_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_payload, sort_keys=True) + "\n")

    def _materialize_tradingagents_run_root(
        self,
        *,
        strategy_class,
        trade_config: dict,
        strat_config: dict,
    ) -> tuple[dict, dict, object | None]:
        if strategy_class.__name__ != "TradingAgentsStrategy":
            return trade_config, strat_config, None

        from llm_traders.finsaber_strategies.tradingagents import (
            materialize_tradingagents_run_identity,
        )

        materialized_strat_config = copy.deepcopy(strat_config)
        artifact_config = materialized_strat_config.get("artifact_config")
        if not isinstance(artifact_config, dict):
            raise ValueError(
                "TradingAgentsStrategy strat_config must provide artifact_config."
            )

        run_identity = materialize_tradingagents_run_identity(
            artifact_config=artifact_config
        )
        benchmark_results_dir = str(run_identity.benchmark_results_dir)
        explicit_result_output_dir = trade_config.get("result_output_dir")

        if explicit_result_output_dir:
            normalized_explicit_dir = os.path.abspath(
                os.path.expanduser(str(explicit_result_output_dir))
            )
            if normalized_explicit_dir != benchmark_results_dir:
                raise ValueError(
                    "TradingAgentsStrategy result_output_dir must match the "
                    "materialized benchmark_results directory."
                )

        materialized_strat_config["artifact_config"] = {
            **artifact_config,
            "root": str(run_identity.artifact_root),
            "run_key": run_identity.run_key,
        }

        materialized_trade_config = dict(trade_config)
        # TradingAgents keeps its private runtime artifacts under the run root
        # and the benchmark writer uses the sibling benchmark_results directory
        # so both evidence trees stay anchored to the same run_key.
        materialized_trade_config["result_output_dir"] = benchmark_results_dir
        return materialized_trade_config, materialized_strat_config, run_identity

    def run(
        self,
        setup_name: str,
        strategy_class: BaseStrategy | BaseStrategyIso,
        custom_trade_config: dict = None,
        strat_config_path: str = None,
    ):
        if setup_name == "cherry_pick_both_finmem":
            tickers = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]
            default_config = {
                "data_loader": self._data_loader(tickers=tickers),
                "date_from": "2022-10-06",
                "date_to": "2023-04-10",
                "tickers": tickers,
                "silence": False,
                "setup_name": setup_name,
            }
            self.mode = "iter"
        elif setup_name == "cherry_pick_both_fincon":
            tickers = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN", "NIO", "GOOG", "AAPL"]
            default_config = {
                "data_loader": self._data_loader(tickers=tickers),
                "date_from": "2022-10-05",
                "date_to": "2023-06-10",
                "tickers": tickers,
                "silence": True,
                "setup_name": setup_name,
            }
            self.mode = "iter"
        elif setup_name in ["selected_5", "selected_4"]:
            tickers = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]
            default_config = {
                "data_loader": self._data_loader(tickers=tickers),
                "date_from": "2004-01-01",
                "date_to": "2024-01-01",
                "training_years": 3,
                "tickers": tickers,
                "silence": True,
                "setup_name": setup_name,
            }
            self.mode = "rolling_window"
        elif setup_name.startswith("random_sp500_"):
            default_config = {
                "data_loader": self._data_loader(),
                "date_from": "2004-01-01",
                "date_to": "2024-01-01",
                "tickers": "all",
                "silence": True,
                "setup_name": setup_name,
                "selection_strategy": RandomSP500Selector(
                    num_tickers=int(setup_name.split("_")[-1]),
                    random_seed_setting="year",
                ),
            }
            self.mode = "rolling_window"
        elif setup_name.startswith("momentum_sp500_"):
            default_config = {
                "data_loader": self._data_loader(),
                "date_from": "2004-01-01",
                "date_to": "2024-01-01",
                "training_years": 2,
                "tickers": "all",
                "silence": True,
                "setup_name": setup_name,
                "selection_strategy": MomentumSP500Selector(
                    num_tickers=int(setup_name.split("_")[-1]),
                    momentum_period=100,
                    skip_period=21,
                    training_period=2,
                ),
            }
            self.mode = "rolling_window"
        elif setup_name.startswith("lowvol_sp500_"):
            default_config = {
                "data_loader": self._data_loader(),
                "date_from": "2004-01-01",
                "date_to": "2024-01-01",
                "training_years": 2,
                "tickers": "all",
                "silence": True,
                "setup_name": setup_name,
                "selection_strategy": LowVolatilitySP500Selector(
                    num_tickers=int(setup_name.split("_")[-1]),
                    lookback_period=21,
                    training_period=2,
                ),
            }
            self.mode = "rolling_window"
        elif setup_name.startswith("fincon_selector_sp500_"):
            from llm_traders.finsaber_strategies.fincon_agent_selector import FinConSP500Selector

            default_config = {
                "data_loader": self._data_loader(),
                "date_from": "2004-01-01",
                "date_to": "2024-01-01",
                "training_years": 2,
                "tickers": "all",
                "silence": True,
                "setup_name": setup_name,
                "selection_strategy": FinConSP500Selector(
                    num_tickers=int(setup_name.split("_")[-1]),
                    lookback_years=2,
                    training_period=2,
                ),
            }
            self.mode = "rolling_window"
        else:
            raise NotImplementedError(f"setup_name {setup_name} is not implemented")

        if custom_trade_config is not None:
            default_config.update(custom_trade_config)
        default_config.setdefault("log_base_dir", self.output_dir)

        strat_config = json.load(open(strat_config_path, encoding="utf-8")) if strat_config_path else None
        run_identity = None
        if strat_config is not None:
            default_config, strat_config, run_identity = self._materialize_tradingagents_run_root(
                strategy_class=strategy_class,
                trade_config=default_config,
                strat_config=strat_config,
            )
        if run_identity is not None:
            self._write_tradingagents_launcher_artifacts(
                run_identity=run_identity,
                setup_name=setup_name,
                strategy_class=strategy_class,
                trade_config=default_config,
                strat_config_path=strat_config_path,
                materialized_strat_config=strat_config,
                status="started",
            )
        try:
            self._run_backtest(strategy_class, default_config, strat_config)
        except Exception as exc:
            if run_identity is not None:
                self._write_tradingagents_launcher_artifacts(
                    run_identity=run_identity,
                    setup_name=setup_name,
                    strategy_class=strategy_class,
                    trade_config=default_config,
                    strat_config_path=strat_config_path,
                    materialized_strat_config=strat_config,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
        if run_identity is not None:
            self._write_tradingagents_launcher_artifacts(
                run_identity=run_identity,
                setup_name=setup_name,
                strategy_class=strategy_class,
                trade_config=default_config,
                strat_config_path=strat_config_path,
                materialized_strat_config=strat_config,
                status="completed",
            )

    def _run_backtest(self, strategy_class, trade_config, strat_config):
        operator = FINSABER(trade_config) if strat_config else FINSABERBt(trade_config)

        if self.mode == "rolling_window":
            operator.run_rolling_window(strategy_class, strat_params=strat_config)
        elif self.mode == "iter":
            operator.run_iterative_tickers(strategy_class, strat_params=strat_config)

        aggregate_results_one_strategy(
            trade_config["setup_name"],
            strategy_class.__name__,
            output_dir=trade_config.get("log_base_dir", self.output_dir),
            # Reuse the same explicit benchmark result root that the backend
            # writer used so the final aggregation step cannot drift back to
            # the legacy setup/strategy directory.
            strategy_output_dir=trade_config.get("result_output_dir"),
        )
