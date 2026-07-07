import json
import os

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

    def _materialize_finmem_run_root(
        self,
        *,
        strategy_class,
        trade_config: dict,
        strat_config: dict | None,
    ):
        if (
            strategy_class.__name__ != "FinMemStrategy"
            or strat_config is None
            or self.mode != "iter"
        ):
            return trade_config, strat_config, None

        import toml

        from llm_traders.finsaber_strategies.finmem_artifacts import (
            materialize_finmem_run_identity,
        )

        artifact_config = dict(strat_config.get("artifact_config") or {})
        if not bool(artifact_config.get("enabled", False)):
            return trade_config, strat_config, None

        config_path = strat_config.get("config_path")
        if not config_path:
            raise ValueError("FinMemStrategy strat_config must provide config_path.")
        resolved_config_path = os.path.abspath(config_path)

        resolved_strategy_params = {
            "config_path": resolved_config_path,
            "date_from": trade_config.get("date_from"),
            "date_to": trade_config.get("date_to"),
            "market_data_info_path": strat_config.get("market_data_info_path"),
            "market_data_root": strat_config.get("market_data_root"),
            "training_period": strat_config.get("training_period", 2),
            "use_filing_sections": strat_config.get("use_filing_sections", True),
            "filing_section_map": strat_config.get("filing_section_map"),
            "filing_payload_kind": strat_config.get("filing_payload_kind", "auto"),
            "filing_failure_mode": strat_config.get("filing_failure_mode", "empty"),
            "filing_merge_policy": strat_config.get("filing_merge_policy", "latest"),
        }
        finmem_config = toml.load(resolved_config_path)
        run_identity = materialize_finmem_run_identity(
            artifact_config=artifact_config,
            output_root=trade_config.get("log_base_dir", self.output_dir),
            setup_name=trade_config["setup_name"],
            strategy_name=strategy_class.__name__,
            config_path=resolved_config_path,
            finmem_config=finmem_config,
            resolved_strategy_params=resolved_strategy_params,
        )

        materialized_trade_config = dict(trade_config)
        materialized_trade_config["result_output_dir"] = str(
            run_identity.benchmark_results_dir
        )

        materialized_strat_config = dict(strat_config)
        materialized_artifact_config = {
            **artifact_config,
            "profile_name": run_identity.profile_name,
            "run_key": run_identity.run_key,
            "benchmark_results_dir": str(run_identity.benchmark_results_dir),
        }
        if artifact_config.get("root") in (None, ""):
            materialized_artifact_config["root"] = str(run_identity.artifact_root)
        materialized_strat_config["artifact_config"] = materialized_artifact_config
        materialized_strat_config["config_path"] = resolved_config_path
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
        default_config, strat_config, _ = self._materialize_finmem_run_root(
            strategy_class=strategy_class,
            trade_config=default_config,
            strat_config=strat_config,
        )
        self._run_backtest(strategy_class, default_config, strat_config)

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
            strategy_output_dir=trade_config.get("result_output_dir"),
        )
