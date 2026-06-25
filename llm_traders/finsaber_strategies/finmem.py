from backtest.strategy.timing_llm.base_strategy_iso import BaseStrategyIso
from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper
import toml
import warnings
import os
from backtest.data_util import trading_data_to_env_dict
from backtest.toolkit.custom_exceptions import InsufficientTrainingDataException

warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
from llm_traders.finmem.data_loading import prepare_finmem_trading_data
from llm_traders.finmem.puppy import MarketEnvironment, LLMAgent, RunMode
from llm_traders.finsaber_strategies.finmem_artifacts import (
    ArtifactWindow,
    FinMemArtifactWriter,
)
from dotenv import load_dotenv
from backtest.toolkit.llm_cost_monitor import get_llm_cost
load_dotenv()

class FinMemStrategy(BaseStrategyIso):
    def __init__(
            self,
            symbol,
            config_path,
            date_from,
            date_to,
            market_data_info_path=None,
            market_data_root=None,
            data_loader=None,
            training_period=2,  # 2 years of daily data for training
            use_filing_sections=True,
            filing_section_map=None,
            filing_payload_kind="auto",
            filing_failure_mode="empty",
            filing_merge_policy="latest",
            artifact_config=None,
    ):
        super().__init__()
        self.logger.info(f"Initialising FinMemStrategy for backtesting {symbol}.")
        self.config = toml.load(config_path)
        self.config["general"]["trading_symbol"] = symbol
        self.config["general"]["character_string"] = symbol
        self._post_train_artifacts_saved = False
        self._test_state_artifacts_saved = False

        if type(training_period) == float or type(training_period) == int:
            train_start_date = datetime.strptime(date_from, "%Y-%m-%d").date() - timedelta(
                days=365 * training_period) if type(date_from) == str else date_from - timedelta(days=365 * training_period)
            # the end date should be 1 day before the start date of the testing period
            train_end_date = datetime.strptime(date_from, "%Y-%m-%d").date() - timedelta(days=1) if type(date_from) == str else date_from - timedelta(days=1)
        elif type(training_period) == tuple or type(training_period) == list:
            train_start_date = datetime.strptime(training_period[0], "%Y-%m-%d").date() if type(training_period[0]) == str else training_period[0]
            train_end_date = datetime.strptime(training_period[1], "%Y-%m-%d").date() if type(training_period[1]) == str else training_period[1]
        else:
            raise ValueError("Invalid training_period type.")
        self.logger.info(f"Training period: {train_start_date} to {train_end_date}")

        test_start_date = datetime.strptime(date_from, "%Y-%m-%d").date() if type(date_from) == str else date_from
        test_end_date = datetime.strptime(date_to, "%Y-%m-%d").date() if type(date_to) == str else date_to
        requested_train_start = train_start_date
        requested_train_end = train_end_date
        requested_test_start = test_start_date
        requested_test_end = test_end_date

        market_data = prepare_finmem_trading_data(
            symbol=symbol,
            data_loader=data_loader,
            market_data_root=market_data_root,
            market_data_info_path=market_data_info_path,
            use_filing_sections=use_filing_sections,
            filing_section_map=filing_section_map,
            filing_payload_kind=filing_payload_kind,
            filing_failure_mode=filing_failure_mode,
            filing_merge_policy=filing_merge_policy,
        )
        env_data_pkl = trading_data_to_env_dict(
            market_data,
            start_date=min(train_start_date, test_start_date),
            end_date=test_end_date,
            tickers=[symbol],
        )
        train_env_data_pkl = {k: v for k, v in env_data_pkl.items() if k >= train_start_date and k <= train_end_date and symbol in v["price"]}

        if len(train_env_data_pkl.keys()) == 0:
            raise InsufficientTrainingDataException

        # let train_start_date be the first available date in the env_data_pkl
        first_avai_date = min(train_env_data_pkl.keys())

        if first_avai_date.year > train_start_date.year:
            raise InsufficientTrainingDataException

        if train_start_date < first_avai_date:
            self.logger.info(f"Training start date is earlier than the first available date in the data. Adjusting training start date to {first_avai_date}.")
            train_start_date = first_avai_date

        self.train_enviroment = MarketEnvironment(
            symbol=self.config["general"]["trading_symbol"],
            env_data_pkl=train_env_data_pkl,
            start_date=train_start_date,
            end_date=train_end_date,
        )

        self.logger.info(f"Testing period: {test_start_date} to {test_end_date}")

        test_env_data_pkl = {k: v for k, v in env_data_pkl.items() if k >= test_start_date and k <= test_end_date}
        self.test_enviroment = MarketEnvironment(
            symbol=self.config["general"]["trading_symbol"],
            env_data_pkl=test_env_data_pkl,
            start_date=test_start_date,
            end_date=test_end_date,
        )
        self.agent = LLMAgent.from_config(self.config)
        normalized_artifact_config = dict(artifact_config or {})
        self.agent.configure_tracing(
            enabled=bool(normalized_artifact_config.get("enabled", False)),
            capture_query_trace=bool(
                normalized_artifact_config.get("save_query_trace", True)
            ),
            capture_llm_trace=bool(
                normalized_artifact_config.get("save_llm_trace", True)
            ),
        )
        resolved_strategy_params = {
            "symbol": symbol,
            "config_path": config_path,
            "date_from": requested_test_start,
            "date_to": requested_test_end,
            "market_data_info_path": market_data_info_path,
            "market_data_root": market_data_root,
            "training_period": training_period,
            "use_filing_sections": use_filing_sections,
            "filing_section_map": filing_section_map,
            "filing_payload_kind": filing_payload_kind,
            "filing_failure_mode": filing_failure_mode,
            "filing_merge_policy": filing_merge_policy,
        }
        self.artifact_writer = FinMemArtifactWriter(
            artifact_config=normalized_artifact_config,
            symbol=symbol,
            config_path=config_path,
            resolved_strategy_params=resolved_strategy_params,
            finmem_config=self.config,
            requested_train_window=ArtifactWindow(
                requested_start=requested_train_start,
                requested_end=requested_train_end,
                effective_start=self.train_enviroment.start_date,
                effective_end=self.train_enviroment.end_date,
            ),
            requested_test_window=ArtifactWindow(
                requested_start=requested_test_start,
                requested_end=requested_test_end,
                effective_start=self.test_enviroment.start_date,
                effective_end=self.test_enviroment.end_date,
            ),
            input_data_loader=data_loader,
            runtime_market_data=market_data,
            filing_options={
                "use_filing_sections": use_filing_sections,
                "filing_section_map": filing_section_map,
                "filing_payload_kind": filing_payload_kind,
                "filing_failure_mode": filing_failure_mode,
                "filing_merge_policy": filing_merge_policy,
            },
        )

    def on_data(
            self,
            date: datetime.date,
            today_data: dict[str, float],
            framework: FINSABERFrameworkHelper
    ):
        prices = today_data["price"]
        symbol = self.config["general"]["trading_symbol"]
        cur_price = prices[symbol]
        if isinstance(cur_price, dict):
            cur_price = cur_price["adjusted_close"]

        # self.logger.info(f"{date} price for {symbol}: {cur_price}")
        market_info = self.test_enviroment.step()

        if market_info[-1]:
            self.logger.info(f"Test environment completed!")
            if not self._test_state_artifacts_saved:
                # This snapshot is taken before the outer framework performs
                # final liquidation and metrics evaluation.
                self.artifact_writer.save_test_state(
                    agent=self.agent,
                    environment=self.test_enviroment,
                    capture_stage="strategy_done_pre_finalization",
                    snapshot_reason="strategy_done",
                    framework_status=True,
                )
                self._test_state_artifacts_saved = True
            return "done"

        if date != self.test_enviroment.cur_date:
            self.logger.warning(f"Date mismatch: {date} vs {self.test_enviroment.cur_date}")

        decision = self.agent.step(market_info=market_info, run_mode=RunMode.Test)['direction']
        # self.logger.info(f"Agent decision on {date}: {decision}")

        if decision == 1:
            if framework.cash >= cur_price:
                framework.buy(
                    date,
                    symbol,
                    cur_price,
                    -1  # buy all available cash
                )
                self.logger.info(f"Executed BUY on {date} for {symbol}.")
        elif decision == -1:
            if symbol in framework.portfolio:
                framework.sell(
                    date,
                    symbol,
                    cur_price,
                    framework.portfolio[symbol]["quantity"]
                )
                self.logger.info(f"Executed SELL on {date} for {symbol}.")
            # else:
            #     self.logger.warning(
            #         f"Insufficient holdings to sell {self.config['general']['trading_symbol']} on {date}.")
        # else:
        #     self.logger.info(f"No action taken on {date}.")

    def train(self):
        run_mode_var = RunMode.Train
        self.logger.info("Starting training...")
        total_steps = self.train_enviroment.simulation_length
        self.logger.info(f"Total training steps: {total_steps}")

        for step in range(total_steps):
            self.agent.counter += 1
            # self.logger.info(f"Step {step} / {self.train_enviroment.cur_date}")

            market_info = self.train_enviroment.step()
            # self.logger.info(f"Market info step forward done!")

            if market_info[-1]:  # if done break
                self.logger.info("Training environment completed, or symbol has been delisted.")
                break

            self.agent.step(market_info=market_info, run_mode=run_mode_var)  # TODO Here occurs SIGSEGV in step 2

            # self.logger.info(f"Agent step forward done!")

            # Log progress manually every 10% of completion
            if total_steps > 10:
                if step % (total_steps // 10) == 0 or step == total_steps - 1:
                    self.logger.info(f"Training progress: {step + 1}/{total_steps} steps completed. Estimated cost: ${get_llm_cost():.2f}.")
            else:
                self.logger.info(f"Training progress: {step + 1}/{total_steps} steps completed. Estimated cost: ${get_llm_cost():.2f}.")

        if not self._post_train_artifacts_saved:
            self.artifact_writer.save_post_train(
                agent=self.agent,
                environment=self.train_enviroment,
            )
            self._post_train_artifacts_saved = True

    def finalize_backtest_artifacts(self, framework_status: bool):
        if self._test_state_artifacts_saved:
            return

        if framework_status:
            capture_stage = "framework_completed_without_strategy_done"
            snapshot_reason = "framework_run_completed_without_done_signal"
        else:
            capture_stage = "framework_aborted"
            snapshot_reason = "framework_run_returned_false"

        self.artifact_writer.save_test_state(
            agent=self.agent,
            environment=self.test_enviroment,
            capture_stage=capture_stage,
            snapshot_reason=snapshot_reason,
            framework_status=framework_status,
        )
        self._test_state_artifacts_saved = True

if __name__ == "__main__":
    from backtest.data_util import create_finsaber2_data_loader
    from backtest.finsaber import FINSABER

    # empty the log dir llm_traders/finmem/data/04_model_output_log/*.log
    import os
    import glob
    log_dir = "llm_traders/finmem/data/04_model_output_log"
    for f in glob.glob(os.path.join(log_dir, "*.log")):
        os.remove(f)

    data = create_finsaber2_data_loader(tickers=["TSLA"])
    trade_config = {
        # "tickers": ["COIN","TSLA", "NFLX", "AMZN", "MSFT",],
        "tickers": ["TSLA"],
        "silence": False,
        "setup_name": "debug",
        "date_from": "2012-01-01",
        "date_to": "2014-01-01",
        "data_loader": data,
        # "date_from": "2022-10-06",
        # "date_to": "2023-04-10"
    }

    engine = FINSABER(trade_config)

    strat_params = {
        "config_path": "strats_configs/finmem_gpt_config.toml",
        "data_loader": "$data_loader",
        "date_from": "$date_from", # auto calculate inside the backtest engine,
        "date_to": "$date_to", # auto calculate inside the backtest engine,
        "symbol": "$symbol",
        # "training_period": ("2021-08-17", "2022-10-05")
        "training_period": 2
    }

    ticker_metrics = engine.run_iterative_tickers(FinMemStrategy, strat_params=strat_params)
    print(ticker_metrics)
    # import pdb; pdb.set_trace()

    # ticker_metrics = engine.run_rolling_window(FinMemStrategy, strat_params=strat_params)
    # from backtest.toolkit.operation_utils import aggregate_results_one_strategy
    # aggregate_results_one_strategy("cherry_pick_both_finmem", "FinMemStrategy")
