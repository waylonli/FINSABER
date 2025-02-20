from backtest.backtest_engine_iso import BacktestingEngineIso
from backtest.strategy.timing_iso.base_strategy_iso import BaseStrategyIso
from backtest.toolkit.backtest_framework_iso import BacktestFrameworkIso
import toml
import pickle
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
from llm_traders.finmem.puppy import MarketEnvironment, LLMAgent, RunMode
from dotenv import load_dotenv
from backtest.toolkit.llm_cost_monitor import get_llm_cost
load_dotenv()


class FinMemStrategy(BaseStrategyIso):
    def __init__(
            self,
            symbol,
            config_path,
            market_data_info_path,
            date_from,
            date_to,
            training_period=2,  # 2 years of daily data for training
    ):
        super().__init__()
        self.logger.info("Initialising FinMemStrategy.")
        self.config = toml.load(config_path)
        self.config["general"]["trading_symbol"] = symbol
        self.config["general"]["character_string"] = symbol

        with open(market_data_info_path, "rb") as f:
            env_data_pkl = pickle.load(f)

        if type(training_period) == float or type(training_period) == int:
            train_start_date = datetime.strptime(date_from, "%Y-%m-%d").date() - timedelta(
                days=365 * training_period) if type(date_from) == str else date_from - timedelta(days=365 * training_period)
            # the end date should be 1 day before the start date of the testing period
            train_end_date = datetime.strptime(date_from, "%Y-%m-%d").date() - timedelta(days=1) if type(date_from) == str else date_from - timedelta(days=1)
        elif type(training_period) == tuple:
            train_start_date = datetime.strptime(training_period[0], "%Y-%m-%d").date() if type(training_period[0]) == str else training_period[0]
            train_end_date = datetime.strptime(training_period[1], "%Y-%m-%d").date() if type(training_period[1]) == str else training_period[1]
        self.logger.info(f"Training period: {train_start_date} to {train_end_date}")

        train_env_data_pkl = {k: v for k, v in env_data_pkl.items() if k >= train_start_date and k <= train_end_date}
        self.train_enviroment = MarketEnvironment(
            symbol=self.config["general"]["trading_symbol"],
            env_data_pkl=train_env_data_pkl,
            start_date=train_start_date,
            end_date=train_end_date,
        )

        test_start_date = datetime.strptime(date_from, "%Y-%m-%d").date() if type(date_from) == str else date_from
        test_end_date = datetime.strptime(date_to, "%Y-%m-%d").date() if type(date_to) == str else date_to
        self.logger.info(f"Testing period: {test_start_date} to {test_end_date}")

        test_env_data_pkl = {k: v for k, v in env_data_pkl.items() if k >= test_start_date and k <= test_end_date}
        self.test_enviroment = MarketEnvironment(
            symbol=self.config["general"]["trading_symbol"],
            env_data_pkl=test_env_data_pkl,
            start_date=test_start_date,
            end_date=test_end_date,
        )
        self.agent = LLMAgent.from_config(self.config)

    def on_data(
            self,
            date: datetime.date,
            prices: dict[str, float],
            framework: BacktestFrameworkIso
    ):
        # self.logger.info(f"Today's price for {self.config['general']['trading_symbol']}: {prices[self.config['general']['trading_symbol']]}")
        market_info = self.test_enviroment.step()
        if market_info[-1]:  # if done break
            self.logger.info("Test environment completed.")
            return

        if date != self.test_enviroment.cur_date:
            self.logger.warning(f"Date mismatch: {date} vs {self.test_enviroment.cur_date}")

        decision = self.agent.step(market_info=market_info, run_mode=RunMode.Test)['direction']
        # self.logger.info(f"Agent decision on {date}: {decision}")

        if decision == 1:
            if framework.cash >= prices[self.config["general"]["trading_symbol"]]:
                framework.buy(
                    date,
                    self.config["general"]["trading_symbol"],
                    prices[self.config["general"]["trading_symbol"]],
                    -1  # buy all available cash
                )
                self.logger.info(f"Executed BUY on {date} for {self.config['general']['trading_symbol']}.")
        elif decision == -1:
            if self.config["general"]["trading_symbol"] in framework.portfolio:
                framework.sell(
                    date,
                    self.config["general"]["trading_symbol"],
                    prices[self.config["general"]["trading_symbol"]],
                    framework.portfolio[self.config["general"]["trading_symbol"]]["quantity"]
                )
                self.logger.info(f"Executed SELL on {date} for {self.config['general']['trading_symbol']}.")
            # else:
            #     self.logger.warning(
            #         f"Insufficient holdings to sell {self.config['general']['trading_symbol']} on {date}.")
        # else:
        #     self.logger.info(f"No action taken on {date}.")

    def train(self):
        run_mode_var = RunMode.Train
        self.logger.info("Starting training...")
        total_steps = self.train_enviroment.simulation_length

        for step in range(total_steps):
            self.agent.counter += 1
            market_info = self.train_enviroment.step()
            if market_info[-1]:  # if done break
                self.logger.info("Training environment completed.")
                break

            self.agent.step(market_info=market_info, run_mode=run_mode_var)  # type: ignore

            # Log progress manually every 10% of completion
            if total_steps > 10:
                if step % (total_steps // 10) == 0 or step == total_steps - 1:
                    self.logger.info(f"Training progress: {step + 1}/{total_steps} steps completed. Estimated cost: ${get_llm_cost():.2f}.")
            else:
                self.logger.info(f"Training progress: {step + 1}/{total_steps} steps completed. Estimated cost: ${get_llm_cost():.2f}.")

        # import pdb; pdb.set_trace()
        # save result after finish
        # the_agent.save_checkpoint(path=result_path, force=True)
        # environment.save_checkpoint(path=result_path, force=True)

if __name__ == "__main__":
    # empty the log dir llm_traders/finmem/data/04_model_output_log/*.log
    import os
    import glob
    log_dir = "llm_traders/finmem/data/04_model_output_log"
    for f in glob.glob(os.path.join(log_dir, "*.log")):
        os.remove(f)

    trade_config = {
        # "tickers": ["COIN"],
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", ],
        "silence": False,
        "selection_strategy": "selected_5",
        "date_from": "2014-01-01",
        "date_to": "2024-01-01",
        "all_data": "data/finmem_data/stock_data_cherrypick_2000_2024.pkl"
        # "date_from": "2022-10-06",
        # "date_to": "2023-04-10"
    }
    engine = BacktestingEngineIso(trade_config)

    strat_params = {
        "config_path": "llm_traders/finmem/config/tsla_gpt_config.toml",
        "market_data_info_path": "data/finmem_data/stock_data_cherrypick_2000_2024.pkl",
        "date_from": "$date_from", # auto calculate inside the backtest engine,
        "date_to": "$date_to", # auto calculate inside the backtest engine,
        "symbol": "$symbol",
        # "training_period": ("2021-08-17", "2022-10-05")
        "training_period": 3
    }

    # ticker_metrics = engine.run_iterative_tickers(FinMemStrategy, strat_params=strat_params)
    # print(ticker_metrics)

    ticker_metrics = engine.run_rolling_window(FinMemStrategy, strat_params=strat_params)
    # from backtest.toolkit.operation_utils import aggregate_results_one_strategy
    # aggregate_results_one_strategy("cherry_pick_both_finmem", "FinMemStrategy")