import os
import pickle
import warnings

from backtest.strategy.selection import RandomSP500Selector
from backtest.toolkit.custom_exceptions import InsufficientTrainingDataException

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from types import SimpleNamespace
from datetime import datetime, timedelta
from llm_traders.finagent.registry import DATASET, ENVIRONMENT, MEMORY, PROVIDER, PROMPT, PLOTS
from llm_traders.finagent.asset import ASSET
from backtest.toolkit.llm_cost_monitor import get_llm_cost
from llm_traders.finagent.query import DiverseQuery
from llm_traders.finagent.prompt import (prepare_latest_market_intelligence_params,
                                         prepare_low_level_reflection_params,
                                         prepare_high_level_reflection_params,
                                         prepared_tools_params)
from llm_traders.finagent.tools import StrategyAgents
from backtest.strategy.timing_llm.base_strategy_iso import BaseStrategyIso
from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


class FinAgentStrategy(BaseStrategyIso):
    def __init__(self, symbol, market_data_info_path, date_from, date_to, training_period=3):
        super().__init__()
        self.logger.info("Initialising FinAgentStrategy.")

        # Hard-coded configuration values
        self.selected_asset = symbol
        self.asset_type = "company"
        self.workdir = "llm_traders/finagent/workdir/trading"
        self.tag = self.selected_asset
        self.trader_preference = "aggressive_trader"

        self.initial_amount = 100000
        self.transaction_cost_pct = 1e-3

        self.short_term_past_date_range = 1
        self.medium_term_past_date_range = 7
        self.long_term_past_date_range = 14
        self.short_term_next_date_range = 1
        self.medium_term_next_date_range = 7
        self.long_term_next_date_range = 14
        self.look_forward_days = 0
        self.look_back_days = self.long_term_past_date_range
        self.previous_action_look_back_days = 14
        self.top_k = 5
        self.llm_model_id = "gpt-4o-mini"

        # Template paths for valid mode
        self.valid_latest_market_intelligence_template = self._read_template(
            "llm_traders/finagent/res/prompts/template/valid/trading/latest_market_intelligence_summary.html")
        self.valid_past_market_intelligence_template = self._read_template(
            "llm_traders/finagent/res/prompts/template/valid/trading/past_market_intelligence_summary.html")
        self.valid_low_level_reflection_template = self._read_template(
            "llm_traders/finagent/res/prompts/template/valid/trading/low_level_reflection.html")
        self.valid_high_level_reflection_template = self._read_template(
            "llm_traders/finagent/res/prompts/template/valid/trading/high_level_reflection.html")
        self.valid_decision_template = self._read_template("llm_traders/finagent/res/prompts/template/valid/trading/decision.html")

        # Template paths for train mode
        self.train_latest_market_intelligence_template = self._read_template(
            "llm_traders/finagent/res/prompts/template/train/trading/latest_market_intelligence_summary.html")
        self.train_past_market_intelligence_template = self._read_template(
            "llm_traders/finagent/res/prompts/template/train/trading/past_market_intelligence_summary.html")
        self.train_low_level_reflection_template = self._read_template(
            "llm_traders/finagent/res/prompts/template/train/trading/low_level_reflection.html")
        self.train_high_level_reflection_template = self._read_template(
            "llm_traders/finagent/res/prompts/template/train/trading/high_level_reflection.html")
        self.train_decision_template = self._read_template("llm_traders/finagent/res/prompts/template/train/trading/decision.html")

        # Build dataset
        self.dataset = DATASET.build(cfg={
            "type": "Dataset",
            "asset": [self.selected_asset],
            "price_path": "datasets/exp_stocks/price",
            "news_path": "datasets/exp_stocks/news",
            "interval": "1d",
            "workdir": self.workdir,
            "tag": self.tag
        })

        # === Keep original settings for market data loading and period definitions ===
        with open(market_data_info_path, "rb") as f:
            env_data = pickle.load(f)

        if isinstance(training_period, (int, float)):
            train_start_date = datetime.strptime(date_from, "%Y-%m-%d").date() - timedelta(days=365 * training_period)
            train_end_date = datetime.strptime(date_from, "%Y-%m-%d").date() - timedelta(days=1)
        elif isinstance(training_period, tuple):
            train_start_date = datetime.strptime(training_period[0], "%Y-%m-%d").date() if isinstance(
                training_period[0], str) else training_period[0]
            train_end_date = datetime.strptime(training_period[1], "%Y-%m-%d").date() if isinstance(training_period[1],
                                                                                                    str) else \
            training_period[1]
        else:
            raise ValueError("Invalid training_period type.")

        train_data = {k: v for k, v in env_data.items() if train_start_date <= k <= train_end_date and symbol in v["price"]}

        if len(train_data.keys()) == 0:
            raise InsufficientTrainingDataException

        first_avai_date = min(train_data.keys())

        if first_avai_date.year > train_start_date.year:
            raise InsufficientTrainingDataException

        if train_start_date < first_avai_date:
            self.logger.info(
                f"Training start date is earlier than the first available date in the data. Adjusting training start date to {first_avai_date}.")
            train_start_date = first_avai_date


        self.logger.info(f"Training period: {train_start_date} to {train_end_date}")


        test_start_date = datetime.strptime(date_from, "%Y-%m-%d").date()
        test_end_date = datetime.strptime(date_to, "%Y-%m-%d").date()
        self.logger.info(f"Testing period for {self.selected_asset}: {test_start_date} to {test_end_date}")
        test_data = {k: v for k, v in env_data.items() if test_start_date <= k <= test_end_date}
        # === End original settings ===

        # Build environments using the dates above
        self.train_env = ENVIRONMENT.build({
            "type": "EnvironmentTrading",
            "mode": "train",
            "dataset": self.dataset,
            "selected_asset": self.selected_asset,
            "asset_type": self.asset_type,
            "start_date": train_start_date,
            "end_date": train_end_date,
            "look_back_days": self.look_back_days,
            "look_forward_days": self.look_forward_days,
            "initial_amount": self.initial_amount,
            "transaction_cost_pct": self.transaction_cost_pct,
            "discount": 1.0,
        })
        self.train_env.reset()

        self.test_env = ENVIRONMENT.build({
            "type": "EnvironmentTrading",
            "mode": "train",
            "dataset": self.dataset,
            "selected_asset": self.selected_asset,
            "asset_type": self.asset_type,
            "start_date": test_start_date,
            "end_date": test_end_date,
            "look_back_days": self.look_back_days,
            "look_forward_days": self.look_forward_days,
            "initial_amount": self.initial_amount,
            "transaction_cost_pct": self.transaction_cost_pct,
            "discount": 1.0,
        })
        self.test_env.reset()

        if hasattr(self.train_env, 'set_data'):
            self.train_env.set_data(train_data)
        if hasattr(self.test_env, 'set_data'):
            self.test_env.set_data(test_data)

        # Build plots
        self.plots = PLOTS.build({
            "type": "PlotsInterface",
            "root": "./",
            "workdir": self.workdir,
            "tag": self.tag,
        })

        # Build provider
        self.provider = PROVIDER.build({
            "type": "OpenAIProvider",
            "provider_cfg_path": "finagent/configs/openai_config.json",
        })

        # Build memory (using provider's embedding dimension)
        self.memory = MEMORY.build({
            "type": "MemoryInterface",
            "root": "./",
            "symbols": [self.selected_asset],
            "memory_path": "memory",
            "embedding_dim": self.provider.get_embedding_dim(),
            "max_recent_steps": 5,
            "workdir": self.workdir,
            "tag": self.tag
        })

        # Build prompts
        self.latest_market_intelligence_prompt = PROMPT.build({
            "type": "LatestMarketIntelligenceSummaryTrading",
            "model": self.llm_model_id
        })
        self.past_market_intelligence_prompt = PROMPT.build({
            "type": "PastMarketIntelligenceSummaryTrading",
            "model": self.llm_model_id
        })
        self.low_level_reflection_prompt = PROMPT.build({
            "type": "LowLevelReflectionTrading",
            "model": self.llm_model_id,
            "short_term_past_date_range": self.short_term_past_date_range,
            "medium_term_past_date_range": self.medium_term_past_date_range,
            "long_term_past_date_range": self.long_term_past_date_range,
            "short_term_next_date_range": self.short_term_next_date_range,
            "medium_term_next_date_range": self.medium_term_next_date_range,
            "long_term_next_date_range": self.long_term_next_date_range,
            "look_back_days": self.long_term_past_date_range,
            "look_forward_days": self.long_term_next_date_range
        })
        self.high_level_reflection_prompt = PROMPT.build({
            "type": "HighLevelReflectionTrading",
            "model": self.llm_model_id,
            "previous_action_look_back_days": self.previous_action_look_back_days
        })
        self.decision_prompt = PROMPT.build({
            "type": "DecisionTrading",
            "model": self.llm_model_id
        })
        self.decision_template = self.valid_decision_template

        # Build diverse query and strategy agents
        self.diverse_query = DiverseQuery(self.memory, self.provider, top_k=self.top_k)
        self.strategy_agents = StrategyAgents()

        # Set experiment path
        self.exp_path = os.path.join(self.workdir, "trading", self.selected_asset)

        # Internal trading records for reflections
        self.trading_records = {
            "symbol": [],
            "day": [],
            "value": [],
            "cash": [],
            "position": [],
            "ret": [],
            "date": [],
            "price": [],
            "discount": [],
            "kline_path": [],
            "trading_path": [],
            "total_profit": [],
            "total_return": [],
            "action": [],
            "reasoning": [],
        }

    def _read_template(self, path):
        with open(path, 'r') as f:
            return f.read()

    def run_step(self, state, info, mode="valid"):
        save_dir = mode
        params = {}
        # Plot kline and add to params
        kline_path = self.plots.plot_kline(state=state, info=info, save_dir=save_dir)
        params["kline_path"] = kline_path

        tool_dict = dict(
            selected_asset=self.selected_asset,
            tool_use_best_params=True,
            tool_params_dir="finagent/res/strategy_record/trading"
        )


        tool_cfg = SimpleNamespace(**tool_dict)
        # Update with tools parameters
        tools_params = prepared_tools_params(state=state, info=info, params=params,
                                             memory=self.memory, provider=self.provider,
                                             diverse_query=self.diverse_query, strategy_agents=self.strategy_agents,
                                             cfg=tool_cfg, mode=mode)
        params.update(tools_params)

        # Latest market intelligence
        template = self.valid_latest_market_intelligence_template if mode == "valid" else self.train_latest_market_intelligence_template
        latest_res = self.latest_market_intelligence_prompt.run(
            state=state, info=info, params=params,
            template=template, memory=self.memory,
            provider=self.provider, diverse_query=self.diverse_query,
            exp_path=self.exp_path, save_dir=save_dir
        )
        params.update(prepare_latest_market_intelligence_params(state=state, info=info,
                                                                params=params, memory=self.memory,
                                                                provider=self.provider,
                                                                diverse_query=self.diverse_query))
        self.latest_market_intelligence_prompt.add_to_memory(state=state, info=info,
                                                             res=latest_res, memory=self.memory, provider=self.provider)

        # Past market intelligence
        template = self.valid_past_market_intelligence_template if mode == "valid" else self.train_past_market_intelligence_template
        self.past_market_intelligence_prompt.run(
            state=state, info=info, template=template, params=params,
            memory=self.memory, provider=self.provider, diverse_query=self.diverse_query,
            exp_path=self.exp_path, save_dir=save_dir
        )

        # Low-level reflection
        template = self.valid_low_level_reflection_template if mode == "valid" else self.train_low_level_reflection_template
        low_res = self.low_level_reflection_prompt.run(
            state=state, info=info, template=template, params=params,
            memory=self.memory, provider=self.provider, diverse_query=self.diverse_query,
            exp_path=self.exp_path, save_dir=save_dir
        )
        params.update(prepare_low_level_reflection_params(state=state, info=info,
                                                          params=params, memory=self.memory,
                                                          provider=self.provider, diverse_query=self.diverse_query))
        self.low_level_reflection_prompt.add_to_memory(state=state, info=info,
                                                       res=low_res, memory=self.memory, provider=self.provider)

        # Trading plot update
        trading_path = self.plots.plot_trading(records=self.trading_records, info=info, save_dir=save_dir) if \
        self.trading_records["date"] else None
        params["trading_path"] = trading_path

        # High-level reflection
        params.update({
            "previous_date": self.trading_records["date"],
            "previous_action": self.trading_records["action"],
            "previous_reasoning": self.trading_records["reasoning"],
        })
        template = self.valid_high_level_reflection_template if mode == "valid" else self.train_high_level_reflection_template
        high_res = self.high_level_reflection_prompt.run(
            state=state, info=info, template=template, params=params,
            memory=self.memory, provider=self.provider, diverse_query=self.diverse_query,
            exp_path=self.exp_path, save_dir=save_dir
        )
        params.update(prepare_high_level_reflection_params(state=state, info=info, params=params,
                                                           memory=self.memory, provider=self.provider,
                                                           diverse_query=self.diverse_query))
        self.high_level_reflection_prompt.add_to_memory(state=state, info=info,
                                                        res=high_res, memory=self.memory, provider=self.provider)

        # Trader preference
        params["trader_preference"] = ASSET.get_trader(self.trader_preference)

        # Final decision prompt
        template = self.valid_decision_template if mode == "valid" else self.train_decision_template
        decision_res = self.decision_prompt.run(
            state=state, info=info, template=template, params=params,
            memory=self.memory, provider=self.provider, diverse_query=self.diverse_query,
            exp_path=self.exp_path, save_dir=save_dir
        )

        # Update internal trading records
        self.trading_records["symbol"].append(info["symbol"])
        self.trading_records["day"].append(info["day"])
        self.trading_records["value"].append(info["value"])
        self.trading_records["cash"].append(info["cash"])
        self.trading_records["position"].append(info["position"])
        self.trading_records["ret"].append(info["ret"])
        self.trading_records["date"].append(info["date"])
        self.trading_records["price"].append(info["price"])
        self.trading_records["discount"].append(info["discount"])
        self.trading_records["kline_path"].append(kline_path)
        self.trading_records["trading_path"].append(trading_path)
        self.trading_records["total_profit"].append(info["total_profit"])
        self.trading_records["total_return"].append(info["total_return"])
        self.trading_records["action"].append(decision_res["response_dict"]["action"])
        self.trading_records["reasoning"].append(decision_res["response_dict"]["reasoning"])

        return decision_res["response_dict"]["action"]

    def on_data(self, date: datetime.date, today_data: dict[str, float], framework: FINSABERFrameworkHelper):
        state = self.test_env.get_state()
        info = self.test_env.get_info()

        if self.selected_asset not in today_data['price']:
            self.logger.info(f"Training environment completed, or symbol {self.selected_asset} has been delisted on {date}.")
            framework.sell(date, self.selected_asset, 1e-10,
                           framework.portfolio[self.selected_asset]["quantity"])
            return "done"

        info["price"] = today_data['price'][self.selected_asset]
        action = self.run_step(state, info, mode="valid")

        # self.logger.info(f"Agent decision on {date}: {action}")

        if action == "BUY" or action == 1:
            if framework.cash >= info["price"]:
                framework.buy(date, self.selected_asset, info["price"], -1)
                self.logger.info(f"Executed BUY on {date} for {self.selected_asset}. Estimated cost: ${get_llm_cost():.2f}.")
        elif action == "SELL" or action == -1:
            if self.selected_asset in framework.portfolio:
                framework.sell(date, self.selected_asset, info["price"],
                               framework.portfolio[self.selected_asset]["quantity"])
                self.logger.info(f"Executed SELL on {date} for {self.selected_asset}. Estimated cost: ${get_llm_cost():.2f}.")

    def train(self):
        self.logger.info("Starting training...")
        total_steps = self.train_env.end_day - self.train_env.init_day
        self.logger.info(f"Total training steps: {total_steps}")
        for step in range(total_steps):
            state, reward, done, truncated, info = self.train_env.step()

            if done:
                self.logger.info("Training environment completed.")
                break
            self.run_step(state, info, mode="train")
            if total_steps > 10 and (step % (total_steps // 10) == 0 or step == total_steps - 1):
                self.logger.info(f"Training progress: {step + 1}/{total_steps} steps completed. Estimated cost: ${get_llm_cost():.2f}.")


if __name__ == "__main__":
    from backtest.data_util import FinMemDataset
    from backtest.finsaber import FINSABER

    trade_config = {
        "tickers": [
            "TSLA",
            # "NFLX",
            # "AMZN",
            # "MSFT",
            # "COIN"
        ],
        "silence": False,
        "setup_name": "selected_4",
        "date_from": "2012-01-01",
        "date_to": "2014-01-01",
        "data_loader": FinMemDataset(pickle_file="data/finmem_data/stock_data_cherrypick_2000_2024.pkl"),
        # "date_from": "2022-10-06",
        # "date_to": "2023-04-10"
    }
    # accept bash arguments for date_from and date_to
    # import sys
    # date_from = sys.argv[1]
    # date_to = sys.argv[2]


    # trade_config = {
    #     "tickers": "all",
    #     "silence": True,
    #     "setup_name": "random_sp500_5",
    #     "date_from": date_from,
    #     "date_to": date_to,
    #     "data_loader": FinMemDataset(pickle_file="data/finmem_data/stock_data_sp500_2000_2024.pkl"),
    #     "selection_strategy": RandomSP500Selector(
    #         num_tickers=5,
    #         random_seed_setting="year"
    #     )
    # }
    engine = FINSABER(trade_config)
    strat_params = {
        "market_data_info_path": "data/finmem_data/stock_data_cherrypick_2000_2024.pkl",
        "date_from": "$date_from",  # auto calculate inside the backtest engine,
        "date_to": "$date_to",  # auto calculate inside the backtest engine,
        "symbol": "$symbol",
        # "training_period": ("2021-08-17", "2022-10-05")
        "training_period": 2
    }
    # metrics = engine.run_rolling_window(FinAgentStrategy, strat_params=strat_params)
    # print(metrics)
    # from backtest.toolkit.operation_utils import aggregate_results_one_strategy
    ticker_metrics = engine.run_iterative_tickers(FinAgentStrategy, strat_params=strat_params)
    print(ticker_metrics)
    # from backtest.toolkit.operation_utils import aggregate_results_one_strategy
    # aggregate_results_one_strategy("selected_4", "FinAgentStrategy")
