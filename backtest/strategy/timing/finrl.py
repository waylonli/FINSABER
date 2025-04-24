import datetime
import pandas as pd
import numpy as np
import itertools
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.backtest_engine import BacktestingEngine
from backtest.toolkit.operation_utils import aggregate_results_one_strategy

# FinRL imports
from rl_traders.finrl.finrl.agents.stablebaselines3.models import DRLAgent
from rl_traders.finrl.finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from rl_traders.finrl.finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from rl_traders.finrl.finrl.meta.preprocessor.preprocessors import FeatureEngineer


class FinRLStrategy(BaseStrategy):
    params = (
        ("algorithm", "a2c"),           # Options: A2C, DDPG, PPO, SAC, TD3
        ("total_timesteps", 50000),
        ("initial_amount", 100000),
        ("total_days", 0),
        ("train_period", 252 * 2),  # Train on past 3 years of daily data
        # ("train_period", 252),  # Train on past 3 years of daily data
    )

    def __init__(self, train_data: pd.DataFrame, strat_params=None):
        super().__init__()
        # print(f"RL Strategy initialised with {self.params.algorithm} algorithm.")
        if train_data is None:
            raise ValueError("Train data must be provided.")

        self.df_actions = []
        self.df_account_value = []

        self.raw_train_data = train_data.copy()
        # Convert multi-index raw data to long format suitable for FeatureEngineer
        self.formatted_raw = self.format_raw_data_for_fe(self.raw_train_data)
        # Preprocess using FinRL's FeatureEngineer
        self.train_data = self.preprocess_data(self.formatted_raw)

        self.test_data = []
        for test_d in self.datas:
            test_df = test_d._dataname
            # rename index as date column
            test_start_date = test_df.index[0]
            test_end_date = test_df.index[-1]

            test_df = test_df.rename_axis('date').reset_index()
            test_df['tic'] = test_d._name

            # concat the train and test data for calculating technical indicators
            test_df = pd.concat([self.formatted_raw, test_df], ignore_index=True)
            test_df = self.preprocess_data(test_df)
            # get the date range for the test data
            test_df = test_df[(test_df['date'] >= test_start_date) & (test_df['date'] <= test_end_date)]
            # sort by date and reset index
            test_df = test_df.sort_values(['date', 'tic']).reset_index(drop=True)

            self.test_data.append(test_df)

        if self.train_data.empty:
            raise ValueError("Preprocessed train data is empty.")
        self.train_drl_model(
            algorithm=self.params.algorithm,
            total_timesteps=self.params.total_timesteps
        )
        # Build history for state computation
        self.history = {tic: [] for tic in self.train_data["tic"].unique()}

    def format_raw_data_for_fe(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw_data from wide MultiIndex format to a long DataFrame with single-level columns.
        Expected raw_data:
            - Index: dates
            - Columns: MultiIndex with levels: [feature, symbol]
        Resulting DataFrame will have columns: date, open, high, low, close, ... and a 'tic' column.
        """
        # Reset index to bring dates into a column.
        df = raw_data.copy().reset_index()
        ticker = df.columns[1][1]
        df.columns = df.columns.droplevel(1)
        # import pdb; pdb.set_trace()
        # If the reset index column is not named 'date', rename it.
        if df.columns[0] != "date":
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

        df["tic"] = ticker
        df["day"] = df["date"].dt.dayofweek

        # drop nan rows
        df = df.dropna()

        return df

    def preprocess_data(self, formatted_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess formatted raw data using FinRL's FeatureEngineer.
        Expects formatted_raw to have columns: date, tic, open, high, low, close, etc.
        """
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=False,
            use_turbulence=True,
            user_defined_feature=False
        )

        return fe.preprocess_data(formatted_raw)

    def train_drl_model(self, algorithm: str = "a2c", total_timesteps: int = 50000, env_kwargs: dict = None):
        """
        Train a DRL agent using FinRL.
        """
        # print("Training a DRL agent...")
        df_env = self.train_data.copy()
        stock_dim = len(df_env["tic"].unique())
        state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
        default_env_kwargs = {
            "hmax": 1000,
            "initial_amount": self.params.initial_amount,
            "num_stock_shares": [0] * stock_dim,
            "buy_cost_pct": [0.0049] * stock_dim,
            "sell_cost_pct": [0.0049] * stock_dim,
            "state_space": state_space,
            "stock_dim": stock_dim,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dim,
            "reward_scaling": 1e-4
        }
        if env_kwargs:
            default_env_kwargs.update(env_kwargs)
        e_train = StockTradingEnv(df=df_env, **default_env_kwargs)
        env_train, _ = e_train.get_sb_env()
        agent = DRLAgent(env=env_train)
        model = agent.get_model(algorithm, verbose=0)
        trained_model = agent.train_model(
            model=model,
            tb_log_name=algorithm,
            total_timesteps=total_timesteps)
        self.model = trained_model
        for test_df in self.test_data:
            e_trade = StockTradingEnv(df=test_df, **default_env_kwargs)
            df_account_value, df_actions = agent.DRL_prediction(
                model=trained_model,
                environment=e_trade
            )

            # turn date to datetime.date
            df_account_value["date"] = pd.to_datetime(df_account_value["date"]).dt.date
            df_actions["date"] = pd.to_datetime(df_actions["date"]).dt.date

            self.df_account_value.append(df_account_value)
            self.df_actions.append(df_actions)


    def compute_state(self):
        state = []
        for d, test_df in zip(self.datas, self.test_data):
            today = d.datetime.date(0)
            tic = d._name

            # assert the ticker name
            assert tic == test_df["tic"].values[0]

            # get today's indicator features
            today_features = test_df[test_df['date'] == today]
            # drop columns that are not features
            today_features = today_features.drop(columns=["date", "tic", "day"])

            state.append(today_features)

        return np.array(state)


    def next(self):
        """
        Predict actions using the trained DRL model and execute trades.
        """

        for d, actions in zip(self.datas, self.df_actions):
            today = d.datetime.date(0)

            if today > actions['date'].tolist()[-1]:
                return

            try:
                act = int(actions[actions["date"] == today]["actions"].values[0])
            except Exception as e:
                print(f"No action detected on {today}: {e}")
                return

            price = d.close[0]

            if act > 0:
                # check if we have enough cash to buy
                size = self._adjust_size_for_commission(min(int(self.broker.cash / price), act))
                self.buy(data=d, size=size)
            elif act < 0:
                pos = self.getposition(data=d)
                size = self._adjust_size_for_commission(min(pos.size, -act))
                self.sell(data=d, size=size)
            else:
                pass

        self.post_next_actions()


if __name__ == "__main__":
    trade_config = {
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        "silence": False,
        "setup_name": "selected_5",
    }
    # trade_config = {
    #     "date_from": "2022-10-06",
    #     "date_to": "2023-04-10",
    #     "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
    #     "setup_name": "cherry_pick_both_finmem",
    # }
    operator = BacktestingEngine(trade_config)
    # operator.execute_iter(FinRLStrategy)
    operator.run_rolling_window(FinRLStrategy)

    aggregate_results_one_strategy(trade_config["selection_strategy"], FinRLStrategy.__name__)
