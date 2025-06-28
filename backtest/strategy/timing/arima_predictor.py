import pandas as pd
import statsmodels.api as sm
from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.finsaber_bt import FINSABERBt
from backtest.toolkit.operation_utils import aggregate_results_one_strategy


class ARIMAPredictorStrategy(BaseStrategy):
    params = (
        ("train_period", 252 * 2),  # Train on the past 3 years of daily data
        ("order", (5, 1, 0)),  # Default ARIMA order (p, d, q)
        ("total_days", 0),
    )

    def __init__(self, train_data=None, strat_params=None):
        super().__init__()
        self.models = {}  # Store trained ARIMA models per symbol
        self.train_data = train_data
        self.train_models()

    def train_models(self):
        """Train ARIMA models for each symbol using past 3 years of data."""
        assert self.train_data is not None, "Train data is not set"
        symbols = self.train_data.columns.get_level_values(1).unique()

        for sym in symbols:
            df = self.train_data.xs(sym, axis=1, level=1, drop_level=True)
            df = df['close'].dropna()

            # import pdb; pdb.set_trace()

            if len(df) < 252:
                continue  # Skip if not enough data

            train_series = df.iloc[-self.params.train_period:]  # Use only the last 3 years
            model = sm.tsa.ARIMA(train_series, order=self.params.order)
            self.models[sym] = model.fit()

    def next(self):
        """Predict next day's price and decide buy/sell actions."""
        for d in self.datas:
            today_date = pd.to_datetime(d.datetime.date(0))
            sym = d._name


            if sym not in self.models:
                continue  # Skip if model is missing

            # add today's price to the training data, we only care about the close price in ARIMA
            self.train_data.loc[today_date, (slice(None), sym)] = d.close[0]
            df = self.train_data.xs(sym, axis=1, level=1, drop_level=True)['close']
            df = df.dropna()


            # Update model with most recent price before making a prediction
            self.models[sym] = self.models[sym].apply(df)

            # Predict next day's price
            forecast = self.models[sym].forecast(steps=1).values[0]
            current_price = df.loc[today_date]

            # Trading logic based on price prediction
            if forecast > current_price:
                if not self.position:
                    self.buy(size=self._adjust_size_for_commission(int(self.broker.cash / d.close[0])))
            elif forecast == current_price:
                pass
            else:
                if self.position:
                    self.sell(size=self.position.size)

        self.post_next_actions()


if __name__ == "__main__":
    # trade_config = {
    #     "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
    #     "silence": False,
    #     "setup_name": "selected_5",
    # }
    trade_config = {
        "date_from": "2022-10-06",
        "date_to": "2023-04-10",
        "tickers": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"],
        "setup_name": "cherry_pick_both_finmem",
    }
    operator = FINSABERBt(trade_config)
    # operator.run_rolling_window(ARIMAPredictorStrategy)
    operator.run_iterative_tickers(ARIMAPredictorStrategy)
    # aggregate_results_one_strategy(trade_config["selection_strategy"], ARIMAPredictorStrategy.__name__)
