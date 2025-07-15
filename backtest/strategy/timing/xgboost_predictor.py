import xgboost as xgb
import pandas as pd

from backtest.strategy.timing.base_strategy import BaseStrategy
from backtest.finsaber_bt import FINSABERBt


class XGBoostPredictorStrategy(BaseStrategy):
    params = (
        ("model_params", None),
        ("train_period", 252 * 3),  # 3 years of daily data for training
        ("total_days", 0),
    )

    def __init__(self, train_data=None, strat_params=None):
        super().__init__()
        self.model = {}
        self.train_data = train_data
        self.column_order = None
        self.train_model()

    def train_model(self):
        assert self.train_data is not None, "Train data is not set"
        symbols = self.train_data.columns.get_level_values(1).unique()
        # Prepare training data
        for sym in symbols:
            df = self.train_data.xs(sym, axis=1, level=1, drop_level=True)
            df['return'] = df['close'].pct_change().shift(-1)
            df.dropna(inplace=True)
            X = df.drop(columns=['return'])
            y = (df['return'] > 0).astype(int)
            self.column_order = list(X.columns)
            dmatrix = xgb.DMatrix(X, label=y)
            # import ipdb; ipdb.set_trace()
            self.model[sym] = xgb.train(self.params.model_params, dmatrix)

    def next(self):
        for d in self.datas:
            try:
                features = d._dataname[self.column_order]
            except KeyError:
                # If the features are not available, skip this data
                continue

            today_date = pd.to_datetime(d.datetime.date(0))
            # get the exact line 0of features
            features = features.loc[today_date]
            # import ipdb; ipdb.set_trace()
            dmatrix = xgb.DMatrix([features], feature_names=self.column_order)
            prediction = self.model[d._name].predict(dmatrix)
            if prediction >= 0.5:
                if not self.position:
                    self.buy(size=self._adjust_size_for_commission(int(self.broker.cash / d.close[0])))
            else:
                if self.position:
                    self.sell(size=self.position.size)
        self.post_next_actions()
        return


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
    # trade_config = {
    #     "tickers": "all",
    #     "silence": True,
    #     "setup_name": "random:50",
    # }
    operator = FINSABERBt(trade_config)
    operator.execute_iter(XGBoostPredictorStrategy)
    # operator.run_rolling_window(XGBoostPredictorStrategy)