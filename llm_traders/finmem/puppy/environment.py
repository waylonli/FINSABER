import os
import shutil
import pickle
from datetime import date
from typing import List, Dict, Tuple, Union, Any
from pydantic import BaseModel, ValidationError

# type alias
market_info_type = Tuple[
    date,  # cur date
    float,  # cur price
    Union[str, None],  # cur filing_k
    Union[str, None],  # cur filing_q
    List[str],  # cur news
    float,  # cur record
    bool,  # termination flag
]
terminated_market_info_type = Tuple[None, None, None, None, None, None, bool]


# env data structure validation
class OneDateRecord(BaseModel):
    price: Dict[str, float]
    filing_k: Dict[str, str]
    filing_q: Dict[str, str]
    news: Dict[str, List[str]]


class MarketEnvironment:
    def __init__(
        self,
        env_data_pkl: Dict[date, Dict[str, Any]],
        start_date: date,
        end_date: date,
        symbol: str,
    ) -> None:
        # validate structure
        first_date = list(env_data_pkl.keys())[0]
        if not isinstance(first_date, date):
            raise TypeError("env_data_pkl keys must be date type")
        try:
            OneDateRecord.model_validate(env_data_pkl[first_date])
        except ValidationError as e:
            raise e
        self.date_series = env_data_pkl.keys()
        if (start_date not in self.date_series):
            # raise ValueError("start_date and end_date must be in env_data_pkl keys")
            # print("start_date not in env_data_pkl keys, adjusting start_date to next nearest available date")
            start_date = min(self.date_series, key=lambda x: abs(x - start_date))
            # print(f"Adjusted start_date: {start_date}")
        if (end_date not in self.date_series):
            # print("end_date not in env_data_pkl keys, adjusting end_date to previous nearest available date")
            end_date = min(self.date_series, key=lambda x: abs(x - end_date))
            # print(f"Adjusted end_date: {end_date}")
        self.date_series = [
            i for i in self.date_series if (i >= start_date) and (i <= end_date)
        ]

        self.date_series = sorted(self.date_series)
        self.date_series_keep = self.date_series.copy()
        self.simulation_length = len(self.date_series) - 1
        self.start_date = start_date
        self.end_date = end_date
        self.cur_date = None
        self.env_data = env_data_pkl
        self.symbol = symbol
        self.last_date = None

    def reset(self) -> None:
        self.date_series = [
            i
            for i in self.date_series_keep
            if (i >= self.start_date) and (i <= self.end_date)
        ]
        self.date_series = sorted(self.date_series)
        self.cur_date = None

    def step(self) -> Union[market_info_type, terminated_market_info_type]:
        try:
            self.cur_date = self.date_series.pop(0)  # type: ignore
            future_date = self.date_series[0]  # type: ignore
        except IndexError:
            return None, None, None, None, None, None, True

        cur_date = self.cur_date
        cur_price = self.env_data[self.cur_date]["price"]
        future_price = self.env_data[future_date]["price"]
        cur_filing_k = self.env_data[self.cur_date]["filing_k"]
        cur_filing_q = self.env_data[self.cur_date]["filing_q"]

        if len(cur_price) == 0:
            print(cur_price)
            print(f"cur_date: {cur_date}")
            import ipdb; ipdb.set_trace()
            # get the previous available price
            cur_price = self.env_data[self.last_date]["price"]

        if self.env_data[self.cur_date]["news"] != {}:
            cur_news = self.env_data[self.cur_date]["news"]
        else:
            cur_news = {self.symbol: ''}

        cur_record = {symbol: future_price[symbol] - cur_price[symbol] for symbol in cur_price if symbol in future_price}

        # handle none filing case
        if len(cur_filing_k) == 0 or self.symbol not in cur_filing_k:
            cur_filing_k = None
        else:
            cur_filing_k = cur_filing_k[self.symbol]
        if len(cur_filing_q) == 0 or self.symbol not in cur_filing_q:
            cur_filing_q = None
        else:
            cur_filing_q = cur_filing_q[self.symbol]
        # import ipdb; ipdb.set_trace()
        self.last_date = cur_date

        if self.symbol not in cur_price:
            return None, None, None, None, None, None, True

        return (
            cur_date,
            # yesterday's price if current price is none
            cur_price[self.symbol],
            cur_filing_k,
            cur_filing_q,
            cur_news[self.symbol] if self.symbol in cur_news else None,
            cur_record[self.symbol] if self.symbol in cur_record else None,
            False,
        )

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        path = os.path.join(path, "env")
        if os.path.exists(path):
            if force:
                shutil.rmtree(path)
            else:
                raise FileExistsError(f"Path {path} already exists")
        os.mkdir(path)
        with open(os.path.join(path, "env.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MarketEnvironment":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exists")
        with open(os.path.join(path, "env.pkl"), "rb") as f:
            env = pickle.load(f)
        # update
        env.simulation_length = len(env.date_series)
        return env
