from backtest.data_util.backtest_dataset import BacktestDataset
import pickle
import pandas as pd

class FinMemDataset(BacktestDataset):
    def __init__(self, pickle_file=None, data=None):
        if pickle_file is None and data is None:
            raise ValueError("Either pickle_file or data must be provided")
        if pickle_file is not None and data is not None:
            raise ValueError("Only one of pickle_file or data must be provided")

        self.data = pickle.load(open(pickle_file, "rb")) if pickle_file is not None else data
        self.tickers_list = None
        self.tickers_list = self.get_tickers_list()

    def get_ticker_price_by_date(self, ticker, date):
        if type(date) == str:
            date = pd.to_datetime(date).date()
        return self.data[date]["price"][ticker]

    def get_data_by_date(self, date):
        if type(date) == str:
            date = pd.to_datetime(date).date()
        if date in self.data:
            return self.data[date]
        else:
            return {}

    def get_subset_by_time_range(self, start_date, end_date):
        if type(start_date) == str:
            start_date = pd.to_datetime(start_date).date()
        if type(end_date) == str:
            end_date = pd.to_datetime(end_date).date()
        subset = {date: self.data[date] for date in self.data.keys() if start_date <= date <= end_date}
        return FinMemDataset(data=subset) if len(subset) > 0 else None

    def get_ticker_subset_by_time_range(self, ticker, start_date, end_date):
        if type(start_date) == str:
            start_date = pd.to_datetime(start_date).date()
        if type(end_date) == str:
            end_date = pd.to_datetime(end_date).date()
        data = {}
        for date in self.data.keys():
            if start_date <= date <= end_date and ticker in self.data[date]["price"]:
                data[date] = {"price": {ticker: self.data[date]["price"][ticker]}}
        return FinMemDataset(data=data) if len(data) > 0 else None

    def get_date_range(self):
        return list(self.data.keys())

    def get_tickers_list(self):
        if self.tickers_list is None:
            tickers_list = set()
            for date in self.data.keys():
                tickers_list.update(self.data[date]["price"].keys())
            self.tickers_list = list(tickers_list)

        return self.tickers_list

