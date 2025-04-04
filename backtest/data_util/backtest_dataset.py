class BacktestDataset:
    def __init__(self):
        pass

    def get_ticker_price_by_date(self, ticker, date):
        raise NotImplementedError

    def get_ticker_data_by_date(self, ticker, date):
        raise NotImplementedError

    def get_ticker_list(self):
        raise NotImplementedError

    def get_data_by_time_range(self, start_date, end_date):
        raise NotImplementedError

    def get_ticker_data_by_time_range(self, ticker, start_date, end_date):
        raise NotImplementedError

    def get_date_range(self):
        raise NotImplementedError