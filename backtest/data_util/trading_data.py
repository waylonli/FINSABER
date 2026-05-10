from abc import ABC, abstractmethod
from typing import Any


class TradingData(ABC):
    """Minimal interface for pluggable market datasets.

    Implementations may store additional modalities such as news, filings,
    earnings calls, or transcripts. Backtest engines should only depend on this
    interface, not on a specific storage format.
    """

    @abstractmethod
    def get_data_by_date(self, date) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_ticker_price_by_date(self, ticker: str, date, price_field: str | None = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_ticker_data_by_date(self, ticker: str, date) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_tickers_list(self) -> list[str]:
        raise NotImplementedError

    def get_ticker_list(self) -> list[str]:
        return self.get_tickers_list()

    @abstractmethod
    def get_subset_by_time_range(self, start_date, end_date):
        raise NotImplementedError

    def get_data_by_time_range(self, start_date, end_date):
        return self.get_subset_by_time_range(start_date, end_date)

    @abstractmethod
    def get_ticker_subset_by_time_range(self, ticker: str, start_date, end_date):
        raise NotImplementedError

    def get_ticker_data_by_time_range(self, ticker: str, start_date, end_date):
        return self.get_ticker_subset_by_time_range(ticker, start_date, end_date)

    @abstractmethod
    def get_date_range(self) -> list:
        raise NotImplementedError

    def get_modalities(self) -> list[str]:
        dates = self.get_date_range()
        if not dates:
            return []
        return list(self.get_data_by_date(dates[0]).keys())
