import pickle
from typing import Any

import pandas as pd

from backtest.data_util.trading_data import TradingData


class FinsaberDataset(TradingData):
    """TradingData adapter for FINSABER aggregated date dictionaries.

    Expected daily shape is extensible:
    ``{date: {"price": {ticker: bar}, "news": {ticker: [...]}, ...}}``.
    Additional modalities are preserved and filtered by ticker when possible.
    """

    def __init__(
        self,
        pickle_file: str | None = None,
        data: dict | None = None,
        price_field: str = "adjusted_close",
        source_kind: str | None = None,
        filing_payload_kind: str = "raw_filing",
    ):
        if pickle_file is None and data is None:
            raise ValueError("Either pickle_file or data must be provided")
        if pickle_file is not None and data is not None:
            raise ValueError("Only one of pickle_file or data must be provided")

        if pickle_file is not None:
            with open(pickle_file, "rb") as file:
                self.data = pickle.load(file)
        else:
            self.data = data

        self.price_field = price_field
        self.source_kind = source_kind or ("pickle" if pickle_file is not None else "in_memory")
        self.filing_payload_kind = self._validate_filing_payload_kind(filing_payload_kind)
        self._tickers_list = None
        self._date_range = sorted(self.data.keys())

    @staticmethod
    def _normalize_date(date):
        if isinstance(date, str):
            return pd.to_datetime(date).date()
        if isinstance(date, pd.Timestamp):
            return date.date()
        return date

    @staticmethod
    def _validate_filing_payload_kind(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"raw_filing", "section_text"}:
            raise ValueError(
                "Unsupported filing_payload_kind. Expected 'raw_filing' or 'section_text'."
            )
        return normalized

    def get_ticker_price_by_date(self, ticker: str, date, price_field: str | None = None) -> float:
        date = self._normalize_date(date)
        price = self.data[date]["price"][ticker]
        if isinstance(price, dict):
            field = price_field or self.price_field
            if field in price:
                return price[field]
            if field.startswith("adjusted_") and "adjusted_close" in price and "close" in price:
                if price["close"] == 0:
                    return 0
                adjustment = price["adjusted_close"] / price["close"]
                raw_field = field.removeprefix("adjusted_")
                if raw_field in price:
                    return price[raw_field] * adjustment
            return price.get("close", price.get("adjusted_close"))
        return price

    def get_ticker_data_by_date(self, ticker: str, date) -> dict[str, Any]:
        daily_data = self.get_data_by_date(date)
        ticker_data = {}
        for modality, values in daily_data.items():
            if isinstance(values, dict) and ticker in values:
                ticker_data[modality] = values[ticker]
        return ticker_data

    def get_data_by_date(self, date) -> dict[str, Any]:
        date = self._normalize_date(date)
        return self.data.get(date, {})

    def get_subset_by_time_range(self, start_date, end_date):
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        subset = {
            date: self.data[date]
            for date in self._date_range
            if start_date <= date <= end_date
        }
        return (
            FinsaberDataset(
                data=subset,
                price_field=self.price_field,
                source_kind=self.source_kind,
                filing_payload_kind=self.filing_payload_kind,
            )
            if subset
            else None
        )

    def get_ticker_subset_by_time_range(self, ticker: str, start_date, end_date):
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        subset = {}
        for date in self._date_range:
            if not start_date <= date <= end_date:
                continue
            daily_ticker_data = {}
            for modality, values in self.data[date].items():
                if isinstance(values, dict) and ticker in values:
                    daily_ticker_data[modality] = {ticker: values[ticker]}
            if "price" in daily_ticker_data:
                subset[date] = daily_ticker_data
        return (
            FinsaberDataset(
                data=subset,
                price_field=self.price_field,
                source_kind=self.source_kind,
                filing_payload_kind=self.filing_payload_kind,
            )
            if subset
            else None
        )

    def get_date_range(self) -> list:
        return list(self._date_range)

    def get_tickers_list(self) -> list[str]:
        if self._tickers_list is None:
            tickers = set()
            for date in self._date_range:
                tickers.update(self.data[date].get("price", {}).keys())
            self._tickers_list = sorted(tickers)
        return self._tickers_list

    def get_price_dataframe(self, tickers=None, date_from=None, date_to=None, adjust: bool = True) -> pd.DataFrame:
        if tickers is None or tickers == "all":
            tickers = set(self.get_tickers_list())
        elif isinstance(tickers, str):
            tickers = {tickers}
        else:
            tickers = set(tickers)
        start_date = self._normalize_date(date_from) if date_from is not None else None
        end_date = self._normalize_date(date_to) if date_to is not None else None

        records = []
        for date in self._date_range:
            if start_date is not None and date < start_date:
                continue
            if end_date is not None and date > end_date:
                continue
            for symbol, price in self.data[date].get("price", {}).items():
                if symbol not in tickers:
                    continue
                if isinstance(price, dict):
                    record = {
                        "date": pd.to_datetime(date),
                        "symbol": symbol,
                        "volume": price.get("volume", 0),
                    }
                    if adjust and "adjusted_close" in price and "close" in price:
                        adjustment = 0 if price["close"] == 0 else price["adjusted_close"] / price["close"]
                        record.update({
                            "open": price.get("adjusted_open", price.get("open", 0) * adjustment),
                            "high": price.get("adjusted_high", price.get("high", 0) * adjustment),
                            "low": price.get("adjusted_low", price.get("low", 0) * adjustment),
                            "close": price["adjusted_close"],
                        })
                    else:
                        record.update({
                            "open": price.get("open", price.get("close", price.get("adjusted_close"))),
                            "high": price.get("high", price.get("close", price.get("adjusted_close"))),
                            "low": price.get("low", price.get("close", price.get("adjusted_close"))),
                            "close": price.get("close", price.get("adjusted_close")),
                        })
                else:
                    record = {
                        "date": pd.to_datetime(date),
                        "symbol": symbol,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 0,
                    }
                records.append(record)

        return pd.DataFrame.from_records(records)
