from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from backtest.data_util.trading_data import TradingData


class FinsaberParquetDataset(TradingData):
    """TradingData adapter for the FINSABER-2 partitioned parquet dataset."""

    def __init__(
        self,
        root: str | Path,
        start_date=None,
        end_date=None,
        tickers: Iterable[str] | None = None,
        modalities: Iterable[str] = ("price", "news", "filing_k", "filing_q"),
        price_field: str = "adjusted_close",
        filing_merge_policy: str = "concat",
    ):
        self.root = Path(root)
        self.start_date = self._normalize_date(start_date)
        self.end_date = self._normalize_date(end_date)
        if isinstance(tickers, str):
            self.tickers = None if tickers == "all" else [tickers]
        else:
            self.tickers = sorted(set(tickers)) if tickers is not None else None
        self.modalities = tuple(modalities)
        self.price_field = price_field
        self.source_kind = "parquet"
        self.filing_payload_kind = "raw_filing"
        self.filing_merge_policy = self._validate_filing_merge_policy(filing_merge_policy)
        self._data_cache = None
        self._date_range_cache = None
        self._tickers_cache = None

    @staticmethod
    def _normalize_date(date):
        if date is None:
            return None
        if isinstance(date, str):
            return pd.to_datetime(date).date()
        if isinstance(date, pd.Timestamp):
            return date.date()
        return date

    @staticmethod
    def _validate_filing_merge_policy(policy: str) -> str:
        normalized = str(policy).strip().lower()
        if normalized not in {"concat", "latest"}:
            raise ValueError(
                "Unsupported filing_merge_policy. Expected 'concat' or 'latest'."
            )
        return normalized

    def _merge_filing_text(self, current_text: str | None, new_text: str) -> str:
        # Preserve the current default behavior, while allowing callers to keep
        # only the latest same-day filing for a ticker when duplicate filings
        # should not be concatenated into one payload.
        if not current_text or self.filing_merge_policy == "latest":
            return new_text
        return f"{current_text}\n\n{new_text}"

    def _date_filter(self):
        filters = []
        if self.start_date is not None:
            filters.append(ds.field("date") >= self.start_date)
        if self.end_date is not None:
            filters.append(ds.field("date") <= self.end_date)
        if self.tickers is not None:
            filters.append(ds.field("symbol").isin(self.tickers))
        if not filters:
            return None
        expr = filters[0]
        for item in filters[1:]:
            expr = expr & item
        return expr

    def _read_price(self) -> pd.DataFrame:
        columns = ["date", "symbol", "cik", "open", "high", "low", "close", "adjusted_close", "volume", "year"]
        dataset = ds.dataset(str(self.root / "price_daily"), format="parquet", partitioning="hive")
        df = dataset.to_table(columns=columns, filter=self._date_filter()).to_pandas()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"]).dt.date
        adjustment = df["adjusted_close"] / df["close"]
        adjustment = adjustment.where((df["close"] != 0) & adjustment.notna(), 0)
        for field in ["open", "high", "low"]:
            df[f"adjusted_{field}"] = df[field] * adjustment
        return df.sort_values(["date", "symbol"])

    def _read_news(self) -> pd.DataFrame:
        if "news" not in self.modalities:
            return pd.DataFrame()
        columns = ["date", "symbol", "cik", "item_index", "news_text", "text_len", "text_crc32", "year"]
        dataset = ds.dataset(str(self.root / "news_items"), format="parquet", partitioning="hive")
        df = dataset.to_table(columns=columns, filter=self._date_filter()).to_pandas()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values(["date", "symbol", "item_index"])

    def _iter_filing_files(self, folder: str):
        for file in sorted((self.root / folder).glob("year=*/part-000.parquet")):
            year = int(file.parent.name.split("=")[-1])
            if self.start_date is not None and year < self.start_date.year:
                continue
            if self.end_date is not None and year > self.end_date.year:
                continue
            yield file

    def _read_filings(self, folder: str) -> pd.DataFrame:
        frames = []
        for file in self._iter_filing_files(folder):
            parquet_file = pq.ParquetFile(file)
            df = parquet_file.read().to_pandas()
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"]).dt.date
            if self.start_date is not None:
                df = df[df["date"] >= self.start_date]
            if self.end_date is not None:
                df = df[df["date"] <= self.end_date]
            if self.tickers is not None:
                df = df[df["symbol"].isin(self.tickers)]
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol", "filing_idx"])

    def _load_data(self) -> dict:
        if self._data_cache is not None:
            return self._data_cache

        data = defaultdict(lambda: defaultdict(dict))
        price = self._read_price()
        for row in price.itertuples(index=False):
            data[row.date]["price"][row.symbol] = {
                "cik": row.cik,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "adjusted_open": row.adjusted_open,
                "adjusted_high": row.adjusted_high,
                "adjusted_low": row.adjusted_low,
                "adjusted_close": row.adjusted_close,
                "volume": row.volume,
            }

        news = self._read_news()
        for row in news.itertuples(index=False):
            data[row.date]["news"].setdefault(row.symbol, []).append(row.news_text)

        filing_specs = (("filing_k", "filingk"), ("filing_q", "filingq"))
        for modality, folder in filing_specs:
            if modality not in self.modalities:
                continue
            filings = self._read_filings(folder)
            for row in filings.itertuples(index=False):
                current = data[row.date][modality].get(row.symbol)
                data[row.date][modality][row.symbol] = self._merge_filing_text(
                    current,
                    row.filing_text,
                )

        self._data_cache = {date: dict(values) for date, values in data.items()}
        return self._data_cache

    def get_data_by_date(self, date) -> dict[str, Any]:
        date = self._normalize_date(date)
        return self._load_data().get(date, {})

    def get_ticker_price_by_date(self, ticker: str, date, price_field: str | None = None) -> float:
        date = self._normalize_date(date)
        price = self._load_data()[date]["price"][ticker]
        return price[price_field or self.price_field]

    def get_ticker_data_by_date(self, ticker: str, date) -> dict[str, Any]:
        daily_data = self.get_data_by_date(date)
        return {
            modality: values[ticker]
            for modality, values in daily_data.items()
            if isinstance(values, dict) and ticker in values
        }

    def get_tickers_list(self) -> list[str]:
        if self._tickers_cache is not None:
            return self._tickers_cache
        if self.tickers is not None:
            self._tickers_cache = list(self.tickers)
            return self._tickers_cache
        table = ds.dataset(str(self.root / "price_daily"), format="parquet", partitioning="hive").to_table(
            columns=["symbol"],
            filter=self._date_filter(),
        )
        self._tickers_cache = sorted(pc.unique(table["symbol"]).to_pylist())
        return self._tickers_cache

    def get_subset_by_time_range(self, start_date, end_date):
        subset = FinsaberParquetDataset(
            root=self.root,
            start_date=start_date,
            end_date=end_date,
            tickers=self.tickers,
            modalities=self.modalities,
            price_field=self.price_field,
            filing_merge_policy=self.filing_merge_policy,
        )
        return subset if subset.get_date_range() else None

    def get_ticker_subset_by_time_range(self, ticker: str, start_date, end_date):
        subset = FinsaberParquetDataset(
            root=self.root,
            start_date=start_date,
            end_date=end_date,
            tickers=[ticker],
            modalities=self.modalities,
            price_field=self.price_field,
            filing_merge_policy=self.filing_merge_policy,
        )
        return subset if subset.get_date_range() else None

    def get_date_range(self) -> list:
        if self._date_range_cache is not None:
            return self._date_range_cache
        price = self._read_price()
        self._date_range_cache = sorted(price["date"].unique().tolist()) if not price.empty else []
        return self._date_range_cache

    def get_price_dataframe(self, tickers=None, date_from=None, date_to=None, adjust: bool = True) -> pd.DataFrame:
        if tickers is None:
            tickers = self.tickers
        subset = FinsaberParquetDataset(
            root=self.root,
            start_date=date_from if date_from is not None else self.start_date,
            end_date=date_to if date_to is not None else self.end_date,
            tickers=None if tickers == "all" else tickers,
            modalities=("price",),
            price_field=self.price_field,
            filing_merge_policy=self.filing_merge_policy,
        )
        df = subset._read_price()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        if adjust:
            df = df.rename(
                columns={
                    "open": "raw_open",
                    "high": "raw_high",
                    "low": "raw_low",
                    "close": "raw_close",
                    "adjusted_open": "open",
                    "adjusted_high": "high",
                    "adjusted_low": "low",
                    "adjusted_close": "close",
                }
            )
        return df[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
