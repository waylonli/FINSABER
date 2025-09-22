"""Market microstructure data accessors used by the execution layer."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest.toolkit.operation_utils import load_price_dataset, get_tickers_price

LOGGER = logging.getLogger(__name__)

DEFAULT_SP500_PATH = os.path.join("data", "extra", "sp500_historical_2000_2024.csv")
DEFAULT_MARKET_CAP_TEMPLATE = os.path.join("data", "extra", "{ticker}-market-cap.csv")


@dataclass(slots=True)
class MarketSnapshot:
    adv: float = 1.0
    vix: float = 0.0
    idiosyncratic_vol: float = 0.0
    log_market_cap: float = 2.0
    beta: float = 0.0
    sp500_return: float = 0.0


class MarketDataProvider:
    """Caches market microstructure data needed by execution cost models."""

    def __init__(
        self,
        adv_window: int = 252,
        sp500_file_path: str = DEFAULT_SP500_PATH,
        market_cap_template: str = DEFAULT_MARKET_CAP_TEMPLATE,
    ) -> None:
        self.adv_window = adv_window
        self.sp500_file_path = sp500_file_path
        self.market_cap_template = market_cap_template

        self._adv_cache: Dict[str, pd.Series] = {}
        self._idiosync_cache: Dict[str, pd.Series] = {}
        self._market_cap_cache: Dict[str, pd.Series] = {}
        self._beta_cache: Dict[str, pd.Series] = {}

        self._vix_series: Optional[pd.Series] = None
        self._sp500_return_series: Optional[pd.Series] = None

    def snapshot(self, ticker: str, date: pd.Timestamp) -> MarketSnapshot:
        ts = pd.Timestamp(date)
        return MarketSnapshot(
            adv=float(self._value_at(self._adv_lookup(ticker), ts, default=1.0)),
            vix=float(self._value_at(self._vix_lookup(), ts, default=0.0)),
            idiosyncratic_vol=float(self._value_at(self._idiosync_lookup(ticker), ts, default=0.0)),
            log_market_cap=float(self._value_at(self._market_cap_lookup(ticker), ts, default=np.nan)),
            beta=float(self._value_at(self._beta_lookup(ticker), ts, default=0.0)),
            sp500_return=float(self._value_at(self._sp500_returns_lookup(), ts, default=0.0)),
        )

    # ------------------------------------------------------------------
    # Lookup builders
    # ------------------------------------------------------------------

    def _adv_lookup(self, ticker: str) -> pd.Series:
        if ticker not in self._adv_cache:
            self._adv_cache[ticker] = self._build_adv_series(ticker)
        return self._adv_cache[ticker]

    def _vix_lookup(self) -> pd.Series:
        if self._vix_series is None:
            self._vix_series = self._build_vix_series()
        return self._vix_series

    def _idiosync_lookup(self, ticker: str) -> pd.Series:
        if ticker not in self._idiosync_cache:
            self._idiosync_cache[ticker] = self._build_idiosync_series(ticker)
        return self._idiosync_cache[ticker]

    def _market_cap_lookup(self, ticker: str) -> pd.Series:
        if ticker not in self._market_cap_cache:
            self._market_cap_cache[ticker] = self._build_market_cap_series(ticker)
        return self._market_cap_cache[ticker]

    def _beta_lookup(self, ticker: str) -> pd.Series:
        if ticker not in self._beta_cache:
            self._beta_cache[ticker] = self._build_beta_series(ticker)
        return self._beta_cache[ticker]

    def _sp500_returns_lookup(self) -> pd.Series:
        if self._sp500_return_series is None:
            self._sp500_return_series = self._build_sp500_return_series()
        return self._sp500_return_series

    # ------------------------------------------------------------------
    # Builders (per series)
    # ------------------------------------------------------------------

    def _build_adv_series(self, ticker: str) -> pd.Series:
        df = load_price_dataset(symbols=[ticker], adjust=False)
        if df.empty:
            LOGGER.warning("No price data available for %s. Using unit ADV.", ticker)
            return pd.Series(dtype=float)

        df = df.sort_values("date")
        df["dollar_volume"] = df["close"].astype(float) * df["volume"].astype(float)

        adv = (
            df["dollar_volume"]
            .rolling(window=self.adv_window, min_periods=self.adv_window // 4)
            .mean()
            .shift(1)
        )

        adv_series = adv.set_axis(pd.to_datetime(df["date"].values))
        adv_series = adv_series.reindex(_calendar_index(df["date"]))
        adv_series = adv_series.ffill().fillna(1.0)
        return adv_series

    def _build_vix_series(self) -> pd.Series:
        df = _load_sp500_dataframe(self.sp500_file_path)
        if df.empty:
            LOGGER.warning("No SP500 data found at %s. Using zero VIX.", self.sp500_file_path)
            return pd.Series(dtype=float)

        df["daily_return"] = df["Close"].pct_change()
        df = df.dropna(subset=["daily_return"]).reset_index(drop=True)
        if df.empty:
            return pd.Series(dtype=float)

        df["YearMonth"] = df["Date"].dt.to_period("M")
        monthly = df.groupby("YearMonth").agg(
            daily_var=("daily_return", "var"),
            count=("daily_return", "count"),
        )
        monthly = monthly[monthly["count"] >= 10]
        monthly["vix"] = np.sqrt(monthly["daily_var"] * 252.0) * 100.0

        month_index = _calendar_index(monthly.index.to_timestamp())
        monthly_series = monthly["vix"].set_axis(monthly.index.to_timestamp())
        monthly_series = monthly_series.reindex(month_index).ffill().fillna(0.0)
        daily_index = _calendar_index(df["Date"])
        return monthly_series.reindex(daily_index, method="ffill").fillna(0.0)

    def _build_idiosync_series(self, ticker: str) -> pd.Series:
        stock_df = load_price_dataset(symbols=[ticker], adjust=True)
        if stock_df.empty:
            LOGGER.warning("No stock data available for %s when computing idiosyncratic volatility.", ticker)
            return pd.Series(dtype=float)
        stock_df = stock_df.sort_values("date")
        stock_df["return"] = stock_df["close"].pct_change()

        sp500_df = _load_sp500_dataframe(self.sp500_file_path)
        sp500_df["return"] = sp500_df["Close"].pct_change()

        merged = pd.merge(
            stock_df[["date", "return"]],
            sp500_df[["Date", "return"]],
            left_on="date",
            right_on="Date",
            how="inner",
            suffixes=("_stock", "_market"),
        ).dropna()

        if len(merged) < self.adv_window:
            LOGGER.warning("Insufficient overlapping data for idiosyncratic volatility of %s.", ticker)
            return pd.Series(dtype=float)

        residuals = []
        dates = []
        window = self.adv_window
        sqrt_252 = np.sqrt(252.0) * 100.0

        for idx in range(window, len(merged)):
            window_slice = merged.iloc[idx - window:idx]
            stock_returns = window_slice["return_stock"].to_numpy()
            market_returns = window_slice["return_market"].to_numpy()

            if len(stock_returns) < 30 or np.allclose(market_returns.var(), 0.0):
                continue

            alpha, beta = np.polyfit(market_returns, stock_returns, deg=1)
            predicted = alpha + beta * market_returns
            resid = stock_returns - predicted
            residual_std = float(np.std(resid, ddof=1))
            vol = residual_std * sqrt_252

            residuals.append(vol)
            dates.append(window_slice.iloc[-1]["date"])

        if not residuals:
            return pd.Series(dtype=float)

        series = pd.Series(residuals, index=pd.to_datetime(dates))
        series = series.reindex(_calendar_index(dates)).ffill().fillna(0.0)
        return series

    def _build_market_cap_series(self, ticker: str) -> pd.Series:
        path = self.market_cap_template.format(ticker=ticker.lower())
        if not os.path.exists(path):
            LOGGER.warning("Market cap file not found for %s: %s", ticker, path)
            return pd.Series(dtype=float)

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
        df = df.dropna(subset=["Date", "Market Cap"]).sort_values("Date")
        if df.empty:
            return pd.Series(dtype=float)

        df["log_market_cap"] = np.log1p(df["Market Cap"].astype(float) / 1e9)
        series = pd.Series(df["log_market_cap"].values, index=df["Date"])
        return series.reindex(_calendar_index(df["Date"])).ffill().fillna(np.nan)

    def _build_beta_series(self, ticker: str) -> pd.Series:
        sp500_df = _load_sp500_dataframe(self.sp500_file_path)
        sp500_df["return"] = np.log(sp500_df["Close"] / sp500_df["Close"].shift(1))
        sp500_df = sp500_df.dropna(subset=["return"]).reset_index(drop=True)

        stock_df = get_tickers_price(
            ticker,
            date_from="2000-01-01",
            date_to="2025-01-01",
            return_original=True,
        )
        if stock_df is None or stock_df.empty:
            LOGGER.warning("No stock data available for beta calculation for %s.", ticker)
            return pd.Series(dtype=float)

        stock_df = stock_df[stock_df["symbol"] == ticker].copy()
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        stock_df = stock_df.sort_values("date")
        stock_df["return"] = np.log(stock_df["close"] / stock_df["close"].shift(1))
        stock_df = stock_df.dropna(subset=["return"])

        merged = pd.merge(
            stock_df[["date", "return"]],
            sp500_df[["Date", "return"]],
            left_on="date",
            right_on="Date",
            how="inner",
            suffixes=("_stock", "_market"),
        )
        if len(merged) < self.adv_window:
            LOGGER.warning("Insufficient overlapping data for beta calculation of %s.", ticker)
            return pd.Series(dtype=float)

        betas = []
        dates = []
        window = self.adv_window
        for idx in range(window, len(merged)):
            window_slice = merged.iloc[idx - window:idx]
            market_returns = window_slice["return_market"].to_numpy()
            stock_returns = window_slice["return_stock"].to_numpy()

            variance = market_returns.var(ddof=1)
            if variance == 0:
                continue
            covariance = np.cov(stock_returns, market_returns, ddof=1)[0, 1]
            beta = float(covariance / variance)

            betas.append(beta)
            dates.append(window_slice.iloc[-1]["date"])

        if not betas:
            return pd.Series(dtype=float)

        series = pd.Series(betas, index=pd.to_datetime(dates))
        return series.reindex(_calendar_index(dates)).ffill().fillna(0.0)

    def _build_sp500_return_series(self) -> pd.Series:
        df = _load_sp500_dataframe(self.sp500_file_path)
        if df.empty:
            LOGGER.warning("No SP500 data located at %s for return series.", self.sp500_file_path)
            return pd.Series(dtype=float)

        df["return"] = np.log(df["Close"] / df["Close"].shift(1))
        df = df.dropna(subset=["return"])
        series = pd.Series(df["return"].values, index=df["Date"])
        return series.reindex(_calendar_index(df["Date"])).ffill().fillna(0.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _value_at(series: pd.Series, date: pd.Timestamp, default: float) -> float:
        if series.empty:
            return default
        value = series.get(date)
        if pd.isna(value):
            value = series.asof(date)
        if pd.isna(value):
            return default
        return value


def _calendar_index(dates) -> pd.DatetimeIndex:
    dates = pd.to_datetime(pd.Series(list(dates)))
    if dates.empty:
        return pd.DatetimeIndex([])
    return pd.date_range(start=dates.min(), end=dates.max(), freq="D")


@lru_cache(maxsize=1)
def _load_sp500_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        LOGGER.error("SP500 data file not found: %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return df
