from __future__ import annotations

from typing import Iterable

import pandas as pd

from backtest.data_util import resolve_trading_data, trading_data_to_env_dict
from llm_traders.finagent.registry import DATASET


@DATASET.register_module(force=True)
class FinsaberTradingDataDataset:
    """FinAgent-compatible dataset backed by a FINSABER ``TradingData`` loader."""

    def __init__(
        self,
        asset: Iterable[str] | str,
        data_loader=None,
        market_data_root=None,
        market_data_info_path=None,
        workdir=None,
        tag=None,
    ):
        self.asset = [asset] if isinstance(asset, str) else list(asset)
        self.workdir = workdir
        self.tag = tag
        self.data_loader = resolve_trading_data(
            data_loader=data_loader,
            market_data_root=market_data_root,
            market_data_info_path=market_data_info_path,
            tickers=self.asset,
        )
        self.raw_data = trading_data_to_env_dict(self.data_loader, tickers=self.asset)
        if not self.raw_data:
            raise ValueError("No FINSABER data available for FinAgent dataset adapter.")

        self.prices = {symbol: self._build_price_frame(symbol) for symbol in self.asset}
        self.news = {symbol: self._build_news_frame(symbol) for symbol in self.asset}
        self.guidances = None
        self.sentiments = None
        self.economics = None

    def _build_price_frame(self, symbol: str) -> pd.DataFrame:
        records = []
        for date, day in sorted(self.raw_data.items()):
            bar = day.get("price", {}).get(symbol)
            if bar is None:
                continue
            if isinstance(bar, dict):
                close = bar.get("adjusted_close", bar.get("close"))
                records.append(
                    {
                        "timestamp": date,
                        "open": bar.get("adjusted_open", bar.get("open", close)),
                        "high": bar.get("adjusted_high", bar.get("high", close)),
                        "low": bar.get("adjusted_low", bar.get("low", close)),
                        "close": close,
                        "adj_close": close,
                        "volume": bar.get("volume", 0),
                    }
                )
            else:
                records.append(
                    {
                        "timestamp": date,
                        "open": bar,
                        "high": bar,
                        "low": bar,
                        "close": bar,
                        "adj_close": bar,
                        "volume": 0,
                    }
                )

        return pd.DataFrame.from_records(
            records,
            columns=["timestamp", "open", "high", "low", "close", "adj_close", "volume"],
        ).sort_values("timestamp")

    def _build_news_frame(self, symbol: str) -> pd.DataFrame:
        records = []
        item_id = 0
        for date, day in sorted(self.raw_data.items()):
            news_items = day.get("news", {}).get(symbol, [])
            if isinstance(news_items, str):
                news_items = [news_items]
            for text in news_items:
                item_id += 1
                records.append(
                    {
                        "timestamp": date,
                        "id": item_id,
                        "type": "news",
                        "source": "FINSABER",
                        "title": str(text).splitlines()[0][:160] if text else "",
                        "text": text or "",
                        "url": "",
                    }
                )

        return pd.DataFrame.from_records(
            records,
            columns=["timestamp", "id", "type", "source", "title", "text", "url"],
        ).sort_values("timestamp")
