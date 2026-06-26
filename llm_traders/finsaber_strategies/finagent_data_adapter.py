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

        self.prices = {}
        self.news = {}
        self.guidances = {}
        self.sentiments = None
        self.economics = None
        self._build_frames()

    @staticmethod
    def _news_title_and_text(item) -> tuple[str, str]:
        if isinstance(item, dict):
            title = item.get("title") or item.get("headline") or ""
            text = item.get("text") or item.get("content") or item.get("news_text") or ""
        else:
            title = ""
            text = str(item)
        title = title.strip() or text[:120].strip() or "Market news"
        return title, text

    def _build_frames(self) -> None:
        price_rows = {symbol: [] for symbol in self.asset}
        news_rows = {symbol: [] for symbol in self.asset}
        guidance_rows = {symbol: [] for symbol in self.asset}
        news_ids = {symbol: 0 for symbol in self.asset}

        for date, day in sorted(self.raw_data.items()):
            timestamp = pd.to_datetime(date)
            for symbol in self.asset:
                bar = day.get("price", {}).get(symbol)
                if isinstance(bar, dict):
                    raw_close = bar.get("close")
                    adjusted_close = bar.get("adjusted_close", bar.get("adj_close", raw_close))
                    price_rows[symbol].append(
                        {
                            "timestamp": timestamp,
                            "open": bar.get("adjusted_open", bar.get("open", adjusted_close)),
                            "high": bar.get("adjusted_high", bar.get("high", adjusted_close)),
                            "low": bar.get("adjusted_low", bar.get("low", adjusted_close)),
                            "close": adjusted_close,
                            "adj_close": adjusted_close,
                            "volume": bar.get("volume", 0),
                            "raw_open": bar.get("open"),
                            "raw_high": bar.get("high"),
                            "raw_low": bar.get("low"),
                            "raw_close": raw_close,
                        }
                    )

                news_items = day.get("news", {}).get(symbol, [])
                if isinstance(news_items, (str, dict)):
                    news_items = [news_items]
                for item in news_items or []:
                    title, text = self._news_title_and_text(item)
                    news_ids[symbol] += 1
                    news_rows[symbol].append(
                        {
                            "timestamp": timestamp,
                            "id": news_ids[symbol],
                            "type": "news",
                            "source": "FINSABER",
                            "title": title,
                            "text": text,
                            "url": "",
                        }
                    )

                for filing_key, filing_type in (("filing_k", "10-K"), ("filing_q", "10-Q")):
                    filing = day.get(filing_key, {}).get(symbol)
                    if filing:
                        guidance_rows[symbol].append(
                            {
                                "timestamp": timestamp,
                                "sentiment": "neutral",
                                "title": f"{symbol} {filing_type} Filing",
                                "text": str(filing),
                            }
                        )

        has_guidance = False
        price_columns = [
            "timestamp", "open", "high", "low", "close", "adj_close", "volume",
            "raw_open", "raw_high", "raw_low", "raw_close",
        ]
        news_columns = ["timestamp", "id", "type", "source", "title", "text", "url"]
        guidance_columns = ["timestamp", "sentiment", "title", "text"]
        for symbol in self.asset:
            self.prices[symbol] = pd.DataFrame(
                price_rows[symbol],
                columns=price_columns,
            ).sort_values("timestamp").reset_index(drop=True)
            self.news[symbol] = pd.DataFrame(
                news_rows[symbol],
                columns=news_columns,
            ).sort_values("timestamp").reset_index(drop=True)
            guidance = pd.DataFrame(
                guidance_rows[symbol],
                columns=guidance_columns,
            ).sort_values("timestamp").reset_index(drop=True)
            self.guidances[symbol] = guidance
            has_guidance = has_guidance or not guidance.empty

        if not has_guidance:
            self.guidances = None
