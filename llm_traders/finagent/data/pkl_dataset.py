import os
import pickle
import numpy as np
import pandas as pd

from llm_traders.finagent.data import BaseDataset
from llm_traders.finagent.registry import DATASET


@DATASET.register_module(force=True)
class PklDataset(BaseDataset):
    """Dataset that loads from FINSABER's aggregated .pkl files.

    The enriched pkl structure is:
        {datetime.date: {
            "price": {ticker: {"open", "high", "low", "close", "adjusted_close", "volume"}},
            "news":  {ticker: [str, ...]},
            "filing_k": {ticker: str},
            "filing_q": {ticker: str},
        }}

    This class converts the above into the DataFrame-based attributes
    (prices, news, guidances, sentiments, economics) that EnvironmentTrading expects.
    """

    def __init__(self,
                 asset: list,
                 pkl_path: str,
                 workdir: str = None,
                 tag: str = None,
                 interval: str = "day",
                 ):
        super(PklDataset, self).__init__()

        self.interval = interval
        self.workdir = workdir
        self.tag = tag

        self.exp_path = os.path.join(self.workdir, self.tag)
        os.makedirs(self.exp_path, exist_ok=True)

        self.assets = asset

        with open(pkl_path, "rb") as f:
            self.raw_data = pickle.load(f)

        self.prices = self._build_prices()
        self.news = self._build_news()
        self.guidances = self._build_guidances()
        self.sentiments = None
        self.economics = None

    def _build_prices(self):
        prices = {}
        for asset in self.assets:
            rows = []
            for date in sorted(self.raw_data.keys()):
                day_data = self.raw_data[date]
                if "price" not in day_data or asset not in day_data["price"]:
                    continue
                entry = day_data["price"][asset]
                if isinstance(entry, dict):
                    rows.append({
                        "timestamp": date,
                        "symbol": asset,
                        "open": entry.get("open"),
                        "high": entry.get("high"),
                        "low": entry.get("low"),
                        "close": entry.get("close"),
                        "adj_close": entry.get("adjusted_close"),
                        "volume": entry.get("volume"),
                    })
                else:
                    # Legacy format: single float (adjusted close)
                    rows.append({
                        "timestamp": date,
                        "symbol": asset,
                        "open": np.nan,
                        "high": np.nan,
                        "low": np.nan,
                        "close": np.nan,
                        "adj_close": entry,
                        "volume": np.nan,
                    })

            df = pd.DataFrame(rows)
            if len(df) > 0:
                # Replace None with NaN for numeric consistency
                for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.sort_values(by="timestamp").reset_index(drop=True)
            else:
                df = pd.DataFrame(columns=["timestamp", "symbol", "open", "high",
                                           "low", "close", "adj_close", "volume"])
            prices[asset] = df
        return prices

    def _build_news(self):
        news = {}
        global_id = 0
        for asset in self.assets:
            rows = []
            for date in sorted(self.raw_data.keys()):
                day_data = self.raw_data[date]
                if "news" not in day_data or asset not in day_data.get("news", {}):
                    continue
                news_list = day_data["news"][asset]
                if isinstance(news_list, list):
                    for article in news_list:
                        rows.append({
                            "timestamp": date,
                            "id": f"{global_id:06d}",
                            "source": "aggregated",
                            "title": article if isinstance(article, str) else str(article),
                        })
                        global_id += 1
            df = pd.DataFrame(rows)
            if len(df) > 0:
                df = df.sort_values(by="timestamp").reset_index(drop=True)
            else:
                df = pd.DataFrame(columns=["timestamp", "id", "source", "title"])
            news[asset] = df
        return news

    def _build_guidances(self):
        guidances = {}
        has_any = False
        for asset in self.assets:
            rows = []
            for date in sorted(self.raw_data.keys()):
                day_data = self.raw_data[date]
                for filing_key, filing_type in [("filing_k", "10-K"), ("filing_q", "10-Q")]:
                    filing_dict = day_data.get(filing_key, {})
                    if asset not in filing_dict:
                        continue
                    text = filing_dict[asset]
                    if text is not None and text != "":
                        rows.append({
                            "timestamp": date,
                            "sentiment": "neutral",
                            "title": f"{asset} {filing_type} Filing",
                            "text": text if isinstance(text, str) else str(text),
                        })
                        has_any = True
            df = pd.DataFrame(rows)
            if len(df) > 0:
                df = df.sort_values(by="timestamp").reset_index(drop=True)
            else:
                df = pd.DataFrame(columns=["timestamp", "sentiment", "title", "text"])
            guidances[asset] = df
        return guidances if has_any else None
