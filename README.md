# FINSABER

[![ACM](https://img.shields.io/static/v1?label=ACM&message=10.1145/3770854.3785702&color=blue&logo=acm)](https://dl.acm.org/doi/10.1145/3770854.3785702)
[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=2505.07078&color=red&logo=arxiv)](https://arxiv.org/abs/2505.07078)
<a href="https://pypi.org/project/finsaber/"><img alt="PyPI" src="https://img.shields.io/pypi/v/finsaber"></a>
[![Documentation](https://img.shields.io/badge/docs-FINSABER--2-15616d?logo=readthedocs&logoColor=white)](https://waylonli.github.io/FINSABER/)
[![Dataset](https://img.shields.io/badge/HuggingFace-FINSABER--V2--Data-yellow?logo=huggingface)](https://huggingface.co/datasets/finsaber-team/FINSABER-V2-Data)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/waylonli/FINSABER?style=social)]()

**Official implementation for the KDD 2026 paper: "Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?"**

## 📣 News

- **[11/05/2026]** FINSABER has been upgraded to **FINSABER-2**. The backtesting framework is now package-oriented, supports the new parquet dataset format, and includes explicit execution timing, adjusted OHLC handling, slippage, liquidity caps, structured result artifacts, and LLM cost accounting.
- **[24/11/2025]** We are excited to announce that FINSABER has been accepted to **KDD 2026**.
- **[19/06/2025]** Code and initial benchmarks released.

## 📖 Overview

FINSABER is a framework for evaluating financial trading strategies, including traditional technical analysis, machine learning methods, and LLM-based agents. FINSABER-2 focuses on a cleaner reusable backtesting package over price, news, filings, and extensible market data.

<img src="https://github.com/waylonli/FINSABER/blob/main/figs/framework.png" width="900">

The packaged core is intentionally limited to reusable backtesting code: data loaders, execution models, metrics, result writers, selectors, and strategy interfaces. Paper-specific FinMem, FinAgent, FinCon, and FinRL integrations live outside the core package.

## Branch and Dataset

Use the branch that matches your workflow and dataset:

| Branch | Use case | Dataset |
| --- | --- | --- |
| `main` | Current FINSABER-2 package workflow and active development | [FINSABER-V2-Data](https://huggingface.co/datasets/finsaber-team/FINSABER-V2-Data) |
| `reproduce` | Reproduce FINSABER-1 experimental results with the v1.0 framework | [FINSABER-reproduce](https://huggingface.co/datasets/finsaber-team/FINSABER-reproduce) |

If your goal is to reproduce the original paper experiments on the legacy FINSABER dataset, use the `reproduce` branch. Use `main` for new FINSABER-2 package development.

## Install

For framework use:

```bash
pip install finsaber
```

For local development on FINSABER-2:

```bash
git clone https://github.com/waylonli/FINSABER
cd FINSABER
conda create -n trading python=3.10 -y
conda activate trading
python -m pip install -U pip setuptools wheel
pip install -e ".[dev,research]"
```

If you also want docs dependencies:

```bash
pip install -e ".[docs]"
```

Build the wheel:

```bash
python -m build --wheel
```

## Documentation

Read the FINSABER-2 documentation at [https://waylonli.github.io/FINSABER/](https://waylonli.github.io/FINSABER/).

## Quick Start

Run a Buy-and-Hold backtest on the parquet dataset:

```python
from finsaber import FINSABERBt, FinsaberParquetDataset
from finsaber.strategy.timing import BuyAndHoldStrategy

data = FinsaberParquetDataset("/path/to/sp500_2000_2025_parquet")

config = {
    "data_loader": data,
    "tickers": ["AAPL"],
    "date_from": "2024-01-02",
    "date_to": "2024-01-10",
    "setup_name": "demo_buy_hold",
    "execution_timing": "next_open",
    "slippage_perc": 0.0005,
    "liquidity_cap_pct": 0.025,
    "save_results": True,
    "silence": True,
}

results = FINSABERBt(config).run_iterative_tickers(BuyAndHoldStrategy)
print(results["AAPL"]["total_return"])
```

Outputs are written under `backtest/output/<setup>/<strategy>/` by default:

- `run_config.json`: resolved run configuration.
- `run_manifest.json`: artifact schema.
- `run_summary.csv`: flat per-window/per-ticker summary.
- `metrics.json`: scalar metrics.
- `equity_curve.csv`, `trades.csv`, `orders.csv`, `rejected_orders.csv`, `llm_costs.csv`: detailed tables when available.

## Data

FINSABER expects pluggable data loaders that implement `TradingData`. The built-in parquet loader reads this layout:

```text
sp500_2000_2025_parquet/
  price_daily/year=YYYY/part-000.parquet
  news_items/year=YYYY/part-000.parquet
  filingk/year=YYYY/part-000.parquet
  filingq/year=YYYY/part-000.parquet
```

`FinsaberParquetDataset` computes split-adjusted `adjusted_open`, `adjusted_high`, and `adjusted_low` from raw OHLC and `adjusted_close`. Raw volume is retained for liquidity caps.

For small or custom datasets, `FinsaberDataset` accepts an in-memory dictionary:

```python
from datetime import date
from finsaber import FinsaberDataset

data = {
    date(2024, 1, 2): {
        "price": {
            "DEMO": {
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "adjusted_close": 101.0,
                "volume": 1_000_000,
            }
        },
        "news": {"DEMO": ["optional news text"]},
        "filing_k": {},
        "filing_q": {},
    }
}

loader = FinsaberDataset(data=data)
```

A runnable example is available at `examples/custom_dataset_example.py`.

## Implement Your Own Data Loader

Implement `TradingData` when your storage format is not a date-keyed dictionary.

```python
from finsaber import TradingData

class MyData(TradingData):
    def __init__(self, store):
        self.store = store

    def get_data_by_date(self, date):
        return self.store.get(date, {})

    def get_ticker_price_by_date(self, ticker, date, price_field=None):
        bar = self.store[date]["price"][ticker]
        return bar[price_field or "adjusted_close"]

    def get_ticker_data_by_date(self, ticker, date):
        day = self.get_data_by_date(date)
        return {name: values[ticker] for name, values in day.items() if ticker in values}

    def get_tickers_list(self):
        tickers = set()
        for day in self.store.values():
            tickers.update(day.get("price", {}))
        return sorted(tickers)

    def get_subset_by_time_range(self, start_date, end_date):
        subset = {d: v for d, v in self.store.items() if start_date <= d <= end_date}
        return MyData(subset) if subset else None

    def get_ticker_subset_by_time_range(self, ticker, start_date, end_date):
        subset = {}
        for d, day in self.store.items():
            if start_date <= d <= end_date and ticker in day.get("price", {}):
                subset[d] = {"price": {ticker: day["price"][ticker]}}
        return MyData(subset) if subset else None

    def get_date_range(self):
        return sorted(self.store)
```

You can add extra modalities such as earnings calls, transcripts, analyst reports, or alternative data. Keep them under the daily dictionary, for example `{"earnings_call": {"AAPL": "..."}}`.

## Implement a Backtrader Strategy

Backtrader strategies are used by `FINSABERBt`. Subclass `BaseStrategy`, implement `next()`, and call `post_next_actions()` each bar so equity is tracked.

```python
import backtrader as bt
from finsaber.strategy.timing.base_strategy import BaseStrategy

class MovingAverageCross(BaseStrategy):
    params = (
        ("fast", 20),
        ("slow", 60),
        ("prior_period", 252),
    )

    def __init__(self):
        super().__init__()
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow)

    def next(self):
        if self.fast_ma[0] > self.slow_ma[0] and not self.position:
            size = self._adjust_size_for_commission(int(self.broker.cash / self.data.close[0]))
            if size > 0:
                self.buy(size=size)
        elif self.fast_ma[0] < self.slow_ma[0] and self.position:
            self.close()
        self.post_next_actions()
```

Run it:

```python
results = FINSABERBt(config).run_iterative_tickers(MovingAverageCross)
```

## Implement an LLM-Style Strategy

LLM-style strategies use the pure Python engine `FINSABER`. They receive date-level data and submit orders through the framework object. The default execution timing is `next_open`, which avoids same-day close look-ahead bias.

```python
from finsaber import FINSABER
from finsaber.strategy.timing_llm import BaseStrategyIso

class RuleBasedAgent(BaseStrategyIso):
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def on_data(self, date, today_data, framework):
        bar = today_data["price"][self.symbol]
        news = today_data.get("news", {}).get(self.symbol, [])
        if news and "upgrade" in " ".join(news).lower():
            framework.buy(date, self.symbol, bar["adjusted_close"], -1)
        elif self.symbol in framework.portfolio:
            framework.sell(date, self.symbol, bar["adjusted_close"], -1)

config = {
    "data_loader": data,
    "tickers": ["AAPL"],
    "date_from": "2024-01-02",
    "date_to": "2024-03-01",
    "setup_name": "agent_demo",
    "save_results": True,
}

results = FINSABER(config).run_iterative_tickers(
    RuleBasedAgent,
    strat_params={"symbol": "$symbol"},
)
```

If your strategy calls an LLM, record cost through `finsaber.toolkit.llm_cost_monitor`. FINSABER can include LLM costs in `total_trading_cost`.

## Execution Settings

Important `TradeConfig` fields:

```python
{
    "execution_timing": "next_open",      # or "same_close"
    "commission_per_share": 0.0049,
    "min_commission": 0.99,
    "max_commission_rate": 0.01,
    "slippage_perc": 0.0005,
    "slippage_impact": 0.0,
    "liquidity_lookback_days": 20,
    "liquidity_min_history_days": 1,
    "liquidity_cap_pct": 0.025,
    "llm_cost_as_trade_cost": True,
}
```

Use adjusted OHLC for price simulation and raw volume for liquidity caps. Date-only news or filing data should be treated as available no earlier than the next trading decision unless you have timestamps.

## Experiment Scripts

Research and paper-style launchers are examples, not part of the package wheel:

```bash
python examples/experiments/run_baselines_exp.py --setup selected_4 --include BuyAndHoldStrategy
python examples/experiments/run_llm_traders_exp.py --setup selected_4 --strategy FinMemStrategy --strat_config_path strats_configs/finmem_config_normal.json
```

FinMem, FinAgent, FinCon, and FinRL integrations remain in `llm_traders/` and `rl_traders/` for repository experiments.

## Reproducing FINSABER-1 Results

The original FINSABER-1 reproduction workflow is preserved on the `reproduce` branch:

```bash
git checkout reproduce
```

Use that branch if you want to reproduce the experimental results in the original paper using the v1.0 framework and the legacy aggregated FINSABER dataset. The `reproduce` README contains the environment setup, dataset links, and exact experiment commands for the original benchmark pipeline.

## Citation

```bibtex
@inproceedings{10.1145/3770854.3785702,
    author = {Li, Weixian Waylon and Kim, Hyeonjun and Cucuringu, Mihai and Ma, Tiejun},
    title = {Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?},
    year = {2026},
    isbn = {9798400722585},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3770854.3785702},
    doi = {10.1145/3770854.3785702},
    booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1},
    pages = {2711–2722},
    numpages = {12},
    keywords = {automated trading, llm investors, backtest, benchmark},
    location = {Republic of Korea},
    series = {KDD '26}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=waylonli/FINSABER&type=date&legend=top-left)](https://www.star-history.com/#waylonli/FINSABER&type=date&legend=top-left)
