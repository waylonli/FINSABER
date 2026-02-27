# FINSABER

[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=2505.07078&color=red&logo=arxiv)](https://arxiv.org/abs/2505.07078)
<a href="https://pypi.org/project/finsaber/"><img alt="PyPI" src="https://img.shields.io/pypi/v/finsaber"></a>
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/waylonli/FINSABER?style=social)]()

**Official implementation for the KDD 2026 paper: "Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?"**


## 📣 News
* **[24/11/2025]** We are excited to announce that FINSABER has been accepted to **KDD 2026**! 🚀
* **[19/06/2025]** Code and initial benchmarks released.

## 📖 Overview
FINSABER is a comprehensive framework for evaluating trading strategies with a specific focus on comparing traditional technical analysis approaches with modern machine learning and large language model (LLM) based strategies. 

<img src="https://github.com/waylonli/FINSABER/blob/main/figs/framework.png" width="900">

## 1. Environment Setup

To set up the environment, you can use either only install the FINSABER backtest framework or install the full package with all dependencies for running the experiments in the paper.

### Option 1: Install the full package (recommend):

```bash
git clone https://github.com/waylonli/FINSABER
cd FINSABER
conda create -n finsaber python=3.10
conda activate finsaber
pip install -r requirements-complete.txt --no-deps
```

Alternatively, you can import the conda environment through `conda env create -f finsaber_env.yml`.

### Option 2: Install only the FINSABER backtest framework

Simply pip install the package from PyPI:

```bash
conda create -n finsaber python=3.10
pip install finsaber
```

Note: it seems `pip` has issue with installing the `faiss` package. Please install it via anaconda.

### After Installation

Rename `.env.example` to `.env` and set the environment variables. 
- `OPENAI_API_KEY` is required to run LLM-based strategies. 
- `HF_ACCESS_TOKEN` is optional.

## 2. Data

We provide aggregated datasets on [HuggingFace](https://huggingface.co/datasets/waylonli/FINSABER-data). Datasets are **auto-downloaded** when running experiments, so manual download is optional.

| Dataset | Content | Size | Link |
| :--- | :--- | :--- | :--- |
| **S&P500 Full** | Aggregated data (Price + News + Filings) | ~11 GB | [Download](https://huggingface.co/datasets/waylonli/FINSABER-data/resolve/main/data/finmem_data/stock_data_sp500_2000_2024.pkl) |
| **Price Only** | CSV format price-only data | ~253 MB | [Download](https://huggingface.co/datasets/waylonli/FINSABER-data/resolve/main/data/price/all_sp500_prices_2000_2024_delisted_include.csv) |
| **Selected Symbols** | Aggregated data for TSLA, AMZN, MSFT, NFLX, COIN | ~53 MB | [Download](https://huggingface.co/datasets/waylonli/FINSABER-data/resolve/main/data/finmem_data/stock_data_cherrypick_2000_2024.pkl) |

The aggregated data is organised as a dictionary with the following structure:
```python
{
    datetime.date(2024, 1, 2): {
        "price": {
            "AAPL": {
                "open": 187.15,
                "high": 188.44,
                "low": 183.89,
                "close": 185.64,
                "adjusted_close": 185.3,
                "volume": 82488700
            },
            ...
        },
        "news": {
            "AAPL": ["headline 1", "headline 2", ...],
            ...
        },
        "filing_k": {
            "AAPL": "10-K filing text...",
            ...
        },
        "filing_q": {
            "AAPL": "10-Q filing text...",
            ...
        }
    },
    ...
}
```

To plug in your own data, simply inherit the `backtest.data_util.backtest_dataset.BacktestDataset` class and implement the necessary methods.
An example for processing the data format above is provided in `backtest/data_util/finmem_dataset.py`.


## 3. Reproduce the results in the paper

The paper contains three experimental setup: *selective (cherry picking) setup*, *selected-4 setup*, and *composite setup*.
These three experiments can be reproduced by running the following commands.

Baselines (non-LLM):
```bash
python backtest/run_baselines_exp.py \
    --setup <setup_name> \ # can be "cherry_pick_both_finmem", "cherry_pick_both_fincon", "selected_4", "random_sp500_5", "momentum_sp500_5", "lowvol_sp500_5"
    --include <strategy_name> \ # can be one of the class name under backtest/strategy/timing
    --date_from 2004-01-01 \
    --date_to 2024-01-01 \
    --training_years <training_years> # 2 or 3 in the paper
    --rolling_window_size <window_size> \ # 1 or 2 in the paper
    --rolling_window_step 1
```

LLM Strategies:
```bash
python backtest/run_llm_traders_exp.py \
    --setup <setup_name> \ # can be "cherry_pick_both_finmem", "cherry_pick_both_fincon", "selected_4", "random_sp500_5", "momentum_sp500_5", "lowvol_sp500_5"
    --strategy <strategy_name> \ # can be one of the class name under backtest/strategy/timing_llm
    --strat_config_path <config_path> \ # path to the config file for the LLM strategy, examples under strats_configs folder
    --date_from 2004-01-01 \
    --date_to 2024-01-01 \
    --rolling_window_size <window_size> \ # 1 or 2 in the paper
    --rolling_window_step 1
```

## 4. Extend the framework

You can plug in your own datasets and strategies by subclassing the helpers that live under `backtest/strategy` and `backtest/data_util`. The snippets below show the minimum interfaces you need to implement.

### 4.1 Custom timing strategy (Backtrader)

1. Create a new file inside `backtest/strategy/timing` and subclass `backtest.strategy.timing.base_strategy.BaseStrategy`.
2. Add any tunable values to the class-level `params` tuple and build the indicators you need in `__init__`.
3. Implement `next()` and call `self.post_next_actions()` at the end of each bar so the framework can track equity, trades, and drawdowns.
4. Import your class in `backtest/strategy/timing/__init__.py`; experiment launchers such as `backtest/run_baselines_exp.py` discover strategies from that module.

```python
# backtest/strategy/timing/my_strategy.py
import backtrader as bt
from backtest.strategy.timing.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    params = (
        ("prior_period", 252 * 2),
        ("fast_window", 20),
        ("slow_window", 100),
    )

    def __init__(self):
        super().__init__()
        self.fast = bt.indicators.SMA(self.data.close, period=self.params.fast_window)
        self.slow = bt.indicators.SMA(self.data.close, period=self.params.slow_window)

    def next(self):
        if self.fast[0] > self.slow[0] and not self.position:
            size = self._adjust_size_for_commission(int(self.broker.cash / self.data.close[0]))
            if size > 0:
                self.buy(size=size)
        elif self.fast[0] < self.slow[0] and self.position:
            self.close()
        self.post_next_actions()
```

You can then backtest it with `python backtest/run_baselines_exp.py --setup <setup> --include MyStrategy` or by creating a `FINSABERBt` instance and calling `run_rolling_window(MyStrategy)` directly.

### 4.2 Custom LLM timing strategy

LLM-driven timing strategies inherit from `backtest.strategy.timing_llm.base_strategy_iso.BaseStrategyIso` and implement an `on_data(date, today_data, framework)` loop. The helper `backtest.toolkit.backtest_framework_iso.FINSABERFrameworkHelper` that is passed in exposes `buy(...)`, `sell(...)`, portfolio state, and equity accounting.

```python
# backtest/strategy/timing_llm/my_llm_strategy.py
from backtest.strategy.timing_llm import BaseStrategyIso

class MyLLMStrategy(BaseStrategyIso):
    def __init__(self, symbol, date_from, date_to, model):
        super().__init__()
        self.symbol = symbol
        self.model = model  # plug in your agent / API client here
        self.date_from = date_from
        self.date_to = date_to

    def train(self):
        # Optional: warm up the LLM agent on historical data
        pass

    def on_data(self, date, today_data, framework):
        prices = today_data["price"]
        cur_price = prices[self.symbol]["adjusted_close"]
        signal = self.model.decide(date=date, prices=prices)
        if signal == "buy" and framework.cash >= cur_price:
            framework.buy(date, self.symbol, cur_price, -1)  # invest available cash
        elif signal == "sell" and self.symbol in framework.portfolio:
            qty = framework.portfolio[self.symbol]["quantity"]
            framework.sell(date, self.symbol, cur_price, qty)
```

Expose the strategy via `backtest/strategy/timing_llm/__init__.py` and point `backtest/run_llm_traders_exp.py` to it with `--strategy MyLLMStrategy --strat_config_path <config>`.

### 4.3 Custom selection strategy

Selection strategies control which tickers enter each rolling window. Subclass `backtest.strategy.selection.base_selector.BaseSelector`, implement a `select(data_loader, start_date, end_date)` method that returns a list of tickers, and register the class in `backtest/strategy/selection/__init__.py` if you want to import it elsewhere.

```python
# backtest/strategy/selection/top_volume_selector.py
from backtest.strategy.selection import BaseSelector

class TopVolumeSelector(BaseSelector):
    def __init__(self, top_k=5):
        self.top_k = top_k

    def select(self, data_loader, start_date, end_date):
        window = data_loader.get_subset_by_time_range(start_date, end_date)
        avg_volume = {}
        for date in window.get_date_range():
            for ticker, metrics in window.get_data_by_date(date)["price"].items():
                avg_volume.setdefault(ticker, []).append(metrics["volume"])
        ranked = sorted(avg_volume, key=lambda t: sum(avg_volume[t]) / len(avg_volume[t]), reverse=True)
        return ranked[: self.top_k]
```

Pass an instance through the trade config (`selection_strategy=TopVolumeSelector(top_k=5)`) when you create `FINSABERBt` or `FINSABER`.

### 4.4 Custom dataset loader

Datasets back the LLM framework and custom selection logic. Derive from `backtest.data_util.backtest_dataset.BacktestDataset` and implement the accessor methods that the framework calls.

```python
# backtest/data_util/my_dataset.py
from backtest.data_util import BacktestDataset

class MyDataset(BacktestDataset):
    def __init__(self, dataframe):
        self.data = dataframe  # {date: {"price": {...}}}

    def get_ticker_price_by_date(self, ticker, date):
        return self.data[date]["price"][ticker]

    def get_data_by_date(self, date):
        return self.data.get(date, {})

    def get_subset_by_time_range(self, start_date, end_date):
        subset = {d: v for d, v in self.data.items() if start_date <= d <= end_date}
        return MyDataset(subset) if subset else None

    def get_ticker_subset_by_time_range(self, ticker, start_date, end_date):
        subset = {d: {"price": {ticker: v["price"][ticker]}} for d, v in self.data.items() if start_date <= d <= end_date and ticker in v["price"]}
        return MyDataset(subset) if subset else None

    def get_date_range(self):
        return sorted(self.data.keys())

    def get_tickers_list(self):
        symbols = set()
        for day in self.data.values():
            symbols.update(day["price"].keys())
        return sorted(symbols)
```

Once the loader is available (optionally expose it via `backtest/data_util/__init__.py`), point a trade config at it:

```python
from backtest.finsaber import FINSABER
from backtest.data_util.my_dataset import MyDataset
from backtest.strategy.selection.top_volume_selector import TopVolumeSelector

trade_config = {
    "tickers": "all",
    "setup_name": "custom_run",
    "date_from": "2015-01-01",
    "date_to": "2020-01-01",
    "selection_strategy": TopVolumeSelector(top_k=5),
    "data_loader": MyDataset(my_loaded_data),
}

engine = FINSABER(trade_config)
engine.run_rolling_window(MyLLMStrategy, strat_params={"symbol": "AAPL", "date_from": "2015-01-01", "date_to": "2020-01-01", "model": my_agent})
```

## Citation

```
@misc{li2025llmbasedfinancialinvestingstrategies,
      title={Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?}, 
      author={Weixian Waylon Li and Hyeonjun Kim and Mihai Cucuringu and Tiejun Ma},
      year={2025},
      eprint={2505.07078},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2505.07078}, 
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=waylonli/FINSABER&type=date&legend=top-left)](https://www.star-history.com/#waylonli/FINSABER&type=date&legend=top-left)
