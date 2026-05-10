# FINSABER Experiment Reproduction Guide

This branch is a frozen reproduction branch for the paper **"Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?"** It keeps the original FINSABER experiment code, dataset interfaces, strategy wrappers, and runner scripts used for the paper. For the upgraded package-oriented framework, use the `v2.0` branch instead.

## 1. Repository Scope

Use this branch when you want to reproduce the paper experiments as closely as possible. The important paths are:

- `backtest/`: backtesting engine, experiment runners, baseline strategies, selectors, and aggregation utilities.
- `backtest/strategy/timing/`: non-LLM timing strategies such as buy-and-hold, SMA, WMA, Bollinger Bands, ATR bands, ARIMA, XGBoost, and FinRL.
- `backtest/strategy/timing_llm/`: wrappers for LLM-based strategies such as FinMem and FinAgent.
- `llm_traders/`: vendored agent implementations and their runtime files.
- `rl_traders/`: reinforcement learning strategy dependencies.
- `strats_configs/`: example configuration files for LLM strategies.
- `data/`: lightweight metadata and helper files. Large experiment datasets are downloaded from HuggingFace when needed.

Generated outputs should stay under `backtest/output/` or a custom directory passed through `--output_dir`.

## 2. Environment Setup

Clone this branch and create a Python 3.10 environment:

```bash
git clone -b reproduce https://github.com/waylonli/FINSABER.git
cd FINSABER
conda env create -f finsaber_env.yml
conda activate finsaber
```

If the exported environment is not portable on your machine, create a clean environment and install the pinned requirements:

```bash
conda create -n finsaber python=3.10
conda activate finsaber
pip install -r requirements-complete.txt --no-deps
```

Some optional dependencies are easier to install through conda. If `faiss` or `pandas-ta` fails during installation, install them separately:

```bash
conda install -c conda-forge faiss-cpu
pip install "git+https://github.com/aarigs/pandas-ta.git"
```

## 3. API Keys and Runtime Configuration

Copy the example environment file before running LLM strategies:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Set the following values in `.env`:

- `OPENAI_API_KEY`: required for FinMem and FinAgent experiments.
- `HF_ACCESS_TOKEN`: optional for public downloads, but recommended if HuggingFace rate limits your downloads.

Non-LLM baseline experiments do not require an OpenAI key.

## 4. Data

The paper uses aggregated price, news, and filing data hosted at:

```text
https://huggingface.co/datasets/waylonli/FINSABER-data
```

The runners download the required files automatically when the data is missing locally. The first full run can download several gigabytes, so start with `selected_4` before running the larger S&P 500 setups.

The main dataset variants are:

- S&P 500 full aggregated data: price, news, and filings.
- Price-only S&P 500 data: daily OHLCV data for baseline and selector experiments.
- Selected-symbol data: smaller data for quick FinMem and FinAgent checks.

Run all commands from the repository root so relative data paths resolve correctly.

## 5. Experiment Setups

The runner accepts a `--setup` argument that selects the asset universe and selection rule:

- `selected_4`: small selected-symbol setup for quick checks and selected-4 paper experiments.
- `cherry_pick_both_finmem`: selective setup aligned with FinMem experiments.
- `cherry_pick_both_fincon`: selective setup aligned with FinAgent/FinCon-style experiments.
- `random_sp500_5`: randomly selected S&P 500 portfolio with 5 tickers.
- `momentum_sp500_5`: momentum-selected S&P 500 portfolio with 5 tickers.
- `lowvol_sp500_5`: low-volatility S&P 500 portfolio with 5 tickers.
- `fincon_selector_sp500_5`: FinCon-style selector with 5 tickers, when the required selector assets are available.

Use `--date_from 2004-01-01 --date_to 2024-01-01` for full paper-period runs.

## 6. Quick Smoke Test

Before running expensive experiments, verify the environment and data loader with a short buy-and-hold run:

```bash
python backtest/run_baselines_exp.py \
  --setup selected_4 \
  --include BuyAndHoldStrategy \
  --date_from 2005-01-01 \
  --date_to 2007-01-01 \
  --output_dir backtest/output
```

This should create output under:

```text
backtest/output/selected_4/BuyAndHoldStrategy/
```

If this fails, fix the environment or dataset access before running LLM strategies.

## 7. Reproducing Baseline Strategies

Use `backtest/run_baselines_exp.py` for non-LLM strategies:

```bash
python backtest/run_baselines_exp.py \
  --setup selected_4 \
  --date_from 2004-01-01 \
  --date_to 2024-01-01 \
  --training_years 2 \
  --rolling_window_size 1 \
  --rolling_window_step 1 \
  --output_dir backtest/output
```

To run one strategy only, pass its class name:

```bash
python backtest/run_baselines_exp.py \
  --setup selected_4 \
  --include SMACrossStrategy \
  --date_from 2004-01-01 \
  --date_to 2024-01-01
```

Common baseline class names include:

- `BuyAndHoldStrategy`
- `SMACrossStrategy`
- `WMAStrategy`
- `BollingerBandsStrategy`
- `ATRBandStrategy`
- `TrendFollowingStrategy`
- `TurnOfTheMonthStrategy`
- `ARIMAPredictorStrategy`
- `XGBoostPredictorStrategy`
- `FinRLStrategy`

For paper-style rolling tests, run both relevant rolling-window settings:

```bash
--rolling_window_size 1 --rolling_window_step 1
--rolling_window_size 2 --rolling_window_step 1
```

For strategies requiring training windows, use the paper values:

```bash
--training_years 2
--training_years 3
```

## 8. Reproducing LLM Strategies

Use `backtest/run_llm_traders_exp.py` for FinMem and FinAgent wrappers. These runs require `OPENAI_API_KEY` in `.env`.

FinMem selected-4 example:

```bash
python backtest/run_llm_traders_exp.py \
  --setup selected_4 \
  --strategy FinMemStrategy \
  --strat_config_path strats_configs/finmem_config_normal.json \
  --date_from 2004-01-01 \
  --date_to 2024-01-01 \
  --rolling_window_size 1 \
  --rolling_window_step 1 \
  --output_dir backtest/output
```

FinAgent selected-4 example:

```bash
python backtest/run_llm_traders_exp.py \
  --setup selected_4 \
  --strategy FinAgentStrategy \
  --strat_config_path strats_configs/finagent_config_normal.json \
  --date_from 2004-01-01 \
  --date_to 2024-01-01 \
  --rolling_window_size 1 \
  --rolling_window_step 1 \
  --output_dir backtest/output
```

Use the matching configuration family for each setup:

- `*_config_normal.json`: selected-4 and standard runs.
- `*_config_cherry.json`: selective/cherry-pick runs.
- `*_config_composite.json`: composite S&P 500 selector runs.

LLM experiments can be slow and may incur API costs. Start with a short date range or one ticker setup before launching full-period runs.

## 9. Paper Experiment Checklist

To reproduce the main result tables, run the following groups:

1. Selected-4 experiments: run baselines and LLM strategies with `--setup selected_4`.
2. Selective experiments: run the FinMem-aligned setup with `cherry_pick_both_finmem` and the FinAgent/FinCon-aligned setup with `cherry_pick_both_fincon`.
3. Composite experiments: run S&P 500 selector setups such as `random_sp500_5`, `momentum_sp500_5`, and `lowvol_sp500_5`.
4. Rolling-window sensitivity: repeat the relevant experiments with `--rolling_window_size 1` and `--rolling_window_size 2`.
5. Training-window sensitivity: repeat trainable baselines with `--training_years 2` and `--training_years 3`.

Keep the date range fixed unless you intentionally run an ablation:

```bash
--date_from 2004-01-01 --date_to 2024-01-01
```

## 10. Results and Aggregation

Each run writes per-strategy results under:

```text
backtest/output/<setup>/<strategy>/
```

Typical outputs include serialized result files and aggregated CSV summaries such as `results.csv`. The runner calls aggregation utilities after a strategy finishes. If you use a custom output directory, pass the same `--output_dir` value across related runs so results are grouped consistently.

Notebook utilities are available for additional inspection:

- `backtest/results_aggregation.ipynb`: result table aggregation.
- `backtest/output/market_regime_analysis.ipynb`: market-regime analysis helpers.

Do not commit generated output files, logs, caches, or downloaded datasets.

## 11. Troubleshooting

If data download fails, check your network connection and `HF_ACCESS_TOKEN`. If LLM runs fail, confirm `.env` exists and `OPENAI_API_KEY` is valid. If imports fail, make sure you are using Python 3.10 and running commands from the repository root. If baseline strategies work but LLM strategies fail, isolate the issue by first running `BuyAndHoldStrategy` on `selected_4`, then run one LLM strategy over a short date range.

This branch is intended for reproducibility rather than active framework development. New backtesting framework changes should be made on `v2.0`.
