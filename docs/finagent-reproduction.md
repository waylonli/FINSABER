# FinAgent FINSABER-2 Experiments

This workflow evaluates FinAgent with the FINSABER-2 parquet dataset over the
2024 and 2025 calendar-year windows. The versioned manifest fixes the ticker
selections, model, data modalities, execution assumptions, and random seed.

## Prerequisites

Activate the Python 3.10 environment and configure `OPENAI_API_KEY` in an
ignored `.env` file. Download the FINSABER-2 dataset and pass its local root
explicitly. The expected folders are `price_daily/`, `news_items/`, `filingk/`,
and `filingq/`.

Preview the complete plan without sending data to OpenAI:

```bash
python examples/experiments/run_finagent_finsaber2.py --plan \
  --data-root /path/to/sp500_2000_2025_parquet
```

Run only the Magnificent 7 extension:

```bash
python examples/experiments/run_finagent_finsaber2.py \
  --setups magnificent_7 \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root tmp/finagent-magnificent7-2024-2026-r1 \
  --max-parallel 4
```

The standard Magnificent 7 snapshot is `AAPL`, `AMZN`, `GOOGL`, `META`,
`MSFT`, `NVDA`, and `TSLA`. Each ticker-year is an independent job with a
three-year training window. Successful jobs are skipped on restart.

Run the matching non-LLM benchmark suite:

```bash
python examples/experiments/run_finsaber2_benchmarks.py \
  --setup magnificent_7 \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root tmp/magnificent7-benchmarks-2024-2026-r1
```

This runs Buy-and-Hold, six technical strategies, ARIMA, XGBoost, and FinRL.
ARIMA and XGBoost use three years of training data; FinRL uses a distinct
10-year training window. The runner uses seed 2026 and skips strategies that
already have all 14 ticker-year artifacts.

## Reproducibility Artifacts

Each output directory contains `experiment_config.json`,
`runner_manifest.json`, per-job `job_status.json`, scalar `metrics.json`,
equity/trade/cost CSVs, isolated FinAgent workdirs, generated chart images,
and stdout/stderr logs. The manifest records the Git commit, Python version,
model name, seed, resolved dataset path, and exact selections.

Hosted LLM output is not bitwise reproducible because provider-side model
revisions and sampling can change. Preserve the complete output directory;
the prompt, response, chart, and cost artifacts provide the audit trail.

## Consolidating Results

After all configured runs finish, rebuild the cross-strategy tables and plots:

```bash
python examples/experiments/summarize_finsaber2_results.py \
  --tmp-root tmp \
  --output-root tmp/consolidated-finsaber2-2024-2026-r1
```

The command verifies all 594 expected ticker-year results, including the full
Magnificent 7 benchmark suite, and rejects duplicate identities. It recomputes
volatility, Sharpe, and Sortino from the saved equity curves with the current
framework metrics and a 3% annual risk-free rate. It retains the stored
experiment-time metrics for audit and excludes near-cash runs with less than
0.5% annualized volatility from reported mean Sharpe. Magnificent 7 is the
primary fixed-universe comparison; Selected-4 is retained as a historical
appendix.

## FinRL Result Status

Current 2024-2025 FinRL results are preliminary. The exported action history
has been verified to contain the post-`hmax`, integer share quantities actually
executed inside the FinRL environment, so replay must not scale them again.
However, deterministic 5,000-step runs may still hold cash or trade only one
share even with the required 10-year training window. Longer-training variants
are therefore still recommended before making a definitive RL comparison.
Near-zero-exposure Sharpe values remain undefined rather than being averaged
as extreme ratios.
