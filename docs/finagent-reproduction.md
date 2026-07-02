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

The command verifies all 454 expected ticker-year results, rejects duplicate
identities, and recomputes volatility, Sharpe, and Sortino from the saved equity
curves with the current framework metrics and a 3% annual risk-free rate. It
retains the stored experiment-time metrics for audit and excludes near-cash
runs with less than 0.5% annualized volatility from reported mean Sharpe.

## FinRL Result Status

Current 2024-2025 FinRL results are preliminary. Some runs bought only one
share, leaving almost all capital idle and producing unstable Sharpe values.
Before reporting FinRL, verify whether normalized actions are truncated by
`int()`, whether `hmax` scaling survives prediction export, and whether the
default 5,000 training steps are sufficient. Rerun controlled 3-year and
10-year training-window variants after fixing action sizing. Mark near-zero
exposure Sharpe values as undefined rather than averaging extreme ratios.
