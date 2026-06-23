# FinAgent Slurm Workflow

These scripts run the FINSABER-1 FinAgent paper reruns on the `reproduce` branch with one rolling window per Slurm array job. The default model is `gpt-4o-mini`.

## What Gets Submitted

`SUITE=paper_rest` generates 99 jobs:

- `selected_4`: 19 two-year windows from `2004-01-01_2006-01-01` through `2022-01-01_2024-01-01`, using `TSLA,NFLX,AMZN,MSFT` and `strats_configs/finagent_config_normal.json`.
- `random_sp500_5`, `momentum_sp500_5`, `lowvol_sp500_5`, `fincon_selector_sp500_5`: 20 one-year windows each from `2004-01-01_2005-01-01` through `2023-01-01_2024-01-01`, using selector-driven tickers and `strats_configs/finagent_config_composite.json`.

Alternative suites are `selected4_long` and `composite`.

## Submit

Run from the repository root on the cluster login node:

```bash
CONDA_ENV=trading MAX_PARALLEL=24 bash slurm/submit_finagent_gpt4o_mini_array.sh
```

If the cluster environment is named differently, override it:

```bash
CONDA_ENV=traiding bash slurm/submit_finagent_gpt4o_mini_array.sh
```

Required environment variables for OpenAI should be provided through `.env` or the scheduler environment. Do not commit `.env`.

## Resource Defaults

The worker requests `4` CPUs, `64G` RAM, and `36:00:00` wall time per job. The RAM request is intentionally conservative because each task loads the FINSABER-1 pickle dataset and builds FinAgent memory/plot artifacts. If cluster utilization is too high after a smoke test, reduce `--mem` in `slurm/sbatch_finagent_window.sh`.

## IO Isolation

Each task writes to a unique directory:

```text
runs/finagent_gpt4o_mini/<setup>/<date_from>_<date_to>/job_<slurm_job_id>_<array_task_id>/
```

Inside each directory:

- `metrics.pkl`: pickled metrics for that single window.
- `summary.json`: compact metrics and estimated LLM cost.
- `equity/*.parquet`: per-ticker equity curves.
- `finagent_workdir/`: isolated FinAgent prompts, memory, and chart artifacts.

This avoids conflicts across parallel windows, including windows that trade the same ticker.

## Resume Behavior

`finagent_one_window.py` skips a task when `summary.json` exists with `success=true`. To rerun a task, delete that job directory or pass `--force` when running the Python runner manually.

## Aggregate After Completion

After the array completes:

```bash
python slurm/aggregate_finagent_windows.py --output-root runs/finagent_gpt4o_mini
```

This writes:

```text
runs/finagent_gpt4o_mini/aggregate/combined.pkl
runs/finagent_gpt4o_mini/aggregate/summary.csv
runs/finagent_gpt4o_mini/aggregate/summary.json
```

## Smoke Test

Before submitting the full array, generate a small manifest manually and run one task interactively:

```bash
python slurm/finagent_one_window.py \
  --setup selected_4 \
  --date-from 2004-01-01 \
  --date-to 2006-01-01 \
  --strat-config-path strats_configs/finagent_config_normal.json \
  --tickers TSLA \
  --model-id gpt-4o-mini \
  --output-root runs/finagent_gpt4o_mini_smoke
```
