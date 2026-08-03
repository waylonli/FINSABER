# TradingAgents FINSABER-2 Experiments

This workflow evaluates `TradingAgentsStrategy` with the FINSABER-2 parquet
dataset over the 2024 and 2025 calendar-year windows. The versioned manifest
fixes ticker selections, model, data modalities, execution assumptions, random
seed, and TradingAgents artifact policy.

The official profile uses OpenAI's Responses API with `temperature=0.0`.
The benchmark request seed is `42`, but it is explicitly recorded as not
applied because the Responses API used by this environment does not support a
request `seed`. The experiment seed is `2026`; it controls Python, NumPy, and
the worker interpreter's `PYTHONHASHSEED`.

The formal FINSABER-2 entry point is:

```text
examples/experiments/run_tradingagents_finsaber2.py
```

The older `run_llm_traders_exp.py` path is retained for legacy cherry-pick
experiments, but it is not the recommended path for the manifest-aligned
FINSABER-2 baseline.

## Prerequisites

Run all commands from the repository root.

### Data And Secrets

Use a Python 3.10 conda environment, configure `OPENAI_API_KEY` in an ignored
`.env` file, and pass the local FINSABER-2 dataset root explicitly. The
expected dataset folders are `price_daily/`, `news_items/`, `filingk/`, and
`filingq/`.

The local dataset root used during validation was:

```text
/mnt/cbs1/data/datasets/sp500_2000_2025_parquet
```

Use the real path for the machine running the experiment. The runner also
accepts `FINSABER_DATA_ROOT` when `--data-root` is omitted, but passing
`--data-root` explicitly is preferred for reproducibility.

### Environment

Create the environment from the FINSABER-2 package dependencies plus the
TradingAgents add-on package:

```bash
conda create -n finsaber-ta310 python=3.10 pip -y
conda activate finsaber-ta310

python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev]"
python -m pip install -e ./llm_traders/tradingagent
python -m pip check
```

The `dev` extra is only needed for the validation tests below. For a run-only
environment, `python -m pip install -e .` is sufficient before installing the
TradingAgents package. The checked-in `finsaber_ta310.yml` is a local snapshot
for reference; it may be less portable than the commands above because it
contains conda build pins.

The runner loads `.env` automatically. If only `OPENAI_API_KEY` is set, it is
also copied into `OA_OPENAI_KEY` for the TradingAgents internals.

### Validation

Validate the environment before spending API budget:

```bash
conda run -n finsaber-ta310 python -m py_compile \
  examples/experiments/run_tradingagents_finsaber2.py \
  tests/test_tradingagents_finsaber2_runner.py

conda run -n finsaber-ta310 python -m pytest -q \
  tests/test_tradingagents_finsaber2_runner.py \
  tests/test_tradingagents_experiment_launcher.py
```

Preview the complete plan without running TradingAgents or calling OpenAI:

```bash
conda run -n finsaber-ta310 python examples/experiments/run_tradingagents_finsaber2.py \
  --plan \
  --data-root /path/to/sp500_2000_2025_parquet
```

The installation commands above were validated from scratch in a temporary
conda environment with `pip check`, the two test commands, and the full
manifest `--plan`. These validation steps do not call OpenAI.

### Quick Run Commands

Preview one ticker-year job:

```bash
conda run -n finsaber-ta310 python examples/experiments/run_tradingagents_finsaber2.py \
  --plan \
  --setups selected_4 \
  --windows 2025-01-01_2026-01-01 \
  --tickers COIN \
  --data-root /path/to/sp500_2000_2025_parquet
```

`--tickers` accepts one or more symbols and filters the selected setup/window
jobs. This is useful for overlap cohorts such as `magnificent_7`, where only
the not-yet-run tickers may need to be launched.

Run one ticker-year job through the resumable orchestrator:

```bash
conda run --no-capture-output -n finsaber-ta310 \
  python examples/experiments/run_tradingagents_finsaber2.py \
  --setups selected_4 \
  --windows 2025-01-01_2026-01-01 \
  --tickers COIN \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root tmp/tradingagents-selected4-coin-2025-r1
```

`--no-capture-output` is recommended for long `conda run` jobs because it lets
progress lines stream to the terminal or tmux pane in real time.

Run one setup sequentially:

```bash
conda run --no-capture-output -n finsaber-ta310 \
  python examples/experiments/run_tradingagents_finsaber2.py \
  --setups selected_4 \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root tmp/tradingagents-selected4-2024-2026-r1 \
  --max-parallel 1
```

Run the full public manifest on this local machine:

```bash
conda run --no-capture-output -n finsaber-ta310 \
  python examples/experiments/run_tradingagents_finsaber2.py \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root backtest/output/tradingagents_finsaber2_2024_2026 \
  --max-parallel 2 \
  --job-timeout-hours 12
```

## Manifest

The source manifest is:

```text
examples/experiments/manifests/tradingagents_finsaber2_2024_2026.json
```

It uses the same frozen ticker selections as the FinAgent and FinMem
FINSABER-2 manifests:

- `selected_4`
- `random_sp500_5`
- `momentum_sp500_5`
- `lowvol_sp500_5`
- `magnificent_7`

The manifest expands to 54 jobs. A job is identified by:

```text
setup / window / ticker
```

For example, `selected_4` has five tickers across two windows, so it expands to
10 jobs. The `selected_4` name is historical; in this manifest it intentionally
contains the five cherry-pick FinMem tickers.

Unlike FinMem, TradingAgents does not have an explicit train/test split. Each
job runs directly over the requested test window, and TradingAgents updates its
internal memory and reflection state inside that window.

| Window Key | Effective Test Window |
| --- | --- |
| `2024-01-01_2025-01-01` | `2024-01-02` to `2024-12-31` |
| `2025-01-01_2026-01-01` | `2025-01-02` to `2025-12-31` |

The current local-data policy is:

- local market data only;
- local ticker news only;
- local 10-K and 10-Q filings only;
- no online data fallback;
- benchmark ticker `SPY` is declared but may be unavailable if it is absent
  from the local parquet price universe.

TradingAgents uses broader filing sections than FinMem:

| Form | Section |
| --- | --- |
| `10-K` | `item_1` |
| `10-K` | `item_1a` |
| `10-K` | `item_7` |
| `10-K` | `item_8` |
| `10-Q` | `part_i_item_1` |
| `10-Q` | `part_i_item_2` |
| `10-Q` | `part_ii_item_1a` |

Filing section extraction uses the shared FINSABER filing section extractor. If
a requested section cannot be safely recovered from the clean parquet text, the
extractor should leave it unavailable rather than fabricate content.

## Parallel Execution

A runner is one invocation of `examples/experiments/run_tradingagents_finsaber2.py`.
The runner expands the manifest into `setup/window/ticker` jobs and schedules
those jobs through one global worker pool.

`--max-parallel` controls how many ticker-year jobs may run at the same time.
It does not create setup-level or window-level worker pools. If multiple setups
and windows are selected, the runner still uses one global job queue ordered by
`setup -> window -> ticker`.

For this local machine, use `--max-parallel 2`. A short probe with
`--max-parallel 3` started correctly but put too much pressure on memory. On a
larger machine, increase parallelism only after checking CPU, memory, OpenAI
rate limits, and budget.

Recommended: use one runner to coordinate multiple setups and windows:

```bash
conda run --no-capture-output -n finsaber-ta310 \
  python examples/experiments/run_tradingagents_finsaber2.py \
  --setups selected_4 random_sp500_5 momentum_sp500_5 lowvol_sp500_5 magnificent_7 \
  --windows 2024-01-01_2025-01-01 2025-01-01_2026-01-01 \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root backtest/output/tradingagents_finsaber2_2024_2026 \
  --max-parallel 2
```

Do not run multiple runners against the same `--output-root` at the same time.
Per-ticker directories include the setup name, but top-level files such as
`runner_manifest.json` and `experiment_config.json` would be overwritten by
competing runner processes.

If multiple tmux sessions are necessary, give each runner a separate output
root:

```bash
# tmux A
conda run --no-capture-output -n finsaber-ta310 \
  python examples/experiments/run_tradingagents_finsaber2.py \
  --setups selected_4 \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root backtest/output/tradingagents_selected4 \
  --max-parallel 1
```

```bash
# tmux B
conda run --no-capture-output -n finsaber-ta310 \
  python examples/experiments/run_tradingagents_finsaber2.py \
  --setups random_sp500_5 \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root backtest/output/tradingagents_random5 \
  --max-parallel 1
```

## Job Timeout

`--job-timeout-hours` is a per-job timeout. One job means one
`setup/window/ticker` backtest over one window. The default is
`--job-timeout-hours 12`.

If a valid full-year job exceeds the timeout, rerun with a larger value. Jobs
that already have `metrics.json` are skipped, so completed jobs do not call
TradingAgents or OpenAI again.

## Resume Semantics

Resume is job-level, not trading-day-level. A job is considered complete when
its scalar `metrics.json` exists under the ticker output directory. Completed
jobs are skipped on restart.

Incomplete jobs are rerun from scratch. Before rerunning an incomplete job, the
runner removes that job's stale standard ticker output directory and that
job's deterministic TradingAgents private artifact run directory. This prevents
partial CSVs, memory logs, full-state logs, runtime cache files, or JSONL traces
from being mixed with the replacement run.

Worker stdout/stderr logs are appended across restarts. Seeing pre-interruption
and post-resume logs in the same ticker log file is expected and useful for
auditing.

## Output Structure

Each output root contains top-level orchestration artifacts:

```text
<output_root>/
  experiment_config.json
  runner_manifest.json
  logs/<setup>/<window>/<ticker>.stdout.log
  logs/<setup>/<window>/<ticker>.stderr.log
```

`runner_manifest.json` records the source manifest hash, Git commit, resolved
data/output roots, LLM sampling contract, full ticker selections, selected job
list, per-job status, completion counts, `max_parallel`, and
`job_timeout_hours`.

The runner writes standard FINSABER result artifacts under:

```text
<output_root>/<setup>/TradingAgentsStrategy/
  run_config.json
  run_manifest.json
  run_summary.csv
  <test_start>_<test_end>/<ticker>/
    metrics.json
    metrics.pkl
    job_status.json
    equity_curve.csv
    trades.csv
    rejected_orders.csv
    llm_costs.csv
    external_costs.csv
```

Use `run_summary.csv` for setup-level scalar results. Use each ticker leaf
directory for the raw per-job files. `metrics.pkl` preserves the complete
metrics object, including DataFrames, while `metrics.json` is the scalar
completion sentinel used for resume.

At the end of orchestration, the runner rebuilds setup-level standard outputs
from the selected jobs in the current invocation.

## TradingAgents Artifacts

The public manifest keeps `artifact_config.enabled=true` by default. This is
recommended for official audit runs because TradingAgents evidence is important
for interpreting decisions, memory, and reflection behavior.

If no explicit artifact root is provided, the runner places private
TradingAgents artifacts under the strategy output tree:

```text
<output_root>/<setup>/TradingAgentsStrategy/
  <profile_id>/tradingagents_artifacts/<config_key>/<run_key>/
    manifest.json
    namespace_meta.json
    runtime_cache/
    runtime_results/
    tickers/<ticker>/
      ticker_namespace_meta.json
      full_state_logs/full_states_log_<date>.json
      memory/trading_memory.md
      analyst_input_trace.jsonl
      memory_reads.jsonl
      memory_writes.jsonl
      reflection_trace.jsonl
```

`artifact_config.enabled=false` is not a full artifact-off mode. It disables
the optional JSONL trace writer files, but TradingAgents still creates core
runtime directories, namespace metadata, memory files, and full-state logs that
the strategy needs to operate.

Official experiments should generally keep artifacts enabled and plan disk
space accordingly.

## Cost And Runtime Planning

Historical full-year TradingAgents cherry-pick runs provide the best local
cost anchor:

| Run | Jobs | Total LLM Cost | Average Cost Per Job |
| --- | ---: | ---: | ---: |
| 2024 selected five tickers | 5 | about `$24.93` | about `$4.99` |
| 2025 selected five tickers | 5 | about `$23.65` | about `$4.73` |
| Combined observed average | 10 | about `$48.57` | about `$4.86` |

The full public manifest expands to 54 jobs. A neutral cost estimate is:

```text
54 * $4.86 ~= $260
```

Use `$300-$350` as a practical budget buffer, and `$350-$400` as a conservative
upper bound. Parallelism changes wall-clock time, not total API cost, unless it
causes failed or repeated jobs.

On the current local machine, `--max-parallel 2` is the recommended default.
Actual full-manifest runtime is expected to be measured in days rather than
hours.

## Tmux Background Execution

Example full-manifest launch:

```bash
tmux new-session -d -s ta_finsaber2_full '
cd /path/to/FINSABER &&
mkdir -p backtest/output &&
conda run --no-capture-output -n finsaber-ta310 \
  python examples/experiments/run_tradingagents_finsaber2.py \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root backtest/output/tradingagents_finsaber2_2024_2026 \
  --max-parallel 2 \
  --job-timeout-hours 12 \
  2>&1 | tee backtest/output/tradingagents_finsaber2_2024_2026.log
'
```

Useful monitoring commands:

```bash
tmux ls
tmux capture-pane -pt ta_finsaber2_full | tail -n 80
tail -f backtest/output/tradingagents_finsaber2_2024_2026.log
```

The runner prints `PROGRESS` lines with completed, failed, running, pending,
elapsed time, active worker PIDs, and observed `full_state_logs` counts.

## Validation Checklist

After launching a run, validate:

1. `experiment_config.json` and `runner_manifest.json` exist under the output
   root.
2. `logs/<setup>/<window>/<ticker>.stdout.log` and `.stderr.log` are created.
3. Active jobs print `PROGRESS` lines in the terminal or tmux pane.
4. `full_state_logs/full_states_log_<date>.json` begins to populate under the
   private TradingAgents artifact tree.
5. A completed ticker has `metrics.json`, `equity_curve.csv`, `trades.csv`,
   `llm_costs.csv`, and `job_status.json` under its standard result directory.
6. Setup-level `run_summary.csv` is rebuilt after orchestration completes.

Fast checks:

```bash
conda run -n finsaber-ta310 python examples/experiments/run_tradingagents_finsaber2.py \
  --plan \
  --data-root /path/to/sp500_2000_2025_parquet
```

```bash
find backtest/output/tradingagents_finsaber2_2024_2026 \
  -path '*/job_status.json' -print | head
```

```bash
find backtest/output/tradingagents_finsaber2_2024_2026 \
  -path '*/full_state_logs/full_states_log_*.json' -print | head
```

## Known Limits

Hosted LLM output is not bitwise reproducible because provider-side model
revisions and sampling can change. The Responses API does not apply the
manifest's requested seed; its status is preserved in `llm_sampling` rather
than being reported as effective. Preserve the complete output directory when
using artifacts for audit.

TradingAgents does not provide trading-day-level checkpoint resume in this
runner. An incomplete ticker-year job is rerun from the beginning.

Very short real smoke windows are rejected by the FINSABER ISO backtest engine
when fewer than 21 trading days are available. Use `--plan` for no-cost static
checks, and use a 21+ trading-day window for an end-to-end smoke.

`SPY` benchmark alpha may be unavailable when `SPY` is absent from the local
FINSABER-2 parquet price universe. The runner does not add online or sidecar
benchmark data.

The older `run_llm_traders_exp.py` launcher still exists, but its setup catalog
and output ownership differ from the manifest-aligned FinAgent and FinMem
FINSABER-2 workflows.

## Current Status

The dedicated TradingAgents launcher supports `--plan`, resumable multi-job
orchestration, per-job worker status files, stdout/stderr logs, deterministic
job-scoped TradingAgents artifact roots, progress reporting, and standard
FINSABER result summaries.

The latest validation confirmed:

- the default manifest expands to 54 jobs;
- artifact paths are isolated by setup, window, and ticker;
- interrupted incomplete jobs are rerun from scratch with stale standard and
  private artifact outputs removed first;
- completed jobs are skipped by `metrics.json`;
- TA runner tests and legacy TA launcher tests pass in the existing
  `finsaber-ta310` environment and in a fresh doc-check environment created
  from the installation commands above;
- a fresh doc-check environment completed an end-to-end AMZN smoke over a 21+
  trading-day 2025 window with standard FINSABER outputs and TradingAgents
  memory/reflection artifacts.
