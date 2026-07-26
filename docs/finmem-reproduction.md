# FinMem FINSABER-2 Experiments

This workflow evaluates FinMem with the FINSABER-2 parquet dataset over the
2024 and 2025 calendar-year windows. The versioned manifest fixes ticker
selections, explicit FinMem training windows, model, data modalities, execution
assumptions, and the FinMem TOML configuration.

## Prerequisites

Use the `finsaber2-finmem` conda environment, configure `OPENAI_API_KEY` in an
ignored `.env` file, and pass the local FINSABER-2 dataset root explicitly. The
expected dataset folders are `price_daily/`, `news_items/`, `filingk/`, and
`filingq/`.

Create the environment from the FINSABER-2 project dependencies plus the
FinMem add-on dependencies:

```bash
conda create -n finsaber2-finmem python=3.10 -y
conda activate finsaber2-finmem

python -m pip install -U pip "wheel<0.46"
pip install -r llm_traders/finmem/requirements.txt
pip install -e .

# Optional: needed by the non-LLM FINSABER-2 benchmark launcher.
pip install "statsmodels>=0.14" "xgboost>=2.0" "pandas-datareader>=0.10"

python -m pip check
```

`requirements-complete.txt` is an identical pinned snapshot across the
`reproduce`, `main`, and `finmem` branches. It can be used as a compatibility
reference, but the commands above document the intended FinMem-on-FINSABER-2
installation path.

Do not use `pip install -e ".[dev,research]"` for the FinMem environment. The
full `research` extra currently pulls newer Hugging Face dependencies that
conflict with FinMem's pinned `transformers` stack. The selective optional
installs above keep FinMem and the standard non-LLM benchmark launcher working
in the same environment.

Preview the complete plan without running FinMem or calling OpenAI:

```bash
conda run -n finsaber2-finmem python examples/experiments/run_finmem_finsaber2.py \
  --plan \
  --data-root /path/to/sp500_2000_2025_parquet
```

Preview one ticker-year job:

```bash
conda run -n finsaber2-finmem python examples/experiments/run_finmem_finsaber2.py \
  --plan \
  --setups selected_4 \
  --windows 2025-01-01_2026-01-01 \
  --tickers COIN \
  --data-root /path/to/sp500_2000_2025_parquet
```

Run one ticker-year job through the resumable orchestrator:

```bash
conda run -n finsaber2-finmem python examples/experiments/run_finmem_finsaber2.py \
  --setups selected_4 \
  --windows 2025-01-01_2026-01-01 \
  --tickers COIN \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root tmp/finmem-selected4-coin-2025-r1
```

Run one setup sequentially:

```bash
conda run -n finsaber2-finmem python examples/experiments/run_finmem_finsaber2.py \
  --setups selected_4 \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root tmp/finmem-selected4-2024-2026-r1 \
  --max-parallel 1
```

## Manifest

The source manifest is:

```text
examples/experiments/manifests/finmem_finsaber2_2024_2026.json
```

It uses the same frozen ticker selections as the FinAgent FINSABER-2 manifest:

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
10 jobs.

FinMem uses explicit three-year training windows:

| Test Window | Training Window |
| --- | --- |
| `2024-01-01_2025-01-01` | `2021-01-04` to `2023-12-31` |
| `2025-01-01_2026-01-01` | `2022-01-03` to `2024-12-31` |

The current filing section mapping is:

| Modality | Form | Item |
| --- | --- | --- |
| `filing_k` | `10-K` | `item_7` |
| `filing_q` | `10-Q` | `part_i_item_2` |

Filing section extraction uses the shared FINSABER filing section extractor. If
a requested filing section cannot be safely recovered from the clean parquet
text, the configured FinMem failure mode is `empty`; the runner should not
fabricate section text.

## Parallel Execution

A runner is one invocation of `examples/experiments/run_finmem_finsaber2.py`.
The runner expands the manifest into `setup/window/ticker` jobs and schedules
those jobs through one global worker pool.

`--max-parallel` controls how many jobs may run at the same time. It does not
control setup-level parallelism. If multiple setups and windows are selected,
the runner still uses one global job queue ordered by `setup -> window ->
ticker`.

For local development, prefer `--max-parallel 1` or `--max-parallel 2`. On a
larger machine, increase it only after checking CPU, memory, and OpenAI API rate
limits.

Recommended: use one runner to coordinate multiple setups and windows:

```bash
conda run -n finsaber2-finmem python examples/experiments/run_finmem_finsaber2.py \
  --setups selected_4 random_sp500_5 momentum_sp500_5 lowvol_sp500_5 magnificent_7 \
  --windows 2024-01-01_2025-01-01 2025-01-01_2026-01-01 \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root backtest/output/finmem_finsaber2_2024_2026 \
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
conda run -n finsaber2-finmem python examples/experiments/run_finmem_finsaber2.py \
  --setups selected_4 \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root backtest/output/finmem_selected4 \
  --max-parallel 1
```

```bash
# tmux B
conda run -n finsaber2-finmem python examples/experiments/run_finmem_finsaber2.py \
  --setups random_sp500_5 \
  --data-root /path/to/sp500_2000_2025_parquet \
  --output-root backtest/output/finmem_random5 \
  --max-parallel 1
```

## Resume Semantics

Resume is job-level, not trading-day-level. A job is considered complete when
its scalar `metrics.json` exists under the ticker output directory. Completed
jobs are skipped on restart and do not call FinMem or OpenAI again.

Incomplete jobs are rerun from scratch. Before rerunning an incomplete job, the
runner removes that job's stale ticker output directory and, when FinMem
artifacts are enabled, that job's stale artifact ticker directory. This prevents
partial CSVs, traces, or checkpoints from being mixed with the replacement run.

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
data/output roots, full ticker selections, selected job list, per-job status,
and completion counts.

The runner also writes standard FINSABER result artifacts under:

```text
<output_root>/<setup>/FinMemStrategy/
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

`run_summary.csv` is rebuilt from the selected jobs in the current invocation,
so filtered reruns do not accidentally summarize unrelated artifacts that
happen to share the same output root.

## FinMem Artifacts

The manifest currently keeps `artifact_config.enabled=false` by default.
However, official audit runs should consider enabling FinMem artifacts so the
experiment preserves query traces, LLM traces, reflections, and post-train/test
state checkpoints.

Artifact capture can add meaningful storage overhead because checkpoints and
state pickles are written per ticker job. Plan disk space before enabling it for
large multi-setup runs.

If artifacts are enabled and no explicit artifact root is provided, the runner
places strategy-local artifacts under:

```text
<output_root>/<setup>/FinMemStrategy/finmem_artifacts/
  <config_key>/<run_key>/tickers/<ticker>/
```

The artifact ticker directory may include files such as:

```text
manifest.json
train_query_trace.jsonl
train_llm_trace.jsonl
train_reflections.jsonl
test_query_trace.jsonl
test_llm_trace.jsonl
test_reflections.jsonl
post_train/
test_state/
```

If `artifact_config.root` is set explicitly, do not share the same root across
concurrent experiments unless the resulting `config_key/run_key/ticker` paths
are known not to collide.

## Known Limits

Hosted LLM output is not bitwise reproducible because provider-side model
revisions and sampling can change. Preserve the complete output directory when
using artifacts for audit.

The runner does not provide trading-day-level checkpoint resume. An incomplete
ticker job is rerun from scratch and may incur additional LLM cost.

Do not run multiple runner processes into the same output root concurrently.
Use one runner with `--max-parallel`, or use separate output roots per runner.

FinMem can use substantial CPU and memory even for one ticker. Increase
`--max-parallel` cautiously, especially when artifacts are enabled.

## Current Status

The dedicated FinMem launcher supports `--plan`, resumable multi-job
orchestration, per-job worker status files, stdout/stderr logs, optional FinMem
artifact capture, and standard FINSABER result summaries. It intentionally keeps
FinMem strategy logic and FINSABER core execution unchanged.

The latest plan validation confirmed the manifest, model/TOML match, filing
section map, dataset root, and all 54 planned jobs. The launcher has also been
validated with no-artifact resume and artifact-enabled interrupted resume smoke
runs.
