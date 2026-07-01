# TradingAgents Reproduction Guide

## Purpose and Scope

This document explains how to run `TradingAgentsStrategy` through the formal
FINSABER experiment launcher.

It focuses on the current launcher path:

- `examples/experiments/run_llm_traders_exp.py`
- `examples/experiments/experiment_runner.py`
- `backtest.finsaber.FINSABER`

This is an operator-facing runbook. It does not explain the internal
TradingAgents implementation, prompt design, or research history.

This guide uses a checked-in environment snapshot file:

- `finsaber_ta310.yml`

That file is intended to reproduce the currently validated `finsaber-ta310`
dependency environment quickly. It is not a hand-curated minimal dependency
specification.

## Supported Launcher Setups

The current TradingAgents launcher path inherits its setup catalog from
`ExperimentRunner`. At the time of writing, the formal `--setup` values are:

| Setup name | Execution mode | Ticker source | Notes |
| --- | --- | --- | --- |
| `cherry_pick_both_finmem` | `iter` | Fixed 5 tickers | Canonical TradingAgents benchmark path |
| `cherry_pick_both_fincon` | `iter` | Fixed 8 tickers | Alternate fixed-basket benchmark |
| `selected_4` | `rolling_window` | Legacy fixed list | Currently resolves to the same 5-ticker list as `selected_5`; planned to be replaced by a `magnificent_7` fixed basket in a future update |
| `selected_5` | `rolling_window` | Legacy fixed list | Fixed list: `TSLA`, `NFLX`, `AMZN`, `MSFT`, `COIN` |
| `random_sp500_<N>` | `rolling_window` | Dynamic SP500 selection | Random yearly selector |
| `momentum_sp500_<N>` | `rolling_window` | Dynamic SP500 selection | Momentum selector |
| `lowvol_sp500_<N>` | `rolling_window` | Dynamic SP500 selection | Low-volatility selector |
| `fincon_selector_sp500_<N>` | `rolling_window` | Dynamic SP500 selection | FinCon-based selector |

Important details:

- `cherry_pick_both_finmem` is the most thoroughly validated TradingAgents path
  in this repository.
- `selected_4` is a legacy setup name. In the current code it still resolves to
  the same fixed five-ticker list as `selected_5`.
- The intended future direction is to retire `selected_4` and reuse that
  fixed-basket slot for a `magnificent_7` cohort, but that change has not been
  applied to `ExperimentRunner` yet.
- `iter` setups and `rolling_window` setups do not behave the same way. Runtime,
  ticker selection, and output interpretation differ.

## Future Cohorts and Manifest-Driven Selections

Some experiment families in this repository already use manifest-driven cohort
definitions. For example, FinAgent uses:

- `examples/experiments/manifests/finagent_finsaber2_2024_2026.json`

That manifest includes a `magnificent_7` cohort.

TradingAgents does **not** currently expose `magnificent_7` as a first-class
`--setup` in `ExperimentRunner`. If you want to support that cohort later, add a
dedicated launcher setup or introduce a TradingAgents-specific manifest runner.

The expected direction for this repository is that the old `selected_4`
fixed-basket slot will eventually be replaced by `magnificent_7`, but this is
still a planning target rather than the current launcher behavior.

This guide therefore documents the setups that are formally supported by the
current TradingAgents launcher path today.

## Prerequisites

Before launching a TradingAgents experiment, confirm:

1. A conda environment recreated from `finsaber_ta310.yml` exists and is usable.
2. `OPENAI_API_KEY` is available through the shell or a repo-local `.env`.
3. The FINSABER-2 parquet dataset is available locally.
4. `tmux` is installed if you want a background session.

Expected dataset folders under `--data_root`:

- `price_daily/`
- `news_items/`
- `filingk/`
- `filingq/`

Example local dataset root used in this project:

```bash
./data/sp500_2000_2025_parquet
```

Example interpreter choices:

```bash
/opt/anaconda3/envs/finsaber-ta310/bin/python
/opt/anaconda3/envs/finsaber-ta310-test/bin/python
```

## Environment Setup

The fastest way to reproduce the validated TradingAgents runtime on another
machine is to recreate the current `finsaber-ta310` environment from the
checked-in snapshot file and then reinstall the local repo packages in editable
mode.

### Option A: reproduce the default environment name

```bash
conda env create -f finsaber_ta310.yml
conda activate finsaber-ta310
```

### Option B: reproduce the same environment under a temporary name

This is useful for local verification without touching an existing
`finsaber-ta310`:

```bash
conda env create -f finsaber_ta310.yml -n finsaber-ta310-test
conda activate finsaber-ta310-test
```

### Rebind the environment to the current repo checkout

After the conda environment is created, reinstall the two repo-local packages
from the current checkout:

```bash
python -m pip install -e . --no-deps
python -m pip install -e llm_traders/tradingagent --no-deps
```

This matters because the snapshot file only recreates the dependency
environment. The benchmark itself should run against the current local checkout,
not against any previously published package build.

### Minimal validation gate

The following gate was re-run successfully against a fresh
`finsaber-ta310-test` environment created from `finsaber_ta310.yml`:

```bash
python -m pytest -q \
  tests/test_tradingagents_experiment_launcher.py \
  llm_traders/tradingagent/tests/test_tradingagents_offline_session_adapter.py
```

If this gate passes, the environment is ready for the formal launcher path
documented below.

### Real launcher validation status

This guide was revalidated against a fresh `finsaber-ta310-test` environment.
A real `tmux`-launched one-month `cherry_pick_both_finmem` TradingAgents run:

- started successfully through `run_llm_traders_exp.py`
- materialized a run root under `tradingagents_artifacts/`
- created `launcher/run.log`, `manifest.json`, and `namespace_meta.json`
- created ticker-local traces such as `analyst_input_trace.jsonl`
- produced `tickers/TSLA/full_state_logs/full_states_log_2024-01-02.json`

That confirms the current environment snapshot and launcher path are usable for
real TradingAgents execution, not only for unit tests.

## Launcher Entry Point

The formal user-facing entry point is:

```bash
python examples/experiments/run_llm_traders_exp.py
```

This script:

1. selects a strategy via `--strategy`
2. selects an experiment family via `--setup`
3. loads a TradingAgents strategy config via `--strat_config_path`
4. forwards the resolved request into `ExperimentRunner`
5. sets `MPLBACKEND=Agg` to avoid GUI-blocking matplotlib behavior

For TradingAgents, the most important arguments are:

- `--setup`
- `--strategy tradingagents`
- `--strat_config_path`
- `--date_from`
- `--date_to`
- `--data_root`
- `--output_dir`

## TradingAgents Strategy Config Files

The checked-in TradingAgents window configs are:

- `strats_configs/tradingagents_window_2024.json`
- `strats_configs/tradingagents_window_2025.json`

These files are currently minimal and mainly provide:

- `date_from`
- `date_to`
- `symbol`
- `artifact_config`

The most important field for runtime output ownership is:

```json
"artifact_config": {
  "enabled": true,
  "root": "...",
  "run_key": null
}
```

### Important Output Ownership Warning

For `TradingAgentsStrategy`, the real run root is derived from
`artifact_config.root`, not only from `--output_dir`.

This means:

- if you run `cherry_pick_both_finmem` with the checked-in 2024/2025 configs,
  the output root matches the expected cherry-pick tree
- if you reuse those same config files for another setup such as `selected_4` or
  `random_sp500_5`, the default artifact root will still point into the
  `cherry_pick_both_finmem` output tree unless you change it

For non-cherry setups, the safest practice is:

1. copy an existing TradingAgents config file
2. rename it for the intended setup
3. change `artifact_config.root` to a setup-appropriate output tree

### Concurrency note for the checked-in 2024 and 2025 configs

The checked-in 2024 and 2025 TradingAgents configs already point to different
artifact roots:

- `tradingagents_window_2024.json`
  -> `.../tradingagents_window_2024/tradingagents_artifacts`
- `tradingagents_window_2025.json`
  -> `.../tradingagents_window_2025/tradingagents_artifacts`

That means the two canonical one-month cherry-pick runs can be launched at the
same time without writing into the same artifact tree.

In addition, the default `artifact_config.run_key` is `null`, so each launch
receives a fresh materialized `run_key`. Even repeated launches inside the same
window config fan out into separate run directories unless you explicitly force
the same `run_key`.

## Canonical Cherry-Pick Recipes

These are the canonical one-month TradingAgents launcher commands for the
formal `cherry_pick_both_finmem` benchmark path.

### 2024 One-Month Window

```bash
python examples/experiments/run_llm_traders_exp.py \
  --setup cherry_pick_both_finmem \
  --strategy tradingagents \
  --strat_config_path strats_configs/tradingagents_window_2024.json \
  --date_from 2024-01-02 \
  --date_to 2024-01-31 \
  --data_root ./data/sp500_2000_2025_parquet \
  --output_dir backtest/output
```

### 2025 One-Month Window

```bash
python examples/experiments/run_llm_traders_exp.py \
  --setup cherry_pick_both_finmem \
  --strategy tradingagents \
  --strat_config_path strats_configs/tradingagents_window_2025.json \
  --date_from 2025-01-02 \
  --date_to 2025-01-31 \
  --data_root ./data/sp500_2000_2025_parquet \
  --output_dir backtest/output
```

Notes:

- `cherry_pick_both_finmem` is a fixed 5-ticker setup.
- Changing the date window does not turn it into a single-ticker run.
- The fixed basket is currently:
  - `TSLA`
  - `NFLX`
  - `AMZN`
  - `MSFT`
  - `COIN`

## Running Other TradingAgents Setup Families

The same launcher path can be reused for other setup families, but you should
prepare a setup-appropriate TradingAgents config first.

Recommended workflow:

1. copy one of the window configs
2. rename it for the target setup
3. change `artifact_config.root`
4. run with the desired `--setup`

Example command template:

```bash
python examples/experiments/run_llm_traders_exp.py \
  --setup <setup_name> \
  --strategy tradingagents \
  --strat_config_path <your_tradingagents_config>.json \
  --date_from <date_from> \
  --date_to <date_to> \
  --data_root ./data/sp500_2000_2025_parquet \
  --output_dir backtest/output
```

Examples of valid `setup_name` values:

- `selected_5`
- `random_sp500_5`
- `momentum_sp500_5`
- `lowvol_sp500_5`
- `fincon_selector_sp500_5`

Treat these as advanced runs until you have verified:

- the intended execution mode
- the selected ticker behavior
- the output root

## Tmux Background Execution

Recommended `tmux` launch for the 2024 one-month cherry-pick run:

```bash
REPO_ROOT=/path/to/FINSABER
PYTHON_BIN=/opt/anaconda3/envs/finsaber-ta310/bin/python
# replace with /opt/anaconda3/envs/finsaber-ta310-test/bin/python when validating
# a freshly recreated environment

tmux new-session -d -s ta_cp_finmem_2024_01 '
cd '"$REPO_ROOT"' &&
'"$PYTHON_BIN"' -u examples/experiments/run_llm_traders_exp.py \
  --setup cherry_pick_both_finmem \
  --strategy tradingagents \
  --strat_config_path strats_configs/tradingagents_window_2024.json \
  --date_from 2024-01-02 \
  --date_to 2024-01-31 \
  --data_root ./data/sp500_2000_2025_parquet \
  --output_dir backtest/output \
  2>&1 | tee backtest/output/ta_cp_finmem_2024_01.log
'
```

Useful `tmux` commands:

```bash
tmux ls
tmux attach -t ta_cp_finmem_2024_01
tmux capture-pane -pt ta_cp_finmem_2024_01 | tail -n 40
tmux kill-session -t ta_cp_finmem_2024_01
```

## Parallel Tmux Execution for 2024 and 2025

If your API quota allows it, the 2024 and 2025 one-month cherry-pick runs can
be launched in parallel with two separate `tmux` sessions.

Example:

```bash
REPO_ROOT=/path/to/FINSABER
PYTHON_BIN=/opt/anaconda3/envs/finsaber-ta310/bin/python
# or:
# PYTHON_BIN=/opt/anaconda3/envs/finsaber-ta310-test/bin/python

tmux new-session -d -s ta_cp_finmem_2024_01 '
cd '"$REPO_ROOT"' &&
'"$PYTHON_BIN"' -u examples/experiments/run_llm_traders_exp.py \
  --setup cherry_pick_both_finmem \
  --strategy tradingagents \
  --strat_config_path strats_configs/tradingagents_window_2024.json \
  --date_from 2024-01-02 \
  --date_to 2024-01-31 \
  --data_root ./data/sp500_2000_2025_parquet \
  --output_dir backtest/output \
  2>&1 | tee backtest/output/ta_cp_finmem_2024_01.log
'

tmux new-session -d -s ta_cp_finmem_2025_01 '
cd '"$REPO_ROOT"' &&
'"$PYTHON_BIN"' -u examples/experiments/run_llm_traders_exp.py \
  --setup cherry_pick_both_finmem \
  --strategy tradingagents \
  --strat_config_path strats_configs/tradingagents_window_2025.json \
  --date_from 2025-01-02 \
  --date_to 2025-01-31 \
  --data_root ./data/sp500_2000_2025_parquet \
  --output_dir backtest/output \
  2>&1 | tee backtest/output/ta_cp_finmem_2025_01.log
'
```

Recommended monitoring commands:

```bash
tmux capture-pane -pt ta_cp_finmem_2024_01 | tail -n 40
tmux capture-pane -pt ta_cp_finmem_2025_01 | tail -n 40
```

Important note:

- parallel execution is structurally safe for these two checked-in configs
  because their artifact roots differ
- the practical limiting factor is usually API throughput, rate limits, or
  spend, not filesystem collision

## Output Layout

For TradingAgents, the benchmark output and strategy-local runtime artifacts are
anchored to the same materialized run identity.

The practical run-root shape is:

```text
<artifact_root>/<config_key>/<run_key>/
  benchmark_results/
  launcher/
    run.sh
    run.log
    strat_config.materialized.json
  manifest.json
  namespace_meta.json
  runtime_cache/
  runtime_results/
  tickers/
    <TICKER>/
```

Key points:

- `benchmark_results/` holds the outer FINSABER benchmark outputs
- `tickers/<TICKER>/` holds TradingAgents strategy-local artifacts such as:
  - `full_state_logs/`
  - `reflection_trace.jsonl`
  - `analyst_input_trace.jsonl`
  - `memory/`
- `launcher/run.sh` is the run-local replay script
- `launcher/strat_config.materialized.json` freezes the run-local config snapshot
- `launcher/run.log` is a launcher metadata snapshot, not the main streaming
  progress log for the whole experiment

## Validation Checklist

After launching a run, validate the following:

1. The process starts without immediate Python import errors.
2. `launcher/run.log` is created under the materialized run root.
3. `tickers/<TICKER>/full_state_logs/` begins to populate.
4. `manifest.json` and `namespace_meta.json` exist at the run root.
5. `benchmark_results/.../metrics.json` appears after ticker completion.

Fast checks:

```bash
tail -f backtest/output/ta_cp_finmem_2024_01.log
```

```bash
find backtest/output/cherry_pick_both_finmem/TradingAgentsStrategy/tradingagents_window_2024/tradingagents_artifacts \
  -maxdepth 3 -type d | sort | tail -n 20
```

```bash
tmux capture-pane -pt ta_cp_finmem_2024_01 | tail -n 40
```

## Troubleshooting

### The run starts but no `full_state_logs` appear

Check:

- the latest materialized run root
- `launcher/run.log` to confirm the launcher started and to inspect the exact
  replay command
- `tmux capture-pane -pt <session>` for the live console stream
- `tickers/<TICKER>/analyst_input_trace.jsonl` to see whether the daily analyst
  loop is advancing before `full_state_logs` are emitted
- whether the process failed before entering the daily TradingAgents loop

### The output shows up under the wrong setup tree

Check `artifact_config.root` in the TradingAgents strat config. The launcher
does not automatically rewrite it based on `--setup`.

### `OPENAI_API_KEY` is present in `.env` but the run still fails

Verify that the current Python process actually sees the environment variable.
Using the environment Python directly is safer than relying on shell-local
activation state in long background jobs.

### A setup name works for FinAgent but not for TradingAgents

That usually means the cohort exists in a manifest-driven FinAgent workflow but
has not been added as a first-class `ExperimentRunner` setup for
TradingAgents yet.
