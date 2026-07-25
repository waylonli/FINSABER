# TEMPORARY - DO NOT COMMIT

# FinMem Experiment Entry And Output Layout Research

This is a temporary working note for designing the FinMem FINSABER-2 experiment
workflow. It is intentionally placed under `docs/` for active discussion, but it
should not be included in the final commit.

## Current Situation

The current FinMem workflow can run through:

```bash
python examples/experiments/run_llm_traders_exp.py \
  --setup cherry_pick_both_finmem \
  --strategy FinMemStrategy \
  --strat_config_path strats_configs/finmem_window_2025.json \
  --output_dir backtest/output/finmem_window_2025 \
  --date_from 2025-01-02 \
  --date_to 2025-12-31 \
  --data_root /mnt/cbs1/data/datasets/sp500_2000_2025_parquet
```

This path works, but it mixes three responsibilities:

- `run_llm_traders_exp.py` chooses the strategy and forwards CLI parameters.
- `ExperimentRunner` chooses setup tickers, default trade config, and FinMem
  run materialization.
- `strats_configs/finmem_window_*.json` controls FinMem training windows and
  artifact saving, including a hard-coded artifact root.

As a result, it is not obvious where an experiment is defined or where its
outputs should land.

## Current FinMem Output Semantics

When `artifact_config.enabled=true`, `ExperimentRunner` calls
`materialize_finmem_run_identity(...)` and rewrites the FINSABER
`result_output_dir` to the FinMem benchmark directory.

For the 2025 window, the current layout resolves to:

```text
backtest/output/finmem_window_2025/
  finmem_artifacts/
    finmem_gpt_config_<hash>/
      run_2025-01-02_2025-12-31/
        benchmark_results/
          results.csv
          run_summary.csv
          run_config.json
          run_manifest.json
          2025-01-02_2025-12-31/
            AMZN/
            COIN/
            MSFT/
            NFLX/
            TSLA/
        tickers/
          AMZN/
            manifest.json
            train_query_trace.jsonl
            train_llm_trace.jsonl
            test_query_trace.jsonl
            test_llm_trace.jsonl
            train_reflections.jsonl
            test_reflections.jsonl
            post_train/
            test_state/
          ...
```

This is internally coherent, but it differs from the generic FINSABER result
documentation, which describes:

```text
backtest/output/<setup_name>/<strategy_name>/
```

The mismatch is the main source of confusion.

## Problems To Resolve

1. Experiment entry is unclear.
   There is no dedicated FinMem experiment launcher equivalent to the FinAgent
   FINSABER-2 runner.

2. Experiment definition is split.
   Tickers and setup names live in `ExperimentRunner`, date windows are passed
   through CLI, training windows live in `strats_configs`, and output roots live
   both in CLI and JSON.

3. Output root ownership is unclear.
   `--output_dir` looks authoritative, but `artifact_config.root` currently
   overrides the FinMem artifact location.

4. Documentation is not aligned.
   Generic FINSABER output docs do not describe FinMem's run-scoped artifact
   layout.

5. Resume and skip semantics are weak.
   FinAgent has per ticker-year `metrics.json` and `job_status.json` sentinels.
   FinMem currently relies on whether benchmark/artifact files exist, but there
   is no first-class job manifest for resumable experiments.

6. Completed cherry-pick runs create reuse ambiguity.
   The previous cherry-pick five ticker run overlaps with Magnificent 7
   (`AMZN`, `MSFT`, `TSLA`). Reusing those results saves cost, but strict
   experiment identity may require a separate Magnificent 7 output tree.

## FinAgent Pattern Worth Reusing

FinAgent's experiment flow is organized around:

- A versioned manifest:
  `examples/experiments/manifests/finagent_finsaber2_2024_2026.json`
- A dedicated launcher:
  `examples/experiments/run_finagent_finsaber2.py`
- Explicit setup/window/ticker jobs.
- `--plan` preview.
- `--max-parallel` execution.
- Per-job status files.
- Per-job metrics files as completion sentinels.
- A top-level `runner_manifest.json`.
- Setup-level `run_summary.csv`.

This pattern is useful because the experiment identity is explicit and the
output tree is discoverable.

## Why FinMem Cannot Be A Direct Copy

FinMem has two output categories:

- FINSABER benchmark outputs: equity curves, trades, orders, summary metrics.
- FinMem internal outputs: memory query traces, LLM traces, reflections,
  checkpoints, and strategy/environment state.

A FinMem runner must preserve both categories. The runner should not hide
FinMem artifacts inside unrelated framework logs, and it should not split
benchmark results away from the internal traces in a way that makes audit hard.

## Recommended Direction

Create a dedicated FinMem FINSABER-2 runner:

```text
examples/experiments/run_finmem_finsaber2.py
examples/experiments/manifests/finmem_finsaber2_2024_2026.json
```

The manifest should define:

- `experiment_name`
- `data_root_default`
- `output_root_default`
- `model`
- `seed`
- evaluation assumptions
- windows
- selections
- FinMem training windows
- artifact capture flags

The runner should support:

- `--plan`
- `--setups`
- `--windows`
- `--tickers`
- `--output-root`
- `--data-root`
- `--max-parallel`, possibly defaulting to 1 for FinMem
- `--job-timeout-hours`
- `--force` or `--rerun-failed`

## Proposed Output Layout

The least confusing layout is job-oriented at the top level while keeping the
FinMem run identity inside each job:

```text
<output_root>/
  experiment_config.json
  runner_manifest.json
  logs/
  <setup>/
    FinMemStrategy/
      run_summary.csv
      <window>/
        <ticker>/
          job_status.json
          metrics.json
          metrics.pkl
          equity_curve.csv
          trades.csv
          orders.csv
          rejected_orders.csv
          llm_costs.csv
          finmem_artifacts/
            manifest.json
            train_query_trace.jsonl
            train_llm_trace.jsonl
            test_query_trace.jsonl
            test_llm_trace.jsonl
            train_reflections.jsonl
            test_reflections.jsonl
            post_train/
            test_state/
```

This would be easier to consume than the current run-level shared artifact tree.
However, it may require adapting the existing `FinMemArtifactWriter` root/run
logic or adding a runner-level copy/export step.

## Lower-Risk Alternative

Keep the current run-scoped FinMem layout and only add a manifest-driven runner
around it:

```text
<output_root>/
  experiment_config.json
  runner_manifest.json
  <setup>/
    FinMemStrategy/
      <profile>/
        finmem_artifacts/
          <config_key>/
            <run_key>/
              benchmark_results/
              tickers/
```

This minimizes code changes but is less intuitive for users who expect
`setup/strategy/window/ticker`.

## Open Design Decision

We need to choose between:

- Minimal adaptation: keep current FinMem artifact layout and document it.
- User-facing adaptation: expose a FinAgent-like job layout and move/copy
  FinMem internal artifacts under each job directory.

The second option is cleaner for future users, but it is a larger behavioral
change. The first option is safer but leaves some conceptual complexity.

## Immediate Fix Candidates

Before implementing the full runner, the smallest useful fixes are:

1. Add a FinMem-specific experiment manifest.
2. Add a FinMem-specific launcher with `--plan` and explicit data/output roots.
3. Force headless-safe plotting behavior with `MPLBACKEND=Agg` and
   `silence=True`.
4. Add tests that verify all tickers in one FinMem run share one run identity.
5. Document the chosen output layout in a permanent doc only after the design is
   finalized.

## Deeper Code Findings

### `ExperimentRunner` Is Not A Good Long-Term FinMem Entry

`ExperimentRunner` currently has two modes:

- `iter`: one date window, multiple tickers.
- `rolling_window`: generated calendar windows with selector-driven tickers.

The FinMem run identity materialization is only called when:

```python
strategy_class.__name__ == "FinMemStrategy" and self.mode == "iter"
```

Therefore, FinMem artifacts are explicitly adapted only for iterative runs.
If FinMem is run through the old rolling-window path, the run-scoped output
logic is bypassed.

There is also a structural mismatch in `FINSABER.run_rolling_window` for LLM
strategies: it calls `run_iterative_tickers(...)` per window, but then merges
returned metrics with `eval_metrics.update(metrics)`. If the same ticker appears
in multiple windows, later windows can overwrite earlier ticker keys in the
returned object. The Backtrader path (`FINSABERBt`) has a better nested output
shape, but FinMem uses the Python-native `FINSABER` path, not `FINSABERBt`.

Conclusion: a dedicated FinMem runner should call `FINSABER.run_iterative_tickers`
one ticker-year at a time, rather than relying on `ExperimentRunner` rolling
window behavior.

### FinAgent Runner Pattern

`run_finagent_finsaber2.py` is organized around explicit jobs:

```text
job = setup + window + ticker
```

For each job it:

- creates a single-ticker `FinsaberParquetDataset`;
- sets `save_results=False` in FINSABER;
- runs one ticker through `FINSABER.run_iterative_tickers`;
- writes its own `metrics.pkl`, scalar `metrics.json`, and CSV artifacts;
- writes `job_status.json`;
- uses `metrics.json` as the completion sentinel;
- writes a top-level `runner_manifest.json`;
- builds a setup-level `run_summary.csv`.

This is the cleanest pattern to reuse for FinMem.

### Non-LLM Benchmark Runner Is Not Enough

`run_finsaber2_benchmarks.py` also uses the FinAgent manifest, but it requires a
fixed ticker universe across all windows:

```python
ticker_sets = {tuple(tickers) for tickers in selections.values()}
if len(ticker_sets) != 1:
    raise ValueError(...)
```

This works for `selected_4` and `magnificent_7`, but it does not fit
`random_sp500_5`, `momentum_sp500_5`, or `lowvol_sp500_5`, because those
selections intentionally change by year.

FinMem should follow FinAgent's per-job manifest execution instead of this
fixed-universe benchmark shortcut.

### FinMem Artifact Writer Boundary

FinMem already has artifact switches:

- `enabled`
- `save_agent_checkpoint`
- `save_environment_checkpoint`
- `save_reflections`
- `save_query_trace`
- `save_llm_trace`

The writer currently resolves ticker artifacts as:

```text
<root>/<config_key>/<run_key>/tickers/<ticker>/
```

For example, if a runner sets:

```text
root = <job_dir>/finmem_artifacts
```

the actual ticker directory becomes:

```text
<job_dir>/finmem_artifacts/<config_key>/<run_key>/tickers/<ticker>/
```

This is a workable minimal integration, but it is less user-friendly than:

```text
<job_dir>/finmem_artifacts/
```

To flatten it cleanly, the writer would need a small explicit extension, such as
`artifact_config["ticker_dir"]`, `artifact_config["layout"] = "job"`, or a
similar override. A runner-level copy/export step is possible but would make
manifests point at stale internal paths unless rewritten.

### Window Date Alignment

The FinAgent manifest uses windows like:

```text
2025-01-01_2026-01-01
```

On the local parquet data, `2025-01-01` has no trading bar; the first available
COIN trading day is `2025-01-02`, and the last available day before
`2026-01-01` is `2025-12-31`.

Using the FinAgent-style window labels is still reasonable because the data
loader naturally resolves the effective trading dates, and the framework does
not treat a one-day holiday gap at the end as delisting.

### Training Window Semantics

FinAgent computes the training start by subtracting `training_years` from the
test start year. FinMem currently accepts either:

- a numeric `training_period`, converted by `365 * years` days;
- an explicit `[train_start, train_end]` list.

For official FinMem experiments, a manifest can support both:

- `training_years: 3` for alignment with FinAgent;
- optional explicit per-window `training_periods` when exact dates are required.

The runner should materialize the exact `training_period` passed into
`FinMemStrategy` and record it in `experiment_config.json`, `runner_manifest.json`,
and each ticker artifact manifest.

## Current Recommendation

Build a FinMem-specific manifest runner that is conceptually aligned with
FinAgent's runner, but does not force FinMem into FinAgent's exact internals.

Recommended first implementation:

```text
examples/experiments/run_finmem_finsaber2.py
examples/experiments/manifests/finmem_finsaber2_2024_2026.json
```

The runner should:

- use `setup + window + ticker` as the job identity;
- use single-ticker parquet loaders covering train plus test dates;
- call `FINSABER.run_iterative_tickers` with one ticker;
- set `save_results=False` and write job artifacts itself;
- use `metrics.json` as the completion sentinel;
- write `job_status.json` and `runner_manifest.json`;
- default `max_parallel` to 1 or 2;
- set `silence=True` in FINSABER trade config;
- set `MPLBACKEND=Agg` before importing matplotlib-dependent code;
- pass FinMem artifact flags through `artifact_config`;
- initially accept the nested FinMem artifact path under each job, unless we
  choose to add a small writer extension for a flat job layout.

Recommended later polish:

- add a clean writer option for job-flat artifacts;
- update permanent docs only after the layout is finalized;
- update consolidator logic so FinMem results can join FinAgent and non-LLM
  result tables by `setup/window/ticker/strategy`.

## Filing Section Audit Before Official Baselines

### Why This Matters

FinMem does not consume raw SEC filing documents directly when
`use_filing_sections=True`. Its data path is:

```text
FinMemStrategy
  -> prepare_finmem_trading_data(...)
  -> with_filing_sections(...)
  -> FilingSectionOverlayDataset
  -> item extractor
```

The default FinMem section map is:

```text
filing_k -> 10-K / item_7
filing_q -> 10-Q / part_i_item_2
```

The extracted `filing_q` text is written into FinMem mid-term memory, while the
extracted `filing_k` text is written into long-term memory. If extraction fails
under the current `failure_mode="empty"` policy, the run continues, but that
filing becomes an empty string and is effectively invisible to FinMem.

This is important for baseline fairness. If some ticker universes receive
usable 10-K/10-Q sections and others silently receive empty filing text, the
baseline is no longer comparing identical information availability.

TradingAgents is maintained on a separate branch, but it is expected to use the
same fixed ticker universes. If TradingAgents also consumes extracted filing
items, its wider item coverage needs to be audited against the same extractor
capability. Extractor fixes should therefore be planned as a shared main-branch
data utility improvement, not as a FinMem-only patch.

### Audit Scope

The audit used the FinMem manifest selections:

```text
examples/experiments/manifests/finmem_finsaber2_2024_2026.json
```

The manifest expands to 54 setup/window/ticker jobs, but only 47 unique
window/ticker pairs after deduplication across setups.

Unique ticker counts by window:

```text
2024-01-01_2025-01-01: 23 tickers
2025-01-01_2026-01-01: 24 tickers
```

The audit was run locally with no LLM/API calls. It checked:

- train and test price coverage;
- raw 10-K/10-Q filing visibility on price dates;
- extraction success for `10-K / item_7`;
- extraction success for `10-Q / part_i_item_2`.

### Raw Data Coverage Findings

Raw coverage is generally good.

```text
2024 window: 23/23 pairs have train price, test price, train/test 10-K, and train/test 10-Q.
2025 window: 24/24 pairs have train price, test price, train/test 10-K, and train/test 10-Q.
```

Boundary notes:

- `2024-01-01` and `2025-01-01` are not trading days. Effective test starts
  are `2024-01-02` and `2025-01-02`.
- `K` in the 2025 test window has price data through `2025-12-10`, not
  `2025-12-31`.
- `NFLX` in the 2025 test window has one 10-Q filing on a non-price date, so
  that row is not visible to FinMem's day-by-day environment. The same window
  still has two visible 10-Q rows.

### Section Extraction Findings

Total visible filings audited:

```text
758 visible filings
189 visible 10-K rows
569 visible 10-Q rows
```

Extractor status summary:

```text
filing_k / item_7:
  success + pass: 150
  success + warn: 5
  success + fail: 7
  failed + fail: 27

filing_q / part_i_item_2:
  success + pass: 514
  success + warn: 52
  failed + fail: 3
```

The major issue is `10-K / item_7`, not `10-Q / part_i_item_2`.

Ticker/window pairs with incomplete or empty `item_7` extraction:

```text
2024-01-01_2025-01-01 / BRK-B: train 3 raw -> 0 usable item_7
2024-01-01_2025-01-01 / CB:    train 3 raw -> 0 usable item_7; test 1 raw -> 0 usable item_7
2024-01-01_2025-01-01 / CI:    train 3 raw -> 0 usable item_7; test 1 raw -> 0 usable item_7
2024-01-01_2025-01-01 / EIX:   train 3 raw -> 0 usable item_7; test 1 raw -> 0 usable item_7
2024-01-01_2025-01-01 / INTC:  train 3 raw -> 0 usable item_7; test 1 raw -> 0 usable item_7
2024-01-01_2025-01-01 / MCK:   train 3 raw -> 1 usable item_7; test 1 raw -> 0 usable item_7
2024-01-01_2025-01-01 / PNC:   train 3 raw -> 0 usable item_7; test 1 raw -> 1 usable item_7
2025-01-01_2026-01-01 / REG:   train 3 raw -> 0 usable item_7; test 1 raw -> 0 usable item_7
```

The historical cherry-pick universe is safe for the current FinMem section map:

```text
selected_4: TSLA, NFLX, AMZN, MSFT, COIN
```

Both 2024 and 2025 selected_4 windows have usable `item_7` and
`part_i_item_2` extraction for all five tickers.

The Magnificent 7 universe is also safe for the current FinMem section map:

```text
AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
```

Both 2024 and 2025 Magnificent 7 windows have usable `item_7` and
`part_i_item_2` extraction for all seven tickers.

The affected pairs are mainly in the `random_sp500_5`, `momentum_sp500_5`, and
`lowvol_sp500_5` universes.

### Decision

Do not start official FinMem baseline experiments until the filing extraction
coverage issue is resolved or explicitly accepted as an experimental limitation.

Continue implementing the basic FinMem manifest runner and output structure, but
only use it for plan/smoke-level validation for now.

Treat extractor coverage improvement as a separate shared task:

- perform deeper audits on the failed `item_7` filings;
- inspect TradingAgents' branch-specific filing item requirements;
- extend the extractor on main so FinMem and TradingAgents consume the same
  corrected filing utility;
- merge the improved extractor back into the FinMem and TradingAgents branches
  before official baseline runs.

### Open Questions For The Extractor Repair Phase

- Are `missing_item` failures true source-document absences, alternate wording,
  table-of-contents confusion, or parser boundary failures?
- Should `internal_outline_open` be treated as unusable, warning-only, or
  recoverable with better boundary detection?
- Which additional items does TradingAgents need, and do they fail on the same
  ticker/window set?
- Should official experiments require all requested filing items to be non-empty,
  or should missing items be recorded and allowed under a documented policy?

## Phase 3 Smoke Validation Notes

### Scope

These were integration smoke tests for the FinMem manifest runner, not official
baseline experiments. They used AMZN with a short February/March 2024 window and
were removed after validation.

The smoke was designed to validate:

- manifest expansion and single-job execution guard;
- single-ticker parquet loader construction;
- explicit FinMem train/test window forwarding;
- FINSABER benchmark artifact writing;
- optional FinMem artifact sidecar writing;
- headless execution with `silence=True` and `MPLBACKEND=Agg`.

### Important Finding

The FINSABER ISO framework requires at least 21 available test trading days.
A first 6-trading-day smoke initialized FinMem and completed training, but the
framework rejected the backtest before metrics evaluation:

```text
Not enough data for backtesting. Only 6 days available.
```

The successful smoke therefore used:

```text
ticker: AMZN
train: 2024-02-01 -> 2024-02-06
test:  2024-02-07 -> 2024-03-07
```

The data precheck found 4 train trading days, 21 test trading days, and a
non-empty AMZN `10-K / item_7` section on `2024-02-02`.

### Artifact-Off Smoke

Temporary manifest:

```text
examples/experiments/manifests/_tmp_finmem_phase3_1_smoke.json
```

Temporary output:

```text
backtest/output/_tmp_finmem_phase3_1_no_artifact
```

Result:

- run completed successfully;
- benchmark outputs were written;
- no `finmem_artifacts/` directory was created;
- temporary manifest and output were removed after validation.

### Artifact-On Smoke

Temporary manifest:

```text
examples/experiments/manifests/_tmp_finmem_phase3_2_smoke_artifacts.json
```

Temporary output:

```text
backtest/output/_tmp_finmem_phase3_2_with_artifact
```

Artifact config:

```text
enabled: true
save_agent_checkpoint: false
save_environment_checkpoint: false
save_reflections: true
save_query_trace: true
save_llm_trace: true
```

Result:

- run completed successfully;
- benchmark outputs were written;
- `rejected_orders.csv` was readable with headers after the result-writer fix;
- FinMem artifacts were written under `selected_4/FinMemStrategy/finmem_artifacts`;
- no checkpoint/index/pkl files were written;
- temporary manifest and output were removed after validation.

Observed artifact row counts:

```text
train_query_trace.jsonl: 3
train_llm_trace.jsonl: 3
train_reflections.jsonl: 3
test_query_trace.jsonl: 20
test_llm_trace.jsonl: 20
test_reflections.jsonl: 20
```

### Code Review Notes

- `run_finmem_finsaber2.py` remains intentionally conservative: only `--plan`
  and explicit single-job execution are public.
- Full-manifest orchestration, parallelism, retry, and resume are intentionally
  not implemented yet.
- The runner writes standard benchmark artifacts through
  `backtest.toolkit.result_writer` instead of relying on legacy `.pkl`
  aggregation.
- `FinsaberParquetDataset` is logically narrowed to one ticker, but filing
  parquet reads still load per-year filing files before filtering. This remains
  a performance issue to revisit separately.
- Official experiments remain blocked on shared filing-section extractor repair.
