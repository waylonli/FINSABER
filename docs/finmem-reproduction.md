# FinMem FINSABER-2 Experiments

This workflow is the planned reproducible entry point for evaluating FinMem on
the FINSABER-2 parquet dataset over the 2024 and 2025 calendar-year windows.
The versioned manifest fixes ticker selections, test windows, explicit FinMem
training windows, data modalities, execution assumptions, and the FinMem TOML
configuration used by the strategy.

## Current Status

The FinMem manifest and launcher currently support two safe operations:

- `--plan` validates the experiment contract and expands it into ticker-year
  jobs without running backtests.
- Explicit single-job execution runs exactly one `setup/window/ticker`
  combination. If the selected filters expand to more than one job, the launcher
  stops before calling FinMem.

Full-manifest orchestration is intentionally not enabled yet. Official FinMem
baseline runs remain paused until filing-section extraction coverage is repaired
or explicitly accepted as an experimental limitation.

## Manifest

The source manifest is:

```text
examples/experiments/manifests/finmem_finsaber2_2024_2026.json
```

It uses the same frozen ticker selections as the FinAgent FINSABER-2 manifest
so FinMem and FinAgent can be compared on the same universes:

- `selected_4`
- `random_sp500_5`
- `momentum_sp500_5`
- `lowvol_sp500_5`
- `magnificent_7`

The manifest also records explicit FinMem training windows:

| Test Window | Training Window |
|---|---|
| `2024-01-01_2025-01-01` | `2021-01-04` to `2023-12-31` |
| `2025-01-01_2026-01-01` | `2022-01-03` to `2024-12-31` |

These explicit training windows avoid ambiguity from FinMem's numeric
`training_period` behavior, which converts years into `365 * years` days.

## Configuration Checks

The launcher validates FinMem-specific configuration during planning:

- The FinMem TOML config path exists.
- The manifest model matches `[chat].model` in the FinMem TOML config.
- Filing section requests resolve to exactly one supported extractor item.
- Filing payload, failure, and merge policies use supported values.
- Training windows match the declared test windows and end before testing.
- Each selected universe defines every test window.
- Empty artifact roots are rejected because `Path("")` resolves to the current
  working directory.

The current FinMem filing section requests are:

| Modality | Form | Item |
|---|---|---|
| `filing_k` | `10-K` | `item_7` |
| `filing_q` | `10-Q` | `part_i_item_2` |

Artifact capture is disabled by default in the manifest. If enabled, the runner
materializes FinMem artifact roots under the strategy output directory before
passing artifact configuration into `FinMemStrategy`.

## Preview The Plan

Activate the FinMem conda environment and preview the full experiment plan:

```bash
conda run -n finsaber2-finmem python examples/experiments/run_finmem_finsaber2.py \
  --plan \
  --data-root /mnt/cbs1/data/datasets/sp500_2000_2025_parquet
```

Preview one ticker-year job:

```bash
conda run -n finsaber2-finmem python examples/experiments/run_finmem_finsaber2.py \
  --plan \
  --setups selected_4 \
  --windows 2025-01-01_2026-01-01 \
  --tickers COIN \
  --data-root /mnt/cbs1/data/datasets/sp500_2000_2025_parquet
```

Run one explicit ticker-year job:

```bash
conda run -n finsaber2-finmem python examples/experiments/run_finmem_finsaber2.py \
  --setups selected_4 \
  --windows 2025-01-01_2026-01-01 \
  --tickers COIN \
  --data-root /mnt/cbs1/data/datasets/sp500_2000_2025_parquet
```

The command above is intentionally narrow. Running the launcher without filters
currently fails fast because the manifest expands to multiple jobs.

The full manifest currently expands to 54 jobs:

| Setup | Jobs |
|---|---:|
| `selected_4` | 10 |
| `random_sp500_5` | 10 |
| `momentum_sp500_5` | 10 |
| `lowvol_sp500_5` | 10 |
| `magnificent_7` | 14 |

Each job is identified by:

```text
setup / window / ticker
```

For example:

```text
selected_4 / 2025-01-01_2026-01-01 / COIN
```

## Date Semantics

The manifest keeps FinAgent-style calendar-year labels such as
`2025-01-01_2026-01-01` for cross-baseline comparability. Some boundary dates
are not trading days in the parquet data. FinMem's environment adjusts missing
start or end dates to nearby available dates, and its artifact manifests should
record both requested and effective windows once execution is enabled.

## Current Execution Contract

Single-job execution currently:

- Builds a one-ticker `FinsaberParquetDataset` from the explicit training start
  through the test end.
- Passes the explicit `[train_start, train_end]` tuple into `FinMemStrategy`.
- Keeps `silence=True`, disables FINSABER checkpoint reuse, and sets a
  non-interactive matplotlib backend before importing FINSABER.
- Lets FINSABER return metrics, then writes standard result artifacts with
  `backtest.toolkit.result_writer`.
- Merges existing scalar `metrics.json` leaves into `run_summary.csv` so
  repeated single-job runs do not depend on legacy `.pkl` aggregation order.

The expected benchmark output shape is:

```text
<output_root>/<setup>/FinMemStrategy/
  run_config.json
  run_manifest.json
  run_summary.csv
  <test_start>_<test_end>/<ticker>/
    metrics.json
    equity_curve.csv
    trades.csv
    rejected_orders.csv
    llm_costs.csv
    external_costs.csv
```

## Smoke Validation

The launcher has been smoke-tested with a short single-ticker AMZN job using
the `finsaber2-finmem` conda environment.

The smoke window was intentionally short, but the FINSABER ISO framework
requires at least 21 available test trading days. A shorter 6-trading-day smoke
initialized and trained FinMem successfully, but the outer framework rejected
the test window before metrics evaluation.

Validated behavior:

- `--plan` expands the intended job without running FinMem.
- Unfiltered execution fails fast because the manifest expands to multiple
  jobs.
- Single-job execution runs through FinMem training, FINSABER testing, final
  metric evaluation, and standard result writing.
- `silence=True` plus `MPLBACKEND=Agg` avoids interactive plotting.
- Artifact-off execution writes only benchmark outputs.
- Artifact-on execution writes FinMem traces/reflections under
  `FinMemStrategy/finmem_artifacts/` while leaving benchmark outputs in the
  standard tree.
- Artifact-on smoke with checkpoint saving disabled wrote trace/reflection JSONL
  files and snapshot metadata, but did not write checkpoint/index files.
- Empty `rejected_orders.csv` files are written with headers so downstream
  pandas readers can parse them.

The temporary smoke manifests and temporary output directories were removed
after validation.

## Next Implementation Step

The next code step is not full official execution. Before running complete
baselines, filing-section extraction should be repaired on the shared data
utility path and re-audited for both FinMem and TradingAgents requirements.
