# Validation

Validation answers two separate questions: "Does the code still work?" and "Is the financial dataset suitable for a fair backtest?"

## Test Suite

Run:

```bash
python -m pytest -q tests
```

Current coverage includes:

- data adapters
- adjusted OHLC derivation
- commission, slippage, and liquidity-cap math
- `same_close` and `next_open` execution
- LLM cost ledger
- result artifact writer

## FINSABER-2 Dataset Validation

Temporary validation scripts belong under ignored `tmp/`, not package paths.

The local validation report is generated at:

```text
tmp/finsaber2_validation_report.md
tmp/finsaber2_validation_report.json
```

Current validation flags:

- zero or nonpositive price rows
- raw and adjusted OHLC inconsistencies
- extreme adjustment factors
- large adjusted-return jumps
- duplicate filing accessions from ticker alias mappings
- news and filing rows that do not exactly align to price date/symbol keys

## What To Validate Before A Large Run

| Check | Why it matters |
| --- | --- |
| Nonpositive prices | A zero or negative execution price breaks return and cost calculations. |
| Adjusted/raw consistency | Bad adjustment factors create fake crashes or rallies. |
| Duplicate filings | Repeated filings can overweight one event in text features. |
| Ticker alignment | News and filings must map to the correct tradable symbol. |
| Missing dates | Gaps can skip signals or leave pending orders without fills. |
| Volume availability | Liquidity caps need prior raw volume history. |

## Recommended Data Policy

- Filter or correct zero-price rows before benchmark runs.
- Investigate adjustment factors above `100` or below `0.01`.
- Deduplicate filing text by accession for feature construction.
- Keep symbol-to-accession mappings separately if ticker aliases matter.
- Treat date-only news and filings as next-decision information unless timestamps are available.

## Temporary Scripts Policy

Put one-off validation scripts under `tmp/` and keep that directory ignored by Git. If a validation check becomes generally useful, convert it into a test under `tests/` or a documented utility before committing it.
