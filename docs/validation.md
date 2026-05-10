# Validation

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

## Recommended Data Policy

- Filter or correct zero-price rows before benchmark runs.
- Investigate adjustment factors above `100` or below `0.01`.
- Deduplicate filing text by accession for feature construction.
- Keep symbol-to-accession mappings separately if ticker aliases matter.
- Treat date-only news and filings as next-decision information unless timestamps are available.
