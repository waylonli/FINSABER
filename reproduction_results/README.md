# FinAgent reproduction results (news-feed fix)

Corrected FinAgent backtest results after fixing the test-time news bug (the agent
previously only ever saw the first test day's news; it now advances news day-by-day).
Training was unaffected. Only test-period ticker-years that actually had news changed.

Pickle structure: `{ "<date_from>_<date_to>": { "<TICKER>": {metrics...} } }`, where
metrics include `total_return, annual_return, annual_volatility, sharpe_ratio,
sortino_ratio, max_drawdown` (returns/vol are fractions; `max_drawdown` is already in %).

## Files
- `table3_selected4_extended_gpt4o-mini.pkl` — Table 3, Selected-4 extended (2004–2024),
  rolling 2-yr windows, **gpt-4o-mini**. Per-ticker results across all windows. News-fixed
  has-news ticker-years re-run; no-news ticker-years kept from the original results.
- `table2_selective_gpt4o-mini.pkl` — Table 2, selective short period (2022-10-06→2023-04-10),
  **gpt-4o-mini**.
- `table2_selective_gpt4o.pkl` — same Table 2 period, **gpt-4o** (reference).

Data source: FINSABER-V2 (FNSPID-derived news; identical to the original aggregated pkl for
these tickers). News is sparse pre-2022 for mega-caps by source design (not a bug).

## Headline numbers (per-ticker averages, paper format)
Table 3 (gpt-4o-mini): SPR / STR / AR% / MDD% / AV%
- TSLA  0.546 / 0.981 / 59.835 / -36.897 / 41.579
- NFLX -0.511 / 0.459 / 16.793 / -20.903 / 21.126
- AMZN  0.389 / 0.622 / 13.992 / -25.082 / 24.883
- MSFT  0.301 / 0.513 / 10.760 / -20.877 / 19.978

Table 2 (gpt-4o-mini): SPR / CR% / MDD% / AV%
- TSLA 1.389 / 29.093 / -15.711 / 42.059
- NFLX 0.487 / 7.484 / -14.251 / 42.628
- AMZN 0.883 / 11.830 / -14.217 / 26.810
- MSFT 0.575 / 7.356 / -13.062 / 26.271
