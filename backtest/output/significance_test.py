"""Paired significance tests for the Selected-4 LLM-vs-baseline comparison (Table 5).

Reproduces the Selected-4 block of the significance table: for each symbol, a paired
two-sided t-test across that symbol's rolling windows on the per-window Sharpe ratio
(SPR), comparing strategy pairs (B&H vs FinMem, B&H vs FinAgent, FinMem vs FinAgent).

Run from this directory (backtest/output):
    python significance_test.py

Note: the original ad-hoc script used for the first submission was lost; this is the
recovered/documented method (paired t-test on per-window Sharpe). It is committed so
the significance numbers are reproducible going forward. Composite-setup rows are
computed separately (composite results unchanged here).
"""
from __future__ import annotations
import os, pickle
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATE_RANGE = "2004-01-01_2024-01-01"
SETUP = "selected_4"
SYMBOLS = ["TSLA", "NFLX", "AMZN", "MSFT"]
PAIRS = [("Buy&Hold", "FinMem", "BuyAndHoldStrategy", "FinMemStrategy"),
         ("Buy&Hold", "FinAgent", "BuyAndHoldStrategy", "FinAgentStrategy"),
         ("FinMem", "FinAgent", "FinMemStrategy", "FinAgentStrategy")]
METRIC = "sharpe_ratio"


def load(strategy: str) -> dict:
    path = os.path.join(HERE, SETUP, strategy, f"{DATE_RANGE}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def per_window(results: dict, symbol: str):
    """Ordered (window, value) for windows where the symbol is present."""
    return {w: results[w][symbol][METRIC] for w in results if symbol in results[w]}


def paired_p(a: dict, b: dict, symbol: str) -> float:
    aw, bw = per_window(a, symbol), per_window(b, symbol)
    common = sorted(set(aw) & set(bw))
    if len(common) < 2:
        return float("nan")
    va = np.array([aw[w] for w in common]); vb = np.array([bw[w] for w in common])
    return stats.ttest_rel(va, vb).pvalue


def main():
    data = {s: load(s) for _, _, s1, _ in PAIRS for s in (s1,)}
    # ensure all three strategies loaded
    for *_ , s1, s2 in PAIRS:
        data.setdefault(s1, load(s1)); data.setdefault(s2, load(s2))
    hdr = f"{'Symbol':6}" + "".join(f"{n1+' vs '+n2:>22}" for n1, n2, _, _ in PAIRS)
    print("Paired t-test on per-window Sharpe ratio (Selected-4)\n" + hdr)
    for sym in SYMBOLS:
        row = f"{sym:6}"
        for _, _, s1, s2 in PAIRS:
            row += f"{paired_p(data[s1], data[s2], sym):22.4f}"
        print(row)


if __name__ == "__main__":
    main()
