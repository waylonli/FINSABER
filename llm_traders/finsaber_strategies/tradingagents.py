"""FINSABER-facing strategy wrapper for TradingAgents benchmark integration."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from stockstats import wrap

from backtest.data_util import FinsaberParquetDataset, TradingData
from backtest.data_util.filing_section_extractor import (
    FilingRow,
    audit_extraction_result,
    extract_items_from_filing,
    item_specs_for_request,
)
from backtest.strategy.timing_llm.base_strategy_iso import BaseStrategyIso

TRADINGAGENTS_STRATEGY_NAME = "TradingAgentsStrategy"
TRADINGAGENTS_BASELINE_PROFILE_ID = "finsaber_openai_gpt4omini_v1"
TRADINGAGENTS_BASELINE_SELECTED_ANALYSTS = ("market", "news", "fundamentals")
_LOCAL_TRADINGAGENTS_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "tradingagent"

package_root = str(_LOCAL_TRADINGAGENTS_PACKAGE_ROOT)
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from tradingagents import default_config
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.graph.runtime_wiring import build_runtime_tool_surfaces

_TRADINGAGENTS_GRAPH_BASELINE = {
    "llm_provider": "openai",
    "deep_think_llm": "gpt-4o-mini",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url": None,
    "temperature": None,
    "output_language": "English",
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "analyst_concurrency_limit": 1,
    "checkpoint_enabled": False,
    "benchmark_ticker": "SPY",
    "adapter_contract_version": "finsaber_ta_v1",
    "data_policy": {
        "instrument_identity_mode": "ticker_only",
        "allow_online_market_fallback": False,
        "allow_online_news_fallback": False,
        "allow_online_fundamentals_fallback": False,
        "allow_global_news": False,
        "allow_social": False,
        "allow_insider_transactions": False,
        # TradingAgents keeps raw local filings and extracts latest-visible
        # sections or semantic buckets on demand; this is not FinMem-style
        # overlay materialization.
        "filing_mode": "raw_filing_extraction",
        "unavailable_policy": "explicit_unavailable",
    },
}

_TRADINGAGENTS_EXECUTION_BRIDGE_CONTRACT = {
    "signal_fold_version": "ta_rating_to_finsaber_v1",
    "execution_timing": "next_open",
}

_TRADINGAGENTS_REFLECTION_BRIDGE_CONTRACT = {
    "entry_anchor": "next_open",
    "evaluation_anchor": "open_to_open",
    "horizon_days": 5,
    "cross_ticker_memory": 0,
    "strict_temporal_safety": True,
    "reflection_mode": "upstream_recommendation_space",
}

_TRADINGAGENTS_VALID_RATINGS = frozenset(
    {"Buy", "Overweight", "Hold", "Underweight", "Sell"}
)
_TRADINGAGENTS_VALID_POSITION_STATES = frozenset({"flat", "long"})
_TRADINGAGENTS_STANCE_TARGETS = {
    "Buy": "long",
    "Overweight": "long",
    "Hold": "keep_current_state",
    "Underweight": "flat",
    "Sell": "flat",
}

_TRADINGAGENTS_ARTIFACT_ROOT_LEAF = "tradingagents_artifacts"
_TRADINGAGENTS_DISABLED_GLOBAL_NEWS = (
    "GLOBAL_NEWS_DISABLED: Global or macro news aggregation is disabled in the "
    "FINSABER TradingAgents baseline. Do not infer missing macro developments."
)
_TRADINGAGENTS_DISABLED_INSIDER_TRANSACTIONS = (
    "INSIDER_TRANSACTIONS_DISABLED: Insider transaction retrieval is disabled "
    "in the FINSABER TradingAgents baseline."
)
_TRADINGAGENTS_DISABLED_SOCIAL_BLOCK = (
    "SOCIAL_DATA_DISABLED: StockTwits and Reddit data are disabled in the "
    "FINSABER TradingAgents baseline."
)
_TRADINGAGENTS_DUPLICATE_FUNDAMENTALS_PREFIX = (
    "DUPLICATE_LOCAL_FUNDAMENTALS_SOURCE:"
)
_TRADINGAGENTS_LOCAL_TRUNCATION_MARKER = "[TRUNCATED: local cap reached]"
_TRADINGAGENTS_NEWS_LOOKBACK_DAYS = 7
_TRADINGAGENTS_NEWS_ARTICLE_LIMIT = 20
# Each semantic bucket selects at most one latest visible source under a fixed
# precedence rule; this avoids duplicating overlapping 10-K/10-Q sections.
_TRADINGAGENTS_FUNDAMENTALS_BUCKETS = (
    ("Annual Business Context", (("10-K", "item_1"),)),
    (
        "Current Risk Context",
        (
            ("10-Q", "part_ii_item_1a"),
            ("10-K", "item_1a"),
        ),
    ),
    ("Annual MD&A Baseline", (("10-K", "item_7"),)),
    ("Latest Quarterly MD&A Update", (("10-Q", "part_i_item_2"),)),
)
_TRADINGAGENTS_FUNDAMENTALS_BUCKET_CAPS = {
    "Annual Business Context": 60_000,
    "Current Risk Context": 120_000,
    "Annual MD&A Baseline": 90_000,
    "Latest Quarterly MD&A Update": 80_000,
}
_TRADINGAGENTS_STATEMENT_PROXY_ITEMS = {
    "annual": ("10-K", "item_8", "Annual statement bundle proxy"),
    "quarterly": ("10-Q", "part_i_item_1", "Quarterly statement bundle proxy"),
}
_TRADINGAGENTS_STATEMENT_SOURCE_CAP = 140_000
_TRADINGAGENTS_FUNDAMENTALS_CYCLE_TOTAL_CAP = 420_000
_TRADINGAGENTS_SNAPSHOT_INDICATORS = (
    "close_10_ema",
    "close_50_sma",
    "close_200_sma",
    "rsi",
    "boll",
    "boll_ub",
    "boll_lb",
    "macd",
    "macds",
    "macdh",
    "atr",
)
_TRADINGAGENTS_SUPPORTED_INDICATORS = frozenset(
    _TRADINGAGENTS_SNAPSHOT_INDICATORS + ("vwma",)
)
_TRADINGAGENTS_ANALYST_INPUT_SLOTS = {
    "get_stock_data": ("market", "price_snapshot"),
    "get_indicators": ("market", "technical_indicators"),
    "get_verified_market_snapshot": ("market", "verified_market_snapshot"),
    "get_news": ("news", "ticker_news"),
    "get_global_news": ("news", "global_news"),
    "get_fundamentals": ("fundamentals", "fundamentals_proxy"),
    "get_balance_sheet": ("fundamentals", "balance_sheet_proxy"),
    "get_cashflow": ("fundamentals", "cashflow_proxy"),
    "get_income_statement": ("fundamentals", "income_statement_proxy"),
}
_TRADINGAGENTS_PROMPT_POLICY_VARIANTS = {
    "market": "market_local_na_guard_v1",
    "news": "news_local_macro_guard_v1",
    "fundamentals": "fundamentals_local_filing_guard_v1",
}
_TRADINGAGENTS_REFLECTION_PROMPT_POLICY = (
    "reflection_upstream_alpha_primary_open_to_open_v2"
)

_PATH_ENV_TO_FIELD = {
    "TRADINGAGENTS_RESULTS_DIR": "results_dir",
    "TRADINGAGENTS_CACHE_DIR": "data_cache_dir",
    "TRADINGAGENTS_MEMORY_LOG_PATH": "memory_log_path",
}


@dataclass(frozen=True)
class TAExperimentNamespace:
    strategy_name: str
    profile_name: str
    symbol: str
    date_from: str
    date_to: str
    artifact_root: str
    config_key: str
    run_key: str
    window_key: str
    base_run_dir: Path
    benchmark_results_dir: Path
    launcher_dir: Path
    ticker_dir: Path
    results_dir: Path
    data_cache_dir: Path
    memory_log_path: Path
    full_state_log_dir: Path
    ticker_namespace_meta_path: Path
    namespace_meta_path: Path
    manifest_path: Path

    def as_graph_paths(self) -> dict[str, str]:
        return {
            "results_dir": str(self.results_dir),
            "data_cache_dir": str(self.data_cache_dir),
            "memory_log_path": str(self.memory_log_path),
            "full_state_log_dir": str(self.full_state_log_dir),
        }

    def to_run_meta_payload(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "profile_name": self.profile_name,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "artifact_root": self.artifact_root,
            "config_key": self.config_key,
            "run_key": self.run_key,
            "window_key": self.window_key,
            "base_run_dir": str(self.base_run_dir),
            "benchmark_results_dir": str(self.benchmark_results_dir),
            "launcher_dir": str(self.launcher_dir),
            "results_dir": str(self.results_dir),
            "data_cache_dir": str(self.data_cache_dir),
            "namespace_meta_path": str(self.namespace_meta_path),
            "manifest_path": str(self.manifest_path),
        }

    def to_ticker_meta_payload(self) -> dict[str, Any]:
        return {
            **self.to_run_meta_payload(),
            "symbol": self.symbol,
            "ticker_dir": str(self.ticker_dir),
            "memory_log_path": str(self.memory_log_path),
            "full_state_log_dir": str(self.full_state_log_dir),
            "ticker_namespace_meta_path": str(self.ticker_namespace_meta_path),
        }


@dataclass(frozen=True)
class TARunIdentity:
    profile_name: str
    artifact_root: Path
    config_key: str
    run_key: str
    base_run_dir: Path
    benchmark_results_dir: Path


def _to_date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date value: {value!r}")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    if not override:
        return merged

    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return repr(value)


def _sanitize_run_scoped_graph_config(graph_config: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = copy.deepcopy(dict(graph_config))
    sanitized.pop("memory_log_path", None)
    sanitized.pop("full_state_log_dir", None)
    return sanitized


def _stable_json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":"))


def _build_tradingagents_prompt_policy_payload() -> dict[str, str]:
    payload = dict(_TRADINGAGENTS_PROMPT_POLICY_VARIANTS)
    payload["reflection"] = _TRADINGAGENTS_REFLECTION_PROMPT_POLICY
    return payload


def _build_tradingagents_fundamentals_payload_contract() -> dict[str, Any]:
    return {
        "truncation_marker": _TRADINGAGENTS_LOCAL_TRUNCATION_MARKER,
        "bucket_caps": dict(_TRADINGAGENTS_FUNDAMENTALS_BUCKET_CAPS),
        "statement_source_cap": _TRADINGAGENTS_STATEMENT_SOURCE_CAP,
        "cycle_total_cap": _TRADINGAGENTS_FUNDAMENTALS_CYCLE_TOTAL_CAP,
        "trim_unit": "double_newline_blocks",
        "trim_mode": "under_cap_passthrough__over_cap_block_trim",
    }


def _safe_path_component(value: Any) -> str:
    text = str(value).strip()
    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._")
    return text or "unknown"


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_parent_dir(path)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl_record(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(payload), sort_keys=True) + "\n")


def _format_tool_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _copy_parquet_loader_with_filing_merge_policy(
    data_loader: TradingData,
    *,
    filing_merge_policy: str,
) -> TradingData:
    if not isinstance(data_loader, FinsaberParquetDataset):
        return data_loader
    if data_loader.filing_merge_policy == filing_merge_policy:
        return data_loader
    return FinsaberParquetDataset(
        root=data_loader.root,
        start_date=data_loader.start_date,
        end_date=data_loader.end_date,
        tickers=data_loader.tickers,
        modalities=data_loader.modalities,
        price_field=data_loader.price_field,
        filing_merge_policy=filing_merge_policy,
    )


def _prepare_tradingagents_runtime_loaders(
    *,
    symbol: str,
    date_from: dt.date,
    date_to: dt.date,
    data_loader: TradingData,
) -> tuple[TradingData, TradingData]:
    if data_loader is None:
        raise ValueError("TradingAgentsStrategy requires a TradingData data_loader.")
    if not isinstance(data_loader, TradingData):
        raise TypeError("TradingAgentsStrategy expects data_loader to implement TradingData.")

    normalized_loader = _copy_parquet_loader_with_filing_merge_policy(
        data_loader,
        filing_merge_policy="latest",
    )
    window_loader = normalized_loader.get_subset_by_time_range(date_from, date_to)
    if window_loader is None:
        raise ValueError(
            f"No local TradingData rows are available for {symbol} between "
            f"{date_from.isoformat()} and {date_to.isoformat()}."
        )

    runtime_loader = window_loader.get_ticker_subset_by_time_range(
        symbol,
        date_from,
        date_to,
    )
    if runtime_loader is None:
        raise ValueError(
            f"No local TradingData rows are available for ticker {symbol} between "
            f"{date_from.isoformat()} and {date_to.isoformat()}."
        )
    return window_loader, runtime_loader


def _loader_modalities(data_loader: TradingData) -> list[str]:
    modalities: set[str] = set()
    try:
        for current_date in data_loader.get_date_range():
            day_payload = data_loader.get_data_by_date(current_date) or {}
            modalities.update(str(key) for key in day_payload.keys())
    except Exception:
        return []
    return sorted(modalities)


def _classify_trace_source_mode(tool_output: str) -> str:
    normalized = str(tool_output or "").strip()
    if not normalized:
        return "explicit_unavailable"
    if (
        _TRADINGAGENTS_LOCAL_TRUNCATION_MARKER in normalized
        or _has_marker_prefix_suffix_fragment(
            normalized,
            _TRADINGAGENTS_LOCAL_TRUNCATION_MARKER,
        )
    ):
        return "local_trimmed"
    if normalized.startswith(_TRADINGAGENTS_DUPLICATE_FUNDAMENTALS_PREFIX) or _starts_with_marker_prefix_fragment(
        normalized,
        _TRADINGAGENTS_DUPLICATE_FUNDAMENTALS_PREFIX,
    ):
        return "duplicate_source_note"
    if normalized.startswith(
        (
            "GLOBAL_NEWS_DISABLED:",
            "INSIDER_TRANSACTIONS_DISABLED:",
            "SOCIAL_DATA_DISABLED:",
        )
    ):
        return "disabled_placeholder"
    if normalized.startswith(
        (
            "NO_",
            "UNSUPPORTED_",
            "N/A:",
            "<unavailable",
        )
    ):
        return "explicit_unavailable"
    return "local"


def _contains_embedded_unavailable_markers(tool_output: str) -> bool:
    normalized = str(tool_output or "")
    lowered = normalized.lower()
    if not normalized.strip():
        return False
    if normalized.startswith(
        (
            "GLOBAL_NEWS_DISABLED:",
            "INSIDER_TRANSACTIONS_DISABLED:",
            "SOCIAL_DATA_DISABLED:",
            "NO_",
            "UNSUPPORTED_",
            "N/A:",
            "<unavailable",
        )
    ):
        return False
    return "n/a" in lowered or "unavailable" in lowered


def _summarize_tool_output(tool_output: str, *, max_chars: int = 280) -> str:
    collapsed = " ".join(str(tool_output or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3].rstrip()}..."


def _split_local_trim_blocks(text: str) -> list[str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return [block.strip() for block in normalized.split("\n\n") if block.strip()]


def _starts_with_marker_prefix_fragment(
    text: str,
    marker: str,
    *,
    min_chars: int = 8,
) -> bool:
    normalized = str(text or "")
    if len(normalized) >= len(marker):
        return False
    fragment_length = len(normalized)
    return fragment_length >= min_chars and marker.startswith(normalized)


def _has_marker_prefix_suffix_fragment(
    text: str,
    marker: str,
    *,
    min_chars: int = 8,
) -> bool:
    normalized = str(text or "")
    max_fragment_length = min(len(normalized), len(marker) - 1)
    for fragment_length in range(max_fragment_length, min_chars - 1, -1):
        if normalized.endswith(marker[:fragment_length]):
            return True
    return False


def _apply_local_trim_cap(
    text: str,
    *,
    cap: int,
    marker: str = _TRADINGAGENTS_LOCAL_TRUNCATION_MARKER,
) -> tuple[str, bool]:
    raw_text = str(text or "")
    if cap <= 0:
        return "", True
    if len(raw_text) <= cap:
        return raw_text, False

    blocks = _split_local_trim_blocks(raw_text)
    if not blocks:
        return marker[:cap], True

    kept: list[str] = []
    for idx, block in enumerate(blocks):
        is_last = idx == len(blocks) - 1
        candidate = "\n\n".join([*kept, block])
        reserve = 0 if is_last else len(marker)
        if len(candidate) + reserve <= cap:
            kept.append(block)
            continue

        prefix = "\n\n".join(kept)
        separator = "\n\n" if kept else ""
        remaining_for_text = cap - len(prefix) - len(separator) - len(marker)
        result = prefix
        if remaining_for_text > 0:
            snippet = block[:remaining_for_text]
            result = f"{prefix}{separator}{snippet}" if prefix else f"{snippet}"
        remaining_for_marker = cap - len(result)
        if remaining_for_marker > 0:
            result += marker[:remaining_for_marker]
        return result, True

    return "\n\n".join(kept), True


class TATraceWriter:
    """Optional benchmark-local artifact writer for ticker-scoped audit files."""

    def __init__(self, *, ticker_dir: Path):
        self.memory_reads_path = ticker_dir / "memory_reads.jsonl"
        self.memory_writes_path = ticker_dir / "memory_writes.jsonl"
        self.reflection_trace_path = ticker_dir / "reflection_trace.jsonl"
        self.analyst_input_trace_path = ticker_dir / "analyst_input_trace.jsonl"

    def append_memory_read(self, payload: Mapping[str, Any]) -> None:
        _append_jsonl_record(self.memory_reads_path, payload)

    def append_memory_write(self, payload: Mapping[str, Any]) -> None:
        _append_jsonl_record(self.memory_writes_path, payload)

    def append_reflection_trace(self, payload: Mapping[str, Any]) -> None:
        _append_jsonl_record(self.reflection_trace_path, payload)

    def append_analyst_input_trace(self, payload: Mapping[str, Any]) -> None:
        _append_jsonl_record(self.analyst_input_trace_path, payload)


def _price_dataframe_from_loader(
    data_loader: TradingData,
    *,
    ticker: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    if hasattr(data_loader, "get_price_dataframe"):
        df = data_loader.get_price_dataframe(
            tickers=[ticker],
            date_from=start_date,
            date_to=end_date,
            adjust=True,
        )
    else:
        records = []
        for current_date in data_loader.get_date_range():
            normalized_date = _to_date(current_date)
            if normalized_date < start_date or normalized_date > end_date:
                continue
            try:
                price = data_loader.get_ticker_data_by_date(ticker, normalized_date)["price"]
            except Exception:
                continue
            if not isinstance(price, Mapping):
                continue
            records.append(
                {
                    "date": normalized_date,
                    "symbol": ticker,
                    "open": price.get("adjusted_open", price.get("open")),
                    "high": price.get("adjusted_high", price.get("high")),
                    "low": price.get("adjusted_low", price.get("low")),
                    "close": price.get("adjusted_close", price.get("close")),
                    "volume": price.get("volume", 0),
                }
            )
        df = pd.DataFrame.from_records(records)

    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

    normalized = df.copy()
    if "Date" in normalized.columns and "date" not in normalized.columns:
        normalized = normalized.rename(columns={"Date": "date"})
    if "Symbol" in normalized.columns and "symbol" not in normalized.columns:
        normalized = normalized.rename(columns={"Symbol": "symbol"})

    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = normalized.dropna(subset=["date"])
    if "symbol" in normalized.columns:
        normalized = normalized[normalized["symbol"] == ticker]
    for column in ("open", "high", "low", "close", "volume"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["open", "high", "low", "close"])
    normalized = normalized.sort_values("date")
    normalized["date"] = normalized["date"].dt.date
    return normalized[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()


def _to_stockstats_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    if price_frame.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    normalized = price_frame.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    ).copy()
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    normalized = normalized.dropna(subset=["Date"])
    return normalized[["Date", "Open", "High", "Low", "Close", "Volume"]]


class TAOfflineSessionAdapter:
    """Strategy-local offline data bridge for the FINSABER TradingAgents baseline."""

    def __init__(
        self,
        *,
        symbol: str,
        date_from: dt.date,
        date_to: dt.date,
        window_loader: TradingData,
        runtime_loader: TradingData,
        memory_log: Any,
        default_benchmark_ticker: str,
        trace_writer: TATraceWriter | None = None,
    ):
        self.symbol = symbol
        self.date_from = date_from
        self.date_to = date_to
        self.window_loader = window_loader
        self.runtime_loader = runtime_loader
        self.memory_log = memory_log
        self.default_benchmark_ticker = default_benchmark_ticker
        self.trace_writer = trace_writer
        self.current_day: dt.date | None = None
        self.pre_decision_position_state: str | None = None
        self._memory_read_recorded_day: dt.date | None = None
        self._runtime_dates = [_to_date(value) for value in self.runtime_loader.get_date_range()]
        self._filing_section_cache: dict[tuple[str, str, str, str], dict[str, str]] = {}
        self._fundamentals_source_registry: set[tuple[Any, ...]] = set()
        self._fundamentals_cycle_chars_used = 0

    def _safe_window_price(
        self,
        ticker: str,
        trading_day: dt.date,
        *,
        price_field: str = "adjusted_open",
    ) -> float | None:
        try:
            value = self.window_loader.get_ticker_price_by_date(
                ticker,
                trading_day,
                price_field=price_field,
            )
        except Exception:
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if pd.isna(numeric_value) or numeric_value <= 0:
            return None
        return numeric_value

    def _benchmark_available_in_window(self, benchmark_ticker: str) -> bool:
        for trading_day in self._runtime_dates:
            if self._safe_window_price(
                benchmark_ticker,
                trading_day,
                price_field="adjusted_open",
            ) is not None:
                return True
        return False

    def describe_session(self) -> dict[str, Any]:
        runtime_tickers = []
        if hasattr(self.runtime_loader, "get_tickers_list"):
            runtime_tickers = list(self.runtime_loader.get_tickers_list())
        runtime_dates = self._runtime_dates
        return {
            "window_loader_type": type(self.window_loader).__name__,
            "runtime_loader_type": type(self.runtime_loader).__name__,
            "symbol": self.symbol,
            "date_from": self.date_from.isoformat(),
            "date_to": self.date_to.isoformat(),
            "runtime_date_count": len(runtime_dates),
            "runtime_start_date": (
                runtime_dates[0].isoformat() if runtime_dates else None
            ),
            "runtime_end_date": (
                runtime_dates[-1].isoformat() if runtime_dates else None
            ),
            "runtime_tickers": runtime_tickers,
            "runtime_modalities": _loader_modalities(self.runtime_loader),
            "default_benchmark_ticker": self.default_benchmark_ticker,
            "default_benchmark_available": self._benchmark_available_in_window(
                self.default_benchmark_ticker
            ),
        }

    def bind_decision_day(
        self,
        curr_date: str | dt.date | dt.datetime,
        pre_decision_position_state: str = "flat",
    ) -> None:
        self.current_day = _to_date(curr_date)
        self.pre_decision_position_state = normalize_tradingagents_position_state(
            pre_decision_position_state
        )
        self._memory_read_recorded_day = None
        self._fundamentals_source_registry = set()
        self._fundamentals_cycle_chars_used = 0

    def is_decision_day_bound(self, trade_date: str) -> bool:
        return self.current_day is not None and self.current_day.isoformat() == trade_date

    def dispatch_tool(self, method: str, *args, **kwargs) -> str:
        self._require_bound_day()
        method_name = str(method).strip()
        handler = getattr(self, f"_tool_{method_name}", None)
        if handler is None:
            raise ValueError(f"Unsupported local TradingAgents tool: {method_name!r}")
        output = handler(*args, **kwargs)
        self._record_analyst_input_trace(
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            tool_output=output,
        )
        return output

    def build_instrument_context(
        self,
        ticker: str,
        asset_type: str = "stock",
    ) -> str:
        from tradingagents.agents.utils.agent_utils import (
            build_instrument_context as build_agent_instrument_context,
        )

        return build_agent_instrument_context(ticker, asset_type, identity={})

    def load_past_context(self, ticker: str) -> str:
        self._require_bound_day()
        if ticker != self.symbol:
            return ""
        snapshot = self.memory_log.get_past_context_snapshot(ticker, n_same=5, n_cross=0)
        if self.trace_writer is not None and self._memory_read_recorded_day != self.current_day:
            lesson_dates = [
                entry["date"]
                for entry in snapshot["same_entries"] + snapshot["cross_entries"]
            ]
            self.trace_writer.append_memory_read(
                {
                    "date": self._require_bound_day().isoformat(),
                    "ticker": self.symbol,
                    "same_ticker_count": len(snapshot["same_entries"]),
                    "cross_ticker_count": len(snapshot["cross_entries"]),
                    "lesson_dates": lesson_dates,
                    "past_context_text": snapshot["past_context_text"],
                }
            )
            self._memory_read_recorded_day = self.current_day
        return snapshot["past_context_text"]

    def record_memory_write(
        self,
        *,
        event_type: str,
        decision_date: str,
        rating: str,
        note: str,
    ) -> None:
        if self.trace_writer is None:
            return
        self.trace_writer.append_memory_write(
            {
                "date": self._require_bound_day().isoformat(),
                "ticker": self.symbol,
                "event_type": event_type,
                "decision_date": decision_date,
                "rating": rating,
                "note": note,
            }
        )

    def record_reflection_trace(
        self,
        *,
        decision_date: str,
        benchmark_ticker: str,
        raw_return: float,
        alpha_return: float | None,
        actual_holding_days: int,
        final_trade_decision: str,
        reflection_text: str,
    ) -> None:
        if self.trace_writer is None:
            return
        reflection_contract = _TRADINGAGENTS_REFLECTION_BRIDGE_CONTRACT
        self.trace_writer.append_reflection_trace(
            {
                "ticker": self.symbol,
                "decision_date": decision_date,
                "resolution_date": self._require_bound_day().isoformat(),
                "execution_anchor": reflection_contract["entry_anchor"],
                "evaluation_anchor": reflection_contract["evaluation_anchor"],
                "horizon_days": reflection_contract["horizon_days"],
                "actual_holding_days": actual_holding_days,
                "benchmark_ticker": benchmark_ticker,
                "raw_return_open_to_open": raw_return,
                "alpha_return_open_to_open": alpha_return,
                "final_trade_decision": final_trade_decision,
                "reflection_text": reflection_text,
            }
        )

    def resolve_outcome(
        self,
        trade_date: str,
        holding_days: int,
        benchmark: str,
    ) -> tuple[float | None, float | None, int | None]:
        self._require_bound_day()
        decision_day = _to_date(trade_date)
        runtime_dates = self._runtime_dates
        if decision_day not in runtime_dates:
            return None, None, None

        decision_index = runtime_dates.index(decision_day)
        entry_index = decision_index + 1
        exit_index = entry_index + int(holding_days)
        if entry_index >= len(runtime_dates) or exit_index >= len(runtime_dates):
            return None, None, None

        maturity_day = runtime_dates[exit_index]
        if self.current_day is None or self.current_day < maturity_day:
            return None, None, None

        entry_day = runtime_dates[entry_index]
        exit_day = runtime_dates[exit_index]
        raw_entry = self.runtime_loader.get_ticker_price_by_date(
            self.symbol,
            entry_day,
            price_field="adjusted_open",
        )
        raw_exit = self.runtime_loader.get_ticker_price_by_date(
            self.symbol,
            exit_day,
            price_field="adjusted_open",
        )
        raw_return = float((raw_exit - raw_entry) / raw_entry)
        benchmark_entry = self._safe_window_price(
            benchmark,
            entry_day,
            price_field="adjusted_open",
        )
        benchmark_exit = self._safe_window_price(
            benchmark,
            exit_day,
            price_field="adjusted_open",
        )
        if benchmark_entry is None or benchmark_exit is None:
            return raw_return, None, int(holding_days)
        benchmark_return = float((benchmark_exit - benchmark_entry) / benchmark_entry)
        return raw_return, raw_return - benchmark_return, int(holding_days)

    def load_sentiment_prefetch_blocks(
        self,
        ticker: str,
        trade_date: str | dt.date | dt.datetime,
    ) -> dict[str, str]:
        self._require_bound_day()
        bound_day = min(_to_date(trade_date), self.current_day or _to_date(trade_date))
        start_date, end_date = self._resolve_local_news_window(bound_day)
        news_block = self._render_local_news_block(
            ticker=ticker,
            window_start=start_date,
            window_end=end_date,
        )
        return {
            "news_block": news_block,
            "stocktwits_block": _TRADINGAGENTS_DISABLED_SOCIAL_BLOCK,
            "reddit_block": _TRADINGAGENTS_DISABLED_SOCIAL_BLOCK,
        }

    def _require_bound_day(self) -> dt.date:
        if self.current_day is None:
            raise RuntimeError(
                "TAOfflineSessionAdapter must bind_decision_day(...) before use."
            )
        return self.current_day

    def _normalize_decision_day(self, value: Any = None) -> dt.date:
        if value in (None, ""):
            return self._require_bound_day()
        requested = _to_date(value)
        return min(requested, self._require_bound_day())

    def _record_analyst_input_trace(
        self,
        *,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        tool_output: str,
    ) -> None:
        if self.trace_writer is None or method_name not in _TRADINGAGENTS_ANALYST_INPUT_SLOTS:
            return

        analyst, input_slot = _TRADINGAGENTS_ANALYST_INPUT_SLOTS[method_name]
        visible_start, visible_end = self._resolve_trace_window(
            method_name=method_name,
            args=args,
            kwargs=kwargs,
        )
        source_mode = _classify_trace_source_mode(tool_output)
        summary_text = self._build_trace_summary_text(tool_output)
        if self._is_cycle_cap_exhausted_fundamentals_output(
            method_name=method_name,
            tool_output=tool_output,
        ):
            source_mode = "local_trimmed"
            summary_text = f"LOCAL_TRIMMED: {_TRADINGAGENTS_LOCAL_TRUNCATION_MARKER}"
        payload = {
            "date": self._require_bound_day().isoformat(),
            "ticker": self.symbol,
            "analyst": analyst,
            "prompt_policy_variant": _TRADINGAGENTS_PROMPT_POLICY_VARIANTS[analyst],
            "source_mode": source_mode,
            "input_slot": input_slot,
            "visible_window_start": visible_start,
            "visible_window_end": visible_end,
            "summary_text": summary_text,
        }
        self.trace_writer.append_analyst_input_trace(payload)

    def _build_trace_summary_text(self, tool_output: str) -> str:
        source_mode = _classify_trace_source_mode(tool_output)
        summary = _summarize_tool_output(tool_output)
        if source_mode == "local_trimmed":
            return f"LOCAL_TRIMMED: {summary}"
        if source_mode == "local" and _contains_embedded_unavailable_markers(tool_output):
            return f"LOCAL_WITH_UNAVAILABLE_MARKERS: {summary}"
        return summary

    def _resolve_trace_window(
        self,
        *,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[str, str]:
        decision_day = self._require_bound_day()

        if method_name == "get_stock_data":
            start_value = args[1] if len(args) > 1 else kwargs.get("start_date", decision_day)
            end_value = args[2] if len(args) > 2 else kwargs.get("end_date", decision_day)
            start_date = max(self.date_from, _to_date(start_value))
            end_date = min(decision_day, _to_date(end_value))
            return start_date.isoformat(), end_date.isoformat()

        if method_name == "get_news":
            start_date, end_date = self._resolve_local_news_window(decision_day)
            return start_date.isoformat(), end_date.isoformat()

        if method_name == "get_indicators":
            curr_date = args[2] if len(args) > 2 else kwargs.get("curr_date")
            look_back_days = args[3] if len(args) > 3 else kwargs.get("look_back_days", 30)
            end_date = self._normalize_decision_day(curr_date)
            stock_df = _to_stockstats_frame(
                self._price_frame(
                    ticker=self.symbol,
                    end_date=end_date,
                )
            )
            if stock_df.empty:
                return self.date_from.isoformat(), end_date.isoformat()
            look_back = max(1, int(look_back_days or 1))
            window = stock_df.tail(look_back)
            start_date = pd.to_datetime(window["Date"]).dt.date.iloc[0]
            finish_date = pd.to_datetime(window["Date"]).dt.date.iloc[-1]
            return start_date.isoformat(), finish_date.isoformat()

        if method_name == "get_verified_market_snapshot":
            curr_date = args[1] if len(args) > 1 else kwargs.get("curr_date")
            look_back_days = args[2] if len(args) > 2 else kwargs.get("look_back_days", 30)
            end_date = self._normalize_decision_day(curr_date)
            stock_df = _to_stockstats_frame(
                self._price_frame(
                    ticker=self.symbol,
                    end_date=end_date,
                )
            )
            if stock_df.empty:
                return self.date_from.isoformat(), end_date.isoformat()
            recent = stock_df.tail(max(1, min(int(look_back_days), 30)))
            start_date = pd.to_datetime(recent["Date"]).dt.date.iloc[0]
            finish_date = pd.to_datetime(recent["Date"]).dt.date.iloc[-1]
            return start_date.isoformat(), finish_date.isoformat()

        if method_name == "get_global_news":
            curr_date = args[0] if args else kwargs.get("curr_date", decision_day)
            look_back_days = (
                args[1] if len(args) > 1 else kwargs.get("look_back_days")
            )
            end_date = self._normalize_decision_day(curr_date)
            if look_back_days in (None, ""):
                start_date = self.date_from
            else:
                start_date = max(
                    self.date_from,
                    end_date - dt.timedelta(days=max(0, int(look_back_days))),
                )
            return start_date.isoformat(), end_date.isoformat()

        if method_name in {
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
        }:
            curr_date = None
            if method_name == "get_fundamentals":
                curr_date = args[1] if len(args) > 1 else kwargs.get("curr_date")
            else:
                curr_date = args[2] if len(args) > 2 else kwargs.get("curr_date")
            end_date = self._normalize_decision_day(curr_date)
            return self.date_from.isoformat(), end_date.isoformat()

        return self.date_from.isoformat(), decision_day.isoformat()

    def _clamp_date_range(
        self,
        start_date: str | dt.date | dt.datetime,
        end_date: str | dt.date | dt.datetime,
    ) -> tuple[dt.date, dt.date] | None:
        clamped_start = max(self.date_from, _to_date(start_date))
        clamped_end = min(self._require_bound_day(), _to_date(end_date))
        if clamped_end < clamped_start:
            return None
        return clamped_start, clamped_end

    def _resolve_local_news_window(
        self,
        curr_date: str | dt.date | dt.datetime | None = None,
    ) -> tuple[dt.date, dt.date]:
        end_date = self._normalize_decision_day(curr_date)
        start_date = max(
            self.date_from,
            end_date - dt.timedelta(days=_TRADINGAGENTS_NEWS_LOOKBACK_DAYS),
        )
        return start_date, end_date

    def _collect_visible_local_news_items(
        self,
        *,
        ticker: str,
        start_date: dt.date,
        end_date: dt.date,
    ) -> list[tuple[dt.date, str]]:
        items: list[tuple[dt.date, str]] = []
        for current_date in self.runtime_loader.get_date_range():
            normalized_date = _to_date(current_date)
            if normalized_date < start_date or normalized_date > end_date:
                continue
            news_items = self.runtime_loader.get_ticker_data_by_date(
                ticker,
                normalized_date,
            ).get("news", [])
            if not news_items:
                continue
            for news_item in news_items:
                items.append((normalized_date, str(news_item).strip()))
        items.sort(key=lambda item: item[0], reverse=True)
        return items[:_TRADINGAGENTS_NEWS_ARTICLE_LIMIT]

    def _render_local_news_block(
        self,
        *,
        ticker: str,
        window_start: dt.date,
        window_end: dt.date,
    ) -> str:
        visible_items = self._collect_visible_local_news_items(
            ticker=ticker,
            start_date=window_start,
            end_date=window_end,
        )
        if not visible_items:
            return (
                f"NO_TICKER_NEWS_AVAILABLE: No local ticker news is visible for "
                f"{ticker} between {window_start.isoformat()} and {window_end.isoformat()}."
            )

        lines = [
            f"## Local ticker news for {ticker.upper()}",
            f"- Visible window: {window_start.isoformat()} to {window_end.isoformat()}",
            "",
        ]
        for news_date, news_item in visible_items:
            lines.append(f"- {news_date.isoformat()}: {news_item}")
        return "\n".join(lines)

    def _price_frame(
        self,
        *,
        ticker: str,
        end_date: dt.date,
        start_date: dt.date | None = None,
        loader: TradingData | None = None,
    ) -> pd.DataFrame:
        return _price_dataframe_from_loader(
            loader or self.runtime_loader,
            ticker=ticker,
            start_date=start_date or self.date_from,
            end_date=end_date,
        )

    def _tool_get_stock_data(self, symbol: str, start_date: str, end_date: str) -> str:
        if symbol != self.symbol:
            return (
                f"NO_MARKET_DATA_AVAILABLE: Local price data is only available for "
                f"the active ticker {self.symbol!r}; requested {symbol!r}."
            )
        clamped = self._clamp_date_range(start_date, end_date)
        if clamped is None:
            return (
                f"NO_MARKET_DATA_AVAILABLE: No visible price rows are available for "
                f"{symbol} between {start_date} and {end_date} under the current "
                "test-window boundary."
            )
        frame = self._price_frame(
            ticker=symbol,
            start_date=clamped[0],
            end_date=clamped[1],
        )
        if frame.empty:
            return (
                f"NO_MARKET_DATA_AVAILABLE: No local price rows are available for "
                f"{symbol} between {clamped[0].isoformat()} and {clamped[1].isoformat()}."
            )
        output = frame[["date", "open", "high", "low", "close", "volume"]].copy()
        output["date"] = output["date"].map(lambda value: value.isoformat())
        output = output.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        return output.to_csv(index=False)

    def _tool_get_indicators(
        self,
        symbol: str,
        indicator: str,
        curr_date: str,
        look_back_days: int = 30,
    ) -> str:
        if symbol != self.symbol:
            return (
                f"NO_MARKET_DATA_AVAILABLE: Local indicators are only available for "
                f"the active ticker {self.symbol!r}; requested {symbol!r}."
            )
        indicator_name = str(indicator).strip().lower()
        if indicator_name not in _TRADINGAGENTS_SUPPORTED_INDICATORS:
            return (
                f"UNSUPPORTED_INDICATOR: {indicator_name!r} is not part of the "
                "TradingAgents FINSABER baseline indicator set."
            )

        decision_day = self._normalize_decision_day(curr_date)
        stock_df = _to_stockstats_frame(
            self._price_frame(
                ticker=symbol,
                end_date=decision_day,
            )
        )
        if stock_df.empty:
            return (
                f"NO_MARKET_DATA_AVAILABLE: No local price history is available for "
                f"{symbol} on or before {decision_day.isoformat()}."
            )

        stats_df = wrap(stock_df.copy())
        try:
            stats_df[indicator_name]
        except Exception as exc:
            return f"N/A: Could not compute {indicator_name} from visible history ({type(exc).__name__})."

        look_back = max(1, int(look_back_days or 1))
        window = stats_df.tail(look_back).copy()
        window["Date"] = pd.to_datetime(window["Date"]).dt.strftime("%Y-%m-%d")
        output = window[["Date", indicator_name]].rename(
            columns={indicator_name: indicator_name.upper()}
        )
        return output.to_csv(index=False)

    def _tool_get_verified_market_snapshot(
        self,
        symbol: str,
        curr_date: str,
        look_back_days: int = 30,
    ) -> str:
        if symbol != self.symbol:
            return (
                f"NO_MARKET_DATA_AVAILABLE: Verified snapshots are only available "
                f"for the active ticker {self.symbol!r}; requested {symbol!r}."
            )

        decision_day = self._normalize_decision_day(curr_date)
        stock_df = _to_stockstats_frame(
            self._price_frame(
                ticker=symbol,
                end_date=decision_day,
            )
        )
        if stock_df.empty:
            return (
                f"NO_MARKET_DATA_AVAILABLE: No verified local OHLCV rows are "
                f"available for {symbol} on or before {decision_day.isoformat()}."
            )

        stats_df = wrap(stock_df.copy())
        indicator_values: dict[str, str] = {}
        for indicator_name in _TRADINGAGENTS_SNAPSHOT_INDICATORS:
            try:
                stats_df[indicator_name]
                indicator_values[indicator_name] = _format_tool_value(
                    stats_df.iloc[-1][indicator_name]
                )
            except Exception as exc:
                indicator_values[indicator_name] = f"N/A ({type(exc).__name__})"

        latest = stock_df.iloc[-1]
        recent = stock_df.tail(max(1, min(int(look_back_days), 30)))
        lines = [
            f"## Verified local market data snapshot for {symbol.upper()}",
            "",
            f"- Requested analysis date: {decision_day.isoformat()}",
            f"- Latest trading row used: {_format_tool_value(latest['Date'])}",
            "- Rows after the requested analysis date are excluded before verification.",
            "",
            "### Latest verified OHLCV row",
            "",
            "| Field | Value |",
            "|---|---:|",
        ]
        for field in ("Open", "High", "Low", "Close", "Volume"):
            lines.append(f"| {field} | {_format_tool_value(latest[field])} |")

        lines += [
            "",
            "### Verified technical indicators (latest row)",
            "",
            "| Indicator | Value |",
            "|---|---:|",
        ]
        for indicator_name, indicator_value in indicator_values.items():
            lines.append(f"| {indicator_name} | {indicator_value} |")

        lines += [
            "",
            f"### Recent verified closes (last {len(recent)} rows)",
            "",
            "| Date | Close |",
            "|---|---:|",
        ]
        for _, row in recent.iterrows():
            lines.append(
                f"| {_format_tool_value(row['Date'])} | {_format_tool_value(row['Close'])} |"
            )

        lines += [
            "",
            "Use this snapshot as the source of truth for exact OHLCV, price-level, "
            "and indicator-value claims. If another tool output conflicts with it, "
            "flag the discrepancy rather than inventing a reconciled number.",
        ]
        return "\n".join(lines)

    def _tool_get_news(self, ticker: str, start_date: str, end_date: str) -> str:
        del start_date, end_date
        if ticker != self.symbol:
            return (
                f"NO_TICKER_NEWS_AVAILABLE: Local news is only available for the "
                f"active ticker {self.symbol!r}; requested {ticker!r}."
            )
        window_start, window_end = self._resolve_local_news_window()
        return self._render_local_news_block(
            ticker=ticker,
            window_start=window_start,
            window_end=window_end,
        )

    def _tool_get_global_news(
        self,
        curr_date: str,
        look_back_days: int | None = None,
        limit: int | None = None,
    ) -> str:
        del curr_date, look_back_days, limit
        return _TRADINGAGENTS_DISABLED_GLOBAL_NEWS

    def _tool_get_insider_transactions(self, ticker: str) -> str:
        del ticker
        return _TRADINGAGENTS_DISABLED_INSIDER_TRANSACTIONS

    def _tool_get_fundamentals(self, ticker: str, curr_date: str) -> str:
        if ticker != self.symbol:
            return (
                f"NO_FUNDAMENTALS_AVAILABLE: Local fundamentals are only available "
                f"for the active ticker {self.symbol!r}; requested {ticker!r}."
            )
        as_of_date = self._normalize_decision_day(curr_date)
        blocks = [
            f"## Local filing-based fundamentals proxy for {ticker.upper()}",
            f"- As-of date: {as_of_date.isoformat()}",
            "- Source policy: deterministic semantic-bucket precedence over latest visible filings within the active test window.",
            "- This is a filing-based proxy, not a live company-overview vendor feed.",
            "",
        ]

        rendered_bucket_count = 0
        source_refs: list[tuple[str, str, str, str]] = []
        for bucket_label, candidates in _TRADINGAGENTS_FUNDAMENTALS_BUCKETS:
            rendered_bucket, source_ref = self._render_selected_fundamentals_bucket(
                as_of_date=as_of_date,
                bucket_label=bucket_label,
                candidates=candidates,
            )
            if not rendered_bucket:
                continue
            rendered_bucket_count += 1
            if source_ref is not None:
                source_refs.append(source_ref)
            blocks.extend(rendered_bucket)

        if rendered_bucket_count == 0:
            return (
                f"NO_FUNDAMENTALS_AVAILABLE: No visible local 10-K or 10-Q filings "
                f"exist for {ticker} on or before {as_of_date.isoformat()}."
            )
        duplicate_note = self._claim_fundamentals_source(
            fingerprint=tuple(source_refs),
            duplicate_note=self._format_fundamentals_bundle_duplicate_note(
                ticker=ticker,
                as_of_date=as_of_date,
                source_refs=tuple(source_refs),
            ),
        )
        if duplicate_note is not None:
            return self._apply_fundamentals_cycle_cap(duplicate_note)
        return self._apply_fundamentals_cycle_cap("\n".join(blocks))

    def _tool_get_balance_sheet(
        self,
        ticker: str,
        freq: str = "quarterly",
        curr_date: str | None = None,
    ) -> str:
        return self._render_statement_proxy("Balance Sheet", ticker, freq, curr_date)

    def _tool_get_cashflow(
        self,
        ticker: str,
        freq: str = "quarterly",
        curr_date: str | None = None,
    ) -> str:
        return self._render_statement_proxy("Cash Flow", ticker, freq, curr_date)

    def _tool_get_income_statement(
        self,
        ticker: str,
        freq: str = "quarterly",
        curr_date: str | None = None,
    ) -> str:
        return self._render_statement_proxy("Income Statement", ticker, freq, curr_date)

    def _render_statement_proxy(
        self,
        report_name: str,
        ticker: str,
        freq: str,
        curr_date: str | None,
    ) -> str:
        if ticker != self.symbol:
            return (
                f"NO_FUNDAMENTALS_AVAILABLE: Local statement proxies are only "
                f"available for the active ticker {self.symbol!r}; requested {ticker!r}."
            )

        normalized_freq = str(freq or "quarterly").strip().lower()
        if normalized_freq not in _TRADINGAGENTS_STATEMENT_PROXY_ITEMS:
            return (
                f"UNSUPPORTED_FREQUENCY: {normalized_freq!r} is not supported. "
                "Expected 'annual' or 'quarterly'."
            )

        as_of_date = self._normalize_decision_day(curr_date)
        form, item_key, label = _TRADINGAGENTS_STATEMENT_PROXY_ITEMS[normalized_freq]
        modality = "filing_k" if form == "10-K" else "filing_q"
        latest_filing = self._latest_visible_filing(modality, as_of_date)
        if latest_filing is None:
            return (
                f"NO_FUNDAMENTALS_AVAILABLE: No visible local {form} filing exists "
                f"for {ticker} on or before {as_of_date.isoformat()}."
            )

        extracted = self._extract_sections(
            raw_text=latest_filing["text"],
            form=form,
            current_date=latest_filing["date"],
            item_keys=(item_key,),
        )
        section_text = extracted.get(item_key, "")
        if not section_text:
            return (
                f"NO_FUNDAMENTALS_AVAILABLE: Could not extract {form} {item_key} "
                f"for {ticker} from the latest visible local filing dated "
                f"{latest_filing['date'].isoformat()}."
            )
        duplicate_note = self._claim_fundamentals_source(
            fingerprint=(form, latest_filing["date"].isoformat(), item_key),
            duplicate_note=self._format_statement_proxy_duplicate_note(
                ticker=ticker,
                report_name=report_name,
                form=form,
                filing_date=latest_filing["date"],
                item_key=item_key,
            ),
        )
        if duplicate_note is not None:
            return self._apply_fundamentals_cycle_cap(duplicate_note)
        section_text = self._apply_fundamentals_source_cap(
            section_text,
            cap=_TRADINGAGENTS_STATEMENT_SOURCE_CAP,
        )

        return self._apply_fundamentals_cycle_cap(
            "\n".join(
                [
                    f"## Local {report_name} proxy for {ticker.upper()}",
                    f"- As-of date: {as_of_date.isoformat()}",
                    f"- Source: latest visible {form} filing dated {latest_filing['date'].isoformat()}",
                    f"- Proxy section: {item_key}",
                    f"- Semantics: {label}; this is not a structured statement table.",
                    "",
                    section_text,
                ]
            )
        )

    def _latest_visible_filing(
        self,
        modality: str,
        as_of_date: dt.date,
    ) -> dict[str, Any] | None:
        latest_payload = None
        for current_date in self.runtime_loader.get_date_range():
            normalized_date = _to_date(current_date)
            if normalized_date > as_of_date:
                break
            ticker_payload = self.runtime_loader.get_ticker_data_by_date(
                self.symbol,
                normalized_date,
            )
            raw_text = ticker_payload.get(modality)
            if isinstance(raw_text, str) and raw_text.strip():
                latest_payload = {
                    "date": normalized_date,
                    "text": raw_text,
                }
        return latest_payload

    def _render_selected_fundamentals_bucket(
        self,
        *,
        as_of_date: dt.date,
        bucket_label: str,
        candidates: tuple[tuple[str, str], ...],
    ) -> tuple[list[str], tuple[str, str, str, str] | None]:
        for form, item_key in candidates:
            modality = "filing_k" if form == "10-K" else "filing_q"
            latest_filing = self._latest_visible_filing(modality, as_of_date)
            if latest_filing is None:
                continue

            extracted = self._extract_sections(
                raw_text=latest_filing["text"],
                form=form,
                current_date=latest_filing["date"],
                item_keys=(item_key,),
            )
            section_text = extracted.get(item_key, "")
            if not section_text:
                continue
            capped_text = self._apply_fundamentals_source_cap(
                section_text,
                cap=_TRADINGAGENTS_FUNDAMENTALS_BUCKET_CAPS[bucket_label],
            )

            return (
                [
                    f"### {bucket_label}",
                    f"- Source: latest visible {form} {item_key} dated {latest_filing['date'].isoformat()}",
                    "",
                    capped_text,
                    "",
                ],
                (bucket_label, form, latest_filing["date"].isoformat(), item_key),
            )

        return [], None

    def _claim_fundamentals_source(
        self,
        *,
        fingerprint: tuple[Any, ...],
        duplicate_note: str,
    ) -> str | None:
        if fingerprint in self._fundamentals_source_registry:
            return duplicate_note
        self._fundamentals_source_registry.add(fingerprint)
        return None

    def _apply_fundamentals_source_cap(
        self,
        text: str,
        *,
        cap: int,
    ) -> str:
        trimmed_text, _ = _apply_local_trim_cap(text, cap=cap)
        return trimmed_text

    def _apply_fundamentals_cycle_cap(
        self,
        tool_output: str,
    ) -> str:
        remaining = _TRADINGAGENTS_FUNDAMENTALS_CYCLE_TOTAL_CAP - self._fundamentals_cycle_chars_used
        final_output, _ = _apply_local_trim_cap(tool_output, cap=remaining)
        self._fundamentals_cycle_chars_used += len(final_output)
        return final_output

    def _is_cycle_cap_exhausted_fundamentals_output(
        self,
        *,
        method_name: str,
        tool_output: str,
    ) -> bool:
        if str(tool_output or "").strip():
            return False
        if method_name not in {
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
        }:
            return False
        return (
            self._fundamentals_cycle_chars_used
            >= _TRADINGAGENTS_FUNDAMENTALS_CYCLE_TOTAL_CAP
        )

    def _format_statement_proxy_duplicate_note(
        self,
        *,
        ticker: str,
        report_name: str,
        form: str,
        filing_date: dt.date,
        item_key: str,
    ) -> str:
        return (
            f"{_TRADINGAGENTS_DUPLICATE_FUNDAMENTALS_PREFIX} "
            f"{report_name} for {ticker.upper()} reuses the same local {form} "
            f"{item_key} filing dated {filing_date.isoformat()} that was already "
            "expanded earlier in this decision day. Continue reasoning from the "
            "earlier body instead of requesting it again."
        )

    def _format_fundamentals_bundle_duplicate_note(
        self,
        *,
        ticker: str,
        as_of_date: dt.date,
        source_refs: tuple[tuple[str, str, str, str], ...],
    ) -> str:
        source_summary = "; ".join(
            f"{bucket_label}={form} {item_key} dated {filing_date}"
            for bucket_label, form, filing_date, item_key in source_refs
        )
        return (
            f"{_TRADINGAGENTS_DUPLICATE_FUNDAMENTALS_PREFIX} "
            f"Fundamentals proxy for {ticker.upper()} as of {as_of_date.isoformat()} "
            "reuses the same local filing-source bundle that was already expanded "
            f"earlier in this decision day. Sources: {source_summary}. Continue "
            "reasoning from the earlier body instead of requesting the full bundle "
            "again."
        )

    def _extract_sections(
        self,
        *,
        raw_text: str,
        form: str,
        current_date: dt.date,
        item_keys: tuple[str, ...],
    ) -> dict[str, str]:
        cache_key = (
            form,
            current_date.isoformat(),
            ",".join(item_keys),
            hashlib.sha256(raw_text.encode("utf-8", "ignore")).hexdigest(),
        )
        if cache_key in self._filing_section_cache:
            return self._filing_section_cache[cache_key]

        item_specs = item_specs_for_request([form], list(item_keys))
        filing = FilingRow(
            date=current_date.isoformat(),
            symbol=self.symbol,
            cik="",
            accession="",
            form=form,
            year=current_date.year,
            filing_idx=0,
            filing_text=raw_text,
            source_file="",
            source_row_idx=0,
        )
        extraction = extract_items_from_filing(filing=filing, item_specs=item_specs)
        audit = audit_extraction_result(
            filing=filing,
            extraction_result=extraction,
            item_specs=item_specs,
        )
        allowed_item_keys = {
            item["item_key"]
            for item in audit["item_results"]
            if item.get("status") != "fail"
        }
        extracted = {
            section["item_key"]: str(section.get("section_text") or "").strip()
            for section in extraction["sections"]
            if section["item_key"] in allowed_item_keys
        }
        self._filing_section_cache[cache_key] = extracted
        return extracted


def _get_tradingagents_graph_class():
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    return TradingAgentsGraph


def _resolve_tradingagents_artifact_root(artifact_root: str) -> tuple[Path, str]:
    artifact_root_path = Path(artifact_root).expanduser().resolve()
    if artifact_root_path.name != _TRADINGAGENTS_ARTIFACT_ROOT_LEAF:
        raise ValueError(
            "artifact_config.root must point to a 'tradingagents_artifacts' directory."
        )

    profile_dir = artifact_root_path.parent
    raw_profile_name = profile_dir.name.strip()
    if not raw_profile_name:
        raise ValueError(
            "artifact_config.root must include a non-empty profile directory "
            "before 'tradingagents_artifacts'."
        )

    return artifact_root_path, _safe_path_component(raw_profile_name)


def normalize_tradingagents_rating(raw_rating: str) -> str:
    if not isinstance(raw_rating, str):
        raise TypeError("raw_rating must be a string.")

    normalized = raw_rating.strip().title()
    if normalized not in _TRADINGAGENTS_VALID_RATINGS:
        raise ValueError(f"Unsupported TradingAgents rating: {raw_rating!r}")
    return normalized


def normalize_tradingagents_position_state(position_state: str) -> str:
    if not isinstance(position_state, str):
        raise TypeError("position_state must be a string.")

    normalized = position_state.strip().lower()
    if normalized not in _TRADINGAGENTS_VALID_POSITION_STATES:
        raise ValueError(f"Unsupported position_state: {position_state!r}")
    return normalized


def infer_tradingagents_position_state(framework: Any, symbol: str) -> str:
    portfolio = getattr(framework, "portfolio", {}) or {}
    position = portfolio.get(symbol)
    if not position:
        return "flat"

    quantity = int(position.get("quantity", 0) or 0)
    if quantity < 0:
        raise ValueError("Short positions are not supported in TradingAgentsStrategy.")
    return "long" if quantity > 0 else "flat"


def extract_tradingagents_reference_price(
    today_data: Mapping[str, Any],
    *,
    symbol: str,
) -> float:
    if not isinstance(today_data, Mapping):
        raise TypeError("today_data must be a mapping.")

    price_block = today_data.get("price")
    if not isinstance(price_block, Mapping):
        raise ValueError("today_data must include a 'price' mapping.")

    symbol_price = price_block.get(symbol)
    if not isinstance(symbol_price, Mapping):
        raise ValueError(f"today_data['price'] must include a mapping for {symbol!r}.")

    for field_name in ("adjusted_close", "close", "adjusted_open", "open"):
        value = symbol_price.get(field_name)
        if value is None:
            continue
        price = float(value)
        if price > 0:
            return price

    raise ValueError(
        f"Unable to resolve a positive reference price for {symbol!r} from today_data."
    )


def build_tradingagents_execution_bridge_payload(
    *,
    raw_rating: str,
    pre_decision_position_state: str,
) -> dict[str, str]:
    """Build the decision-day execution bridge payload.

    The payload stays strategy-local and pre-fill. It only captures the folded
    target state and the order intent submitted on the decision day; actual
    execution still depends on the framework on the next trading day's open.
    """

    normalized_rating = normalize_tradingagents_rating(raw_rating)
    normalized_pre_state = normalize_tradingagents_position_state(
        pre_decision_position_state
    )
    stance_target_state = _TRADINGAGENTS_STANCE_TARGETS[normalized_rating]

    if stance_target_state == "keep_current_state":
        mapped_target_state = normalized_pre_state
    else:
        mapped_target_state = stance_target_state

    if mapped_target_state == "long" and normalized_pre_state == "flat":
        executed_action = "submit_buy_all"
    elif mapped_target_state == "flat" and normalized_pre_state == "long":
        executed_action = "submit_sell_all"
    elif mapped_target_state == "long":
        executed_action = "noop"
    else:
        executed_action = "noop"

    return {
        "raw_rating": normalized_rating,
        "pre_decision_position_state": normalized_pre_state,
        "mapped_target_state": mapped_target_state,
        "executed_action": executed_action,
    }


def _describe_data_loader_input(data_loader: Any) -> Any:
    if data_loader is None:
        return None
    if isinstance(data_loader, (str, int, float, bool)):
        return data_loader
    if isinstance(data_loader, Path):
        return str(data_loader)
    return {
        "kind": "runtime_object",
        "type": type(data_loader).__name__,
    }


def normalize_tradingagents_artifact_config(
    artifact_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if artifact_config is None:
        raise ValueError("artifact_config is required for TradingAgentsStrategy.")
    if not isinstance(artifact_config, Mapping):
        raise TypeError("artifact_config must be a mapping.")

    root = artifact_config.get("root")
    if root is None or not str(root).strip():
        raise ValueError("artifact_config.root must be a non-empty path string.")
    _resolve_tradingagents_artifact_root(str(root).strip())

    run_key = artifact_config.get("run_key")
    normalized_run_key = None if run_key in (None, "") else str(run_key).strip()

    return {
        "enabled": bool(artifact_config.get("enabled", True)),
        "root": str(root).strip(),
        "run_key": normalized_run_key,
    }


def build_tradingagents_graph_config(
    *,
    namespace_paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    graph_config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    graph_config = _deep_merge(graph_config, _TRADINGAGENTS_GRAPH_BASELINE)
    graph_config["selected_analysts"] = list(TRADINGAGENTS_BASELINE_SELECTED_ANALYSTS)
    graph_config["baseline_profile_id"] = TRADINGAGENTS_BASELINE_PROFILE_ID
    graph_config["benchmark_execution_contract"] = copy.deepcopy(
        _TRADINGAGENTS_EXECUTION_BRIDGE_CONTRACT
    )
    graph_config["benchmark_reflection_contract"] = copy.deepcopy(
        _TRADINGAGENTS_REFLECTION_BRIDGE_CONTRACT
    )
    graph_config["fundamentals_payload_contract"] = _build_tradingagents_fundamentals_payload_contract()
    graph_config = _deep_merge(graph_config, namespace_paths)
    return graph_config


def build_tradingagents_config_key(
    graph_config: Mapping[str, Any],
    *,
    strategy_name: str = "TradingAgentsStrategy",
    baseline_profile_id: str = TRADINGAGENTS_BASELINE_PROFILE_ID,
    selected_analysts: tuple[str, ...] = TRADINGAGENTS_BASELINE_SELECTED_ANALYSTS,
) -> str:
    payload = {
        "strategy_name": strategy_name,
        "baseline_profile_id": baseline_profile_id,
        "selected_analysts": list(selected_analysts),
        "llm_provider": graph_config.get("llm_provider"),
        "deep_think_llm": graph_config.get("deep_think_llm"),
        "quick_think_llm": graph_config.get("quick_think_llm"),
        "backend_url": graph_config.get("backend_url"),
        "temperature": graph_config.get("temperature"),
        "output_language": graph_config.get("output_language"),
        "max_debate_rounds": graph_config.get("max_debate_rounds"),
        "max_risk_discuss_rounds": graph_config.get("max_risk_discuss_rounds"),
        "analyst_concurrency_limit": graph_config.get("analyst_concurrency_limit"),
        "checkpoint_enabled": graph_config.get("checkpoint_enabled"),
        "benchmark_ticker": graph_config.get("benchmark_ticker"),
        "adapter_contract_version": graph_config.get("adapter_contract_version"),
        "data_policy": graph_config.get("data_policy", {}),
        "prompt_policy": _build_tradingagents_prompt_policy_payload(),
        "execution_bridge_contract": _TRADINGAGENTS_EXECUTION_BRIDGE_CONTRACT,
        "reflection_bridge_contract": _TRADINGAGENTS_REFLECTION_BRIDGE_CONTRACT,
        "fundamentals_payload_contract": _build_tradingagents_fundamentals_payload_contract(),
    }
    fingerprint = hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()[:10]
    return f"ta_{fingerprint}"


def materialize_tradingagents_run_identity(
    *,
    artifact_config: Mapping[str, Any],
    graph_config: Mapping[str, Any] | None = None,
    strategy_name: str = TRADINGAGENTS_STRATEGY_NAME,
    baseline_profile_id: str = TRADINGAGENTS_BASELINE_PROFILE_ID,
    selected_analysts: tuple[str, ...] = TRADINGAGENTS_BASELINE_SELECTED_ANALYSTS,
    now: datetime | None = None,
    random_suffix: str | None = None,
) -> TARunIdentity:
    normalized_artifact_config = normalize_tradingagents_artifact_config(artifact_config)
    path_free_graph_config = (
        copy.deepcopy(graph_config)
        if graph_config is not None
        else build_tradingagents_graph_config()
    )
    config_key = build_tradingagents_config_key(
        path_free_graph_config,
        strategy_name=strategy_name,
        baseline_profile_id=baseline_profile_id,
        selected_analysts=selected_analysts,
    )
    run_key = resolve_tradingagents_run_key(
        normalized_artifact_config,
        now=now,
        random_suffix=random_suffix,
    )
    artifact_root_path, profile_name = _resolve_tradingagents_artifact_root(
        normalized_artifact_config["root"]
    )
    base_run_dir = (
        artifact_root_path
        / _safe_path_component(config_key)
        / _safe_path_component(run_key)
    )
    benchmark_results_dir = base_run_dir / "benchmark_results"

    return TARunIdentity(
        profile_name=profile_name,
        artifact_root=artifact_root_path,
        config_key=config_key,
        run_key=run_key,
        base_run_dir=base_run_dir,
        benchmark_results_dir=benchmark_results_dir,
    )


def resolve_tradingagents_run_key(
    artifact_config: Mapping[str, Any],
    *,
    now: datetime | None = None,
    random_suffix: str | None = None,
) -> str:
    explicit = artifact_config.get("run_key")
    if explicit:
        return str(explicit)

    now = now or datetime.now(timezone.utc)
    random_suffix = random_suffix or secrets.token_hex(3)
    return f"run_{now.strftime('%Y%m%dT%H%M%SZ')}_{random_suffix}"


def build_tradingagents_namespace(
    *,
    strategy_name: str,
    symbol: str,
    date_from: dt.date,
    date_to: dt.date,
    artifact_root: str,
    config_key: str,
    run_key: str,
) -> TAExperimentNamespace:
    artifact_root_path, profile_name = _resolve_tradingagents_artifact_root(
        artifact_root
    )
    safe_symbol = _safe_path_component(symbol)
    safe_config_key = _safe_path_component(config_key)
    safe_run_key = _safe_path_component(run_key)
    window_key = f"test_{date_from.isoformat()}_{date_to.isoformat()}"

    base_run_dir = artifact_root_path / safe_config_key / safe_run_key
    benchmark_results_dir = base_run_dir / "benchmark_results"
    launcher_dir = base_run_dir / "launcher"
    ticker_dir = base_run_dir / "tickers" / safe_symbol
    results_dir = base_run_dir / "runtime_results"
    data_cache_dir = base_run_dir / "runtime_cache"
    memory_log_path = ticker_dir / "memory" / "trading_memory.md"
    full_state_log_dir = ticker_dir / "full_state_logs"
    ticker_namespace_meta_path = ticker_dir / "ticker_namespace_meta.json"
    namespace_meta_path = base_run_dir / "namespace_meta.json"
    manifest_path = base_run_dir / "manifest.json"

    for directory in (
        base_run_dir,
        benchmark_results_dir,
        launcher_dir,
        ticker_dir,
        memory_log_path.parent,
        full_state_log_dir,
        results_dir,
        data_cache_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return TAExperimentNamespace(
        strategy_name=strategy_name,
        profile_name=profile_name,
        symbol=symbol,
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        artifact_root=str(artifact_root_path),
        config_key=config_key,
        run_key=run_key,
        window_key=window_key,
        base_run_dir=base_run_dir,
        benchmark_results_dir=benchmark_results_dir,
        launcher_dir=launcher_dir,
        ticker_dir=ticker_dir,
        results_dir=results_dir,
        data_cache_dir=data_cache_dir,
        memory_log_path=memory_log_path,
        full_state_log_dir=full_state_log_dir,
        ticker_namespace_meta_path=ticker_namespace_meta_path,
        namespace_meta_path=namespace_meta_path,
        manifest_path=manifest_path,
    )


def collect_tradingagents_env_override_fields_present() -> list[str]:
    present_fields: set[str] = set()

    for env_name, field_name in _PATH_ENV_TO_FIELD.items():
        if os.environ.get(env_name):
            present_fields.add(field_name)

    for env_name, field_name in default_config._ENV_OVERRIDES.items():  # noqa: SLF001
        if os.environ.get(env_name):
            present_fields.add(field_name)

    return sorted(present_fields)


def build_tradingagents_manifest_payload(
    *,
    window_config_input: Mapping[str, Any],
    artifact_config: Mapping[str, Any],
    graph_config: Mapping[str, Any],
    config_key: str,
    run_key: str,
    namespace: TAExperimentNamespace | None = None,
    strategy_name: str = TRADINGAGENTS_STRATEGY_NAME,
    baseline_profile_id: str = TRADINGAGENTS_BASELINE_PROFILE_ID,
    selected_analysts: tuple[str, ...] = TRADINGAGENTS_BASELINE_SELECTED_ANALYSTS,
) -> dict[str, Any]:
    payload = {
        "strategy_name": strategy_name,
        "window_config_input": _to_jsonable(window_config_input),
        "artifact_config": _to_jsonable(artifact_config),
        "baseline_profile_id": baseline_profile_id,
        "graph_config": _to_jsonable(_sanitize_run_scoped_graph_config(graph_config)),
        "config_key": config_key,
        "run_key": run_key,
        "selected_analysts": list(selected_analysts),
        "prompt_policy": _to_jsonable(_build_tradingagents_prompt_policy_payload()),
        "benchmark_execution_contract": _to_jsonable(
            _TRADINGAGENTS_EXECUTION_BRIDGE_CONTRACT
        ),
        "benchmark_reflection_contract": _to_jsonable(
            _TRADINGAGENTS_REFLECTION_BRIDGE_CONTRACT
        ),
        "fundamentals_payload_contract": _to_jsonable(
            _build_tradingagents_fundamentals_payload_contract()
        ),
        "env_override_fields_present": collect_tradingagents_env_override_fields_present(),
    }
    if namespace is not None:
        payload["namespace"] = namespace.to_run_meta_payload()
    return payload


class TradingAgentsStrategy(BaseStrategyIso):
    """Configuration-solidified benchmark adapter for TradingAgents."""

    def __init__(
        self,
        symbol: str,
        date_from: str | dt.date | dt.datetime,
        date_to: str | dt.date | dt.datetime,
        data_loader: Any = None,
        artifact_config: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.strategy_name = TRADINGAGENTS_STRATEGY_NAME
        self.symbol = symbol
        self.date_from = _to_date(date_from)
        self.date_to = _to_date(date_to)
        self.data_loader = data_loader
        self.current_date: dt.date | None = None
        self.pre_decision_position_state: str | None = None
        self.last_ta_rating: str | None = None
        self.last_folded_action: str | None = None
        self.last_execution_bridge_payload: dict[str, Any] | None = None
        self.runtime_adapter = None
        self.window_loader = None
        self.runtime_loader = None
        self.runtime_session_summary: dict[str, Any] | None = None
        self.trace_writer: TATraceWriter | None = None
        self._finalized = False

        data_loader_description = _describe_data_loader_input(data_loader)
        self.window_config_input = {
            "symbol": symbol,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "data_loader": data_loader_description,
        }
        self.run_window_config_input = {
            "date_from": self.date_from,
            "date_to": self.date_to,
            "data_loader": data_loader_description,
        }

        self.baseline_profile_id = TRADINGAGENTS_BASELINE_PROFILE_ID
        self.selected_analysts = TRADINGAGENTS_BASELINE_SELECTED_ANALYSTS
        self.artifact_config = normalize_tradingagents_artifact_config(artifact_config)
        self._graph_config_without_paths = build_tradingagents_graph_config()
        self.run_identity = materialize_tradingagents_run_identity(
            artifact_config=self.artifact_config,
            graph_config=self._graph_config_without_paths,
            strategy_name=self.strategy_name,
            baseline_profile_id=self.baseline_profile_id,
            selected_analysts=self.selected_analysts,
        )
        self.config_key = self.run_identity.config_key
        self.run_key = self.run_identity.run_key
        self.namespace = build_tradingagents_namespace(
            strategy_name=self.strategy_name,
            symbol=self.symbol,
            date_from=self.date_from,
            date_to=self.date_to,
            artifact_root=str(self.run_identity.artifact_root),
            config_key=self.config_key,
            run_key=self.run_key,
        )
        self.graph_config = build_tradingagents_graph_config(
            namespace_paths=self.namespace.as_graph_paths()
        )
        self.config = self.graph_config
        self.env_override_fields_present = (
            collect_tradingagents_env_override_fields_present()
        )
        _write_json_file(
            self.namespace.namespace_meta_path,
            {
                **self.namespace.to_run_meta_payload(),
                "artifact_enabled": self.artifact_config["enabled"],
            },
        )

        TradingAgentsGraph = _get_tradingagents_graph_class()
        runtime_tool_surfaces = build_runtime_tool_surfaces(self._runtime_dispatch_tool)
        self.graph = TradingAgentsGraph(
            selected_analysts=list(self.selected_analysts),
            config=self.graph_config,
            analyst_tool_surfaces=runtime_tool_surfaces,
            sentiment_prefetch_loader=self._runtime_load_sentiment_prefetch_blocks,
            instrument_context_builder=self._runtime_build_instrument_context,
            outcome_resolver=self._runtime_resolve_outcome,
        )
        self.window_loader, self.runtime_loader = _prepare_tradingagents_runtime_loaders(
            symbol=self.symbol,
            date_from=self.date_from,
            date_to=self.date_to,
            data_loader=data_loader,
        )
        trace_writer = (
            TATraceWriter(ticker_dir=self.namespace.ticker_dir)
            if self.artifact_config["enabled"]
            else None
        )
        self.runtime_adapter = TAOfflineSessionAdapter(
            symbol=self.symbol,
            date_from=self.date_from,
            date_to=self.date_to,
            window_loader=self.window_loader,
            runtime_loader=self.runtime_loader,
            memory_log=self.graph.memory_log,
            default_benchmark_ticker=str(
                self.graph_config.get("benchmark_ticker", "SPY")
            ),
            trace_writer=trace_writer,
        )
        self.trace_writer = self.runtime_adapter.trace_writer
        self.bind_runtime_adapter(self.runtime_adapter)
        self.runtime_session_summary = self.runtime_adapter.describe_session()
        _write_json_file(
            self.namespace.ticker_namespace_meta_path,
            {
                **self.namespace.to_ticker_meta_payload(),
                "artifact_enabled": self.artifact_config["enabled"],
                "window_config_input": _to_jsonable(self.window_config_input),
                "runtime_session_summary": _to_jsonable(self.runtime_session_summary),
            },
        )
        self.manifest_preview = build_tradingagents_manifest_payload(
            window_config_input=self.run_window_config_input,
            artifact_config=self.artifact_config,
            graph_config=self.graph_config,
            config_key=self.config_key,
            run_key=self.run_key,
            namespace=self.namespace,
            strategy_name=self.strategy_name,
            baseline_profile_id=self.baseline_profile_id,
            selected_analysts=self.selected_analysts,
        )
        _write_json_file(self.namespace.manifest_path, self.manifest_preview)

        self.logger.info(
            "Initialised %s baseline shell for %s (%s -> %s).",
            self.strategy_name,
            self.symbol,
            self.date_from,
            self.date_to,
        )

    def _build_execution_bridge_payload(self, raw_rating: str, framework: Any) -> dict[str, str]:
        pre_decision_position_state = infer_tradingagents_position_state(
            framework,
            self.symbol,
        )
        payload = build_tradingagents_execution_bridge_payload(
            raw_rating=raw_rating,
            pre_decision_position_state=pre_decision_position_state,
        )
        self.pre_decision_position_state = pre_decision_position_state
        self.last_ta_rating = payload["raw_rating"]
        self.last_folded_action = payload["executed_action"]
        self.last_execution_bridge_payload = payload
        return payload

    def _execute_execution_bridge_action(
        self,
        *,
        date: dt.date,
        reference_price: float,
        framework: Any,
        bridge_payload: Mapping[str, str],
    ) -> None:
        # This method submits the benchmark-side order intent on the decision
        # day. FINSABER still resolves the actual fill on the next trading
        # day's open, so the bridge payload is intentionally pre-fill.
        executed_action = bridge_payload["executed_action"]

        if executed_action == "submit_buy_all":
            framework.buy(date, self.symbol, reference_price, -1)
        elif executed_action == "submit_sell_all":
            sell_quantity = int(framework.portfolio[self.symbol]["quantity"])
            framework.sell(date, self.symbol, reference_price, sell_quantity)
        elif executed_action != "noop":
            raise ValueError(f"Unsupported executed_action: {executed_action!r}")

        self.last_execution_bridge_payload = {**bridge_payload, "reference_price": reference_price}

    def _require_runtime_adapter_bound(self):
        if self.runtime_adapter is None:
            raise RuntimeError("TradingAgentsStrategy requires a bound runtime adapter.")
        return self.runtime_adapter

    def _runtime_dispatch_tool(self, method: str, *args, **kwargs) -> str:
        return self._require_runtime_adapter_bound().dispatch_tool(method, *args, **kwargs)

    def _runtime_load_sentiment_prefetch_blocks(
        self,
        ticker: str,
        trade_date: str | dt.date | dt.datetime,
    ) -> dict[str, str]:
        return self._require_runtime_adapter_bound().load_sentiment_prefetch_blocks(
            ticker,
            trade_date,
        )

    def _runtime_build_instrument_context(
        self,
        ticker: str,
        asset_type: str = "stock",
    ) -> str:
        return self._require_runtime_adapter_bound().build_instrument_context(
            ticker,
            asset_type,
        )

    def _runtime_resolve_outcome(
        self,
        *,
        trade_date: str,
        holding_days: int,
        benchmark: str,
    ):
        return self._require_runtime_adapter_bound().resolve_outcome(
            trade_date,
            holding_days,
            benchmark,
        )

    def _apply_execution_bridge(
        self,
        *,
        raw_rating: str,
        date: dt.date,
        reference_price: float,
        framework: Any,
    ) -> dict[str, Any]:
        bridge_payload = self._build_execution_bridge_payload(raw_rating, framework)
        self._execute_execution_bridge_action(
            date=date,
            reference_price=reference_price,
            framework=framework,
            bridge_payload=bridge_payload,
        )
        return self.last_execution_bridge_payload or bridge_payload

    def bind_runtime_adapter(self, runtime_adapter: Any) -> None:
        self.runtime_adapter = runtime_adapter
        self.graph.bind_runtime_adapter(runtime_adapter)

    def on_data(
        self,
        date: dt.date,
        today_data: Mapping[str, Any],
        framework: Any,
    ):
        self.current_date = date
        self.pre_decision_position_state = infer_tradingagents_position_state(
            framework,
            self.symbol,
        )
        if self.runtime_adapter is None:
            raise RuntimeError("TradingAgentsStrategy requires a bound runtime adapter.")
        self.runtime_adapter.bind_decision_day(
            date,
            pre_decision_position_state=self.pre_decision_position_state,
        )
        final_state, raw_rating = self.graph.propagate(
            self.symbol,
            date.isoformat(),
            asset_type="stock",
        )
        if not isinstance(final_state, Mapping):
            raise TypeError(
                "TradingAgentsGraph.propagate(...) must return a mapping-like final state."
            )
        final_trade_decision = final_state.get("final_trade_decision")
        if not isinstance(final_trade_decision, str) or not final_trade_decision.strip():
            raise ValueError(
                "TradingAgentsGraph.propagate(...) must return a non-empty "
                "'final_trade_decision' string."
            )

        reference_price = extract_tradingagents_reference_price(
            today_data,
            symbol=self.symbol,
        )
        bridge_payload = self._apply_execution_bridge(
            raw_rating=raw_rating,
            date=date,
            reference_price=reference_price,
            framework=framework,
        )
        self.last_execution_bridge_payload = bridge_payload
        return None

    def train(self):
        return None

    def finalize_backtest_artifacts(self, framework_status: bool):
        if self._finalized:
            return None
        if not self.namespace.manifest_path.exists():
            _write_json_file(self.namespace.manifest_path, self.manifest_preview)
        self._finalized = True
        return None
