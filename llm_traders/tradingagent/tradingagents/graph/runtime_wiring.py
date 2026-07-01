from __future__ import annotations

from typing import Annotated, Callable, Mapping, Sequence

from langchain_core.tools import BaseTool, tool


RuntimeDispatch = Callable[..., str]


_EXPECTED_SURFACE_NAMES = {
    "market": (
        "get_stock_data",
        "get_indicators",
        "get_verified_market_snapshot",
    ),
    "news": (
        "get_news",
        "get_global_news",
    ),
    "fundamentals": (
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
    ),
}


def build_runtime_tool_surfaces(
    dispatch_tool: RuntimeDispatch,
) -> dict[str, list[BaseTool]]:
    """Build benchmark-local tool bundles backed by a runtime dispatch seam."""

    @tool
    def get_stock_data(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """Retrieve stock price data (OHLCV) for a given ticker symbol."""
        return dispatch_tool("get_stock_data", symbol, start_date, end_date)

    @tool
    def get_indicators(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """Retrieve one or more technical indicators for a given ticker symbol."""
        return dispatch_tool(
            "get_indicators",
            symbol,
            indicator,
            curr_date,
            look_back_days,
        )

    @tool
    def get_verified_market_snapshot(
        symbol: Annotated[str, "ticker symbol of the company"],
        curr_date: Annotated[str, "the current trading date, YYYY-mm-dd"],
        look_back_days: Annotated[
            int, "number of recent trading rows to include for sanity-checking"
        ] = 30,
    ) -> str:
        """Return the benchmark-local verification snapshot for exact market claims."""
        return dispatch_tool(
            "get_verified_market_snapshot",
            symbol,
            curr_date,
            look_back_days,
        )

    @tool
    def get_news(
        ticker: Annotated[str, "Ticker symbol"],
        start_date: Annotated[
            str,
            "Requested start date in yyyy-mm-dd format; benchmark-local mode may ignore it and return the authoritative visible local news window instead",
        ],
        end_date: Annotated[
            str,
            "Requested end date in yyyy-mm-dd format; benchmark-local mode may ignore it and return the authoritative visible local news window instead",
        ],
    ) -> str:
        """Request benchmark-local ticker news. The returned local news block is authoritative even if the requested window is wider or different."""
        return dispatch_tool("get_news", ticker, start_date, end_date)

    @tool
    def get_global_news(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        look_back_days: Annotated[int | None, "Days to look back"] = None,
        limit: Annotated[int | None, "Max articles to return"] = None,
    ) -> str:
        """Retrieve benchmark-local global news output or disabled placeholder."""
        return dispatch_tool("get_global_news", curr_date, look_back_days, limit)

    @tool
    def get_fundamentals(
        ticker: Annotated[str, "ticker symbol"],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ) -> str:
        """Retrieve filing-based local fundamentals context for a ticker."""
        return dispatch_tool("get_fundamentals", ticker, curr_date)

    @tool
    def get_balance_sheet(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
    ) -> str:
        """Retrieve the local statement proxy for balance-sheet requests."""
        return dispatch_tool("get_balance_sheet", ticker, freq, curr_date)

    @tool
    def get_cashflow(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
    ) -> str:
        """Retrieve the local statement proxy for cashflow requests."""
        return dispatch_tool("get_cashflow", ticker, freq, curr_date)

    @tool
    def get_income_statement(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
    ) -> str:
        """Retrieve the local statement proxy for income-statement requests."""
        return dispatch_tool("get_income_statement", ticker, freq, curr_date)

    return {
        "market": [
            get_stock_data,
            get_indicators,
            get_verified_market_snapshot,
        ],
        "news": [
            get_news,
            get_global_news,
        ],
        "fundamentals": [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ],
    }


def validate_runtime_tool_surfaces(
    selected_analysts: Sequence[str],
    tool_surfaces: Mapping[str, Sequence[BaseTool]],
    data_policy: Mapping[str, object] | None = None,
) -> None:
    """Validate benchmark-local analyst tool surfaces and social guardrails."""

    policy = dict(data_policy or {})
    allow_social = bool(policy.get("allow_social", True))

    if "social" in selected_analysts and not allow_social:
        raise ValueError(
            "Benchmark-local graph wiring forbids selecting `social` when "
            "`allow_social` is false."
        )

    for analyst_key in selected_analysts:
        if analyst_key not in _EXPECTED_SURFACE_NAMES:
            continue

        tools = tool_surfaces.get(analyst_key)
        if not tools:
            raise ValueError(
                f"Missing benchmark-local tool surface for analyst `{analyst_key}`."
            )

        actual_names = tuple(tool.name for tool in tools)
        expected_names = _EXPECTED_SURFACE_NAMES[analyst_key]
        if actual_names != expected_names:
            raise ValueError(
                f"Benchmark-local tool surface mismatch for `{analyst_key}`: "
                f"expected {expected_names}, got {actual_names}."
            )


def validate_runtime_benchmark_seams(
    *,
    tool_surfaces: Mapping[str, Sequence[BaseTool]] | None,
    instrument_context_builder,
    outcome_resolver,
) -> None:
    """Validate the minimum benchmark-local runtime seams for graph wiring."""

    if not tool_surfaces:
        return

    if instrument_context_builder is None:
        raise ValueError(
            "Benchmark-local graph wiring requires `instrument_context_builder` "
            "when runtime tool surfaces are injected."
        )

    if outcome_resolver is None:
        raise ValueError(
            "Benchmark-local graph wiring requires `outcome_resolver` when "
            "runtime tool surfaces are injected."
        )
