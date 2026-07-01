import pytest

from llm_traders.tradingagent.tradingagents.agents.analysts.fundamentals_analyst import (
    _build_fundamentals_system_message,
)
from llm_traders.tradingagent.tradingagents.agents.analysts.market_analyst import (
    _build_market_system_message,
)
from llm_traders.tradingagent.tradingagents.agents.analysts.news_analyst import (
    _build_news_system_message,
)


@pytest.mark.unit
def test_market_prompt_marks_na_and_local_history_limits_explicitly():
    message = _build_market_system_message()

    assert "If a tool reports `N/A`, unavailable values, or visibly truncated local history" in message
    assert "Do not backfill pre-window history" in message
    assert "Do not claim historical validation" in message


@pytest.mark.unit
def test_news_prompt_treats_global_macro_gaps_as_limitations():
    message = _build_news_system_message("company")

    assert "current state of the world that is relevant for trading and macroeconomics" in message
    assert "get_news(ticker, start_date, end_date)" in message
    assert "returned get_news(...) content as the available local news evidence" in message
    assert "do not assume additional unseen articles beyond the tool output" in message
    assert "If global or macro news is unavailable, disabled, or empty" in message
    assert "do not infer missing macro developments" in message


@pytest.mark.unit
def test_fundamentals_prompt_is_filing_based_and_not_profile_based():
    message = _build_fundamentals_system_message()

    assert isinstance(message, str)
    assert "using only the filing-based narrative context and statement proxies" in message
    assert "Do not invent missing company profile facts, valuation ratios, market-cap figures" in message
    assert "narrative context should come from `get_fundamentals`" in message


@pytest.mark.unit
def test_analyst_prompt_builders_do_not_expand_past_context_scope():
    messages = (
        _build_market_system_message(),
        _build_news_system_message("company"),
        _build_fundamentals_system_message(),
    )

    assert all("past_context" not in message for message in messages)
