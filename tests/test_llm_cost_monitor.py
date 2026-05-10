from types import SimpleNamespace

import pytest

from backtest.toolkit.llm_cost_monitor import (
    add_openai_cost_from_response,
    add_openai_cost_from_tokens_count,
    get_llm_cost,
    get_llm_cost_ledger,
    reset_llm_cost,
)


def test_llm_cost_ledger_tracks_token_costs():
    reset_llm_cost()

    first = add_openai_cost_from_tokens_count("gpt-4o-mini", prompt_tokens=1_000, generated_tokens=500)
    second = add_openai_cost_from_response(
        {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 1_000, "completion_tokens": 100},
        }
    )

    assert first == pytest.approx(0.00045)
    assert second == pytest.approx(0.0035)
    assert get_llm_cost() == pytest.approx(0.00395)
    assert [record["model"] for record in get_llm_cost_ledger()] == ["gpt-4o-mini", "gpt-4o"]


def test_llm_cost_accepts_object_style_responses_and_output_tokens_alias():
    reset_llm_cost()
    response = SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(prompt_tokens=2_000, output_tokens=1_000),
    )

    cost = add_openai_cost_from_response(response)

    assert cost == pytest.approx(0.0009)
    assert get_llm_cost_ledger()[0]["completion_tokens"] == 1000
