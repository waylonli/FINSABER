import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from backtest.toolkit.llm_cost_monitor import (
    get_llm_cost,
    get_llm_cost_ledger,
    reset_llm_cost,
)
from tradingagents.llm_clients.openai_client import (
    NormalizedChatOpenAI,
    OpenAIClient,
    _extract_usage_from_chat_result,
)


def _chat_result_from_usage(*, llm_output=None, usage_metadata=None):
    message = AIMessage(content="ok")
    if usage_metadata is not None:
        message.usage_metadata = usage_metadata
    return ChatResult(
        generations=[ChatGeneration(message=message)],
        llm_output=llm_output,
    )


@pytest.mark.unit
def test_extract_usage_prefers_llm_output_token_usage():
    result = _chat_result_from_usage(
        llm_output={
            "token_usage": {
                "input_tokens": 17,
                "output_tokens": 9,
                "total_tokens": 26,
            }
        },
        usage_metadata={
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )

    assert _extract_usage_from_chat_result(result) == (17, 9)


@pytest.mark.unit
def test_extract_usage_falls_back_to_message_usage_metadata():
    result = _chat_result_from_usage(
        llm_output=None,
        usage_metadata={
            "input_tokens": 23,
            "output_tokens": 11,
            "total_tokens": 34,
        },
    )

    assert _extract_usage_from_chat_result(result) == (23, 11)


@pytest.mark.unit
def test_generate_records_llm_cost_for_openai_provider(monkeypatch):
    fake_result = _chat_result_from_usage(
        llm_output={
            "token_usage": {
                "input_tokens": 120,
                "output_tokens": 40,
                "total_tokens": 160,
            }
        }
    )

    monkeypatch.setattr(
        NormalizedChatOpenAI.__bases__[0],
        "_generate",
        lambda self, messages, stop=None, run_manager=None, **kwargs: fake_result,
    )

    reset_llm_cost()
    client = NormalizedChatOpenAI(model="gpt-4o-mini", api_key="placeholder")
    client._tradingagents_provider = "openai"

    result = client._generate([HumanMessage(content="hello")])

    assert result is fake_result
    ledger = get_llm_cost_ledger()
    assert len(ledger) == 1
    assert ledger[0]["model"] == "gpt-4o-mini"
    assert ledger[0]["prompt_tokens"] == 120
    assert ledger[0]["completion_tokens"] == 40
    assert get_llm_cost() > 0.0


@pytest.mark.unit
def test_generate_skips_cost_record_for_non_openai_provider(monkeypatch):
    fake_result = _chat_result_from_usage(
        llm_output={
            "token_usage": {
                "input_tokens": 120,
                "output_tokens": 40,
                "total_tokens": 160,
            }
        }
    )

    monkeypatch.setattr(
        NormalizedChatOpenAI.__bases__[0],
        "_generate",
        lambda self, messages, stop=None, run_manager=None, **kwargs: fake_result,
    )

    reset_llm_cost()
    client = NormalizedChatOpenAI(model="deepseek-v4-flash", api_key="placeholder")
    client._tradingagents_provider = "deepseek"

    client._generate([HumanMessage(content="hello")])

    assert get_llm_cost_ledger() == []
    assert get_llm_cost() == 0.0


@pytest.mark.unit
def test_openai_client_sets_runtime_provider_on_returned_llm():
    client = OpenAIClient(
        model="gpt-4o-mini",
        provider="openai",
        api_key="placeholder",
    )

    llm = client.get_llm()

    assert getattr(llm, "_tradingagents_provider", None) == "openai"
