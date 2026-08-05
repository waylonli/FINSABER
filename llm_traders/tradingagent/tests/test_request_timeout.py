"""Tests for finite OpenAI-compatible request timeouts."""

import pytest

from tradingagents.llm_clients.openai_client import OpenAIClient


@pytest.mark.unit
def test_openai_client_uses_finite_default_request_timeout():
    llm = OpenAIClient(
        model="gpt-4o-mini",
        provider="openai",
        api_key="placeholder",
    ).get_llm()

    assert llm.request_timeout == 600.0
    assert llm.root_client.timeout == 600.0


@pytest.mark.unit
def test_openai_client_preserves_explicit_request_timeout():
    llm = OpenAIClient(
        model="gpt-4o-mini",
        provider="openai",
        api_key="placeholder",
        timeout=30.0,
    ).get_llm()

    assert llm.request_timeout == 30.0
    assert llm.root_client.timeout == 30.0
