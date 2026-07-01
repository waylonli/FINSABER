from __future__ import annotations

from typing import Sequence

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def _build_fundamentals_system_message() -> str:
    return (
        "You are a researcher tasked with analyzing fundamental information about a company using only the filing-based narrative context and statement proxies that are explicitly available as of the current analysis date. Please write a comprehensive report of the company's visible fundamentals from local filings, including business context, risk context, management discussion, and the latest visible annual or quarterly statement proxies when available. Use the available tools: `get_fundamentals` for filing-based company analysis, and `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for statement-specific annual or quarterly proxies."
        + " If a filing section, statement proxy, or broader company-overview field is unavailable, state that limitation explicitly. Do not invent missing company profile facts, valuation ratios, market-cap figures, or filing periods that are not supported by tool output."
        + " Make sure to respect the tool semantics: narrative context should come from `get_fundamentals`, while statement tools provide the latest visible annual/quarterly proxy instead of a full database extract."
        + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
        + get_language_instruction()
    )


def create_fundamentals_analyst(llm, tools: Sequence[BaseTool] | None = None):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        available_tools = list(
            tools
            or [
                get_fundamentals,
                get_balance_sheet,
                get_cashflow,
                get_income_statement,
            ]
        )

        system_message = _build_fundamentals_system_message()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(
            tool_names=", ".join([tool.name for tool in available_tools])
        )
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(available_tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
