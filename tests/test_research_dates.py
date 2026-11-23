import asyncio
import importlib
import re
import unittest
from datetime import datetime
from unittest.mock import patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from agents.client_research_agent import build_client_research_agent


class ResearchModel(BaseChatModel):
    requests: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "research-date-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(messages)
        user = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        system = messages[0].text
        if isinstance(messages[-1], ToolMessage):
            response = AIMessage(content=messages[-1].content)
        elif "You are a research assistant" in system:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": user},
                        "id": "search",
                    }
                ],
            )
        else:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"subagent_type": "research-agent", "description": user},
                        "id": "research",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=response)])


class ResearchDateTests(unittest.TestCase):
    def setUp(self):
        self.queries = []

        @tool
        def web_search(query: str) -> str:
            """Return research from a local fixture."""
            self.queries.append(query)
            return "Fixture research for " + query

        self.search = web_search

    def assert_research_context(self, model, date):
        self.assertGreaterEqual(len(model.requests), 4)
        for messages in model.requests:
            system = messages[0].text
            self.assertIn("Today is " + date, system)
            self.assertIn("latest, recent, this month, and this quarter", system)
            self.assertIn("Preserve periods explicitly supplied by the user", system)
            self.assertIn("do not add a year or specific quarter", system)
            self.assertIn("determine the most recent available period", system)
        self.assertTrue(any("You are a research assistant" in r[0].text for r in model.requests))

    def test_regular_agent_refreshes_date_and_preserves_request_period(self):
        model = ResearchModel()
        with patch("agents.client_research_agent.datetime") as clock:
            clock.now.return_value = datetime(2025, 1, 1)
            agent = build_client_research_agent(model=model, search_tool=self.search)
            for index, request in enumerate(
                (
                    "Find the latest Apple news",
                    "Research Apple conditions this month",
                    "Research Apple in Q2 2024",
                )
            ):
                with self.subTest(request=request):
                    model.requests.clear()
                    clock.now.return_value = datetime(2026, 11, 23)
                    result = agent.invoke(
                        {"messages": [("user", request)]},
                        {"configurable": {"thread_id": str(index)}},
                    )
                    self.assert_research_context(model, "2026-11-23")
                    self.assertEqual(self.queries[-1], request)
                    self.assertIn(request, result["messages"][-1].content)
                    delegations = [
                        call["args"]["description"]
                        for r in model.requests
                        for m in r
                        if isinstance(m, AIMessage)
                        for call in m.tool_calls
                        if call["name"] == "task"
                    ]
                    self.assertIn(request, delegations)
                    if "2024" not in request:
                        self.assertFalse(re.search(r"\b20\d{2}\b", self.queries[-1]))

    def test_deployable_graph_refreshes_date_without_rebuilding(self):
        model = ResearchModel()
        with patch("utils.models.make_model", return_value=model):
            module = importlib.import_module("agents.deployable_agents.client_research.agent")
        with (
            patch("agents.client_research_agent.datetime") as clock,
            patch(
                "agents.client_research_agent.resilient_tavily_search",
                side_effect=lambda query, **kwargs: "Fixture research for " + query,
            ),
        ):
            for date in (datetime(2026, 11, 23), datetime(2027, 1, 2)):
                with self.subTest(date=date):
                    model.requests.clear()
                    clock.now.return_value = date
                    request = "Research the latest Apple news this month"
                    result = asyncio.run(module.agent.ainvoke({"messages": [("user", request)]}))
                    self.assert_research_context(model, date.strftime("%Y-%m-%d"))
                    self.assertIn(request, result["messages"][-1].content)
                    self.assertIn("Delegate web research", model.requests[0][0].text)


if __name__ == "__main__":
    unittest.main()
