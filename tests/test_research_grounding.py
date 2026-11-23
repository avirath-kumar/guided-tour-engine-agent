"""Behavior checks for research grounding; model checks require the configured gateway key."""

import importlib
import os
import unittest
from unittest.mock import patch

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agents import client_research_agent as client
from utils.models import API_KEY_ENV, make_model
from utils.search import OFFLINE_TAG, resilient_tavily_search

URL = "https://example.com/results"
CONTEXT = "Orion makes industrial sensors. Demand depends on factory investment."
LIVE = f"**Orion results**\n{URL}\n{CONTEXT} Revenue was $17.3 million."
UNSUPPORTED = 'Revenue was $99.9 million. CEO Vale said: "Our best quarter ever."'


@tool
def fixture_search(query: str) -> str:
    """Return controlled research evidence."""
    return LIVE


def recording_search(calls):
    """Build a search fixture that records the model's search budget usage."""

    @tool
    def fixture_search(query: str) -> str:
        """Look up one research topic."""
        calls.append(query)
        return f"{OFFLINE_TAG}\n{URL}\n{CONTEXT}"

    return fixture_search


def configurations():
    """Read the configurations supplied by both production builders."""
    with patch.object(client, "create_deep_agent", side_effect=lambda **kwargs: kwargs):
        standard = client.build_client_research_agent(
            model=object(), search_tool=fixture_search
        )
    with (
        patch("utils.models.make_model", return_value=object()),
        patch("deepagents.create_deep_agent", side_effect=lambda **kwargs: kwargs),
    ):
        deployed = importlib.import_module(
            "agents.deployable_agents.client_research.agent"
        )
        deployment = deployed.build_deployable_agent(object())
    return standard, deployment


class SearchAvailabilityTests(unittest.TestCase):
    def test_offline_result_identifies_background(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}):
            result = resilient_tavily_search("apple")
        self.assertTrue(result.startswith(OFFLINE_TAG))
        self.assertIn("https://", result)

    def test_failed_live_search_identifies_fallback(self):
        with patch("utils.search.TavilyClient") as search:
            search.return_value.search.side_effect = RuntimeError("unavailable")
            result = resilient_tavily_search(
                "apple", client=search.return_value, max_retries=0
            )
        self.assertTrue(result.startswith("[fallback content; live search unavailable"))
        self.assertIn("https://", result)


@unittest.skipUnless(
    API_KEY_ENV is None or os.getenv(API_KEY_ENV),
    "Model gateway credential unavailable",
)
class ResearchGroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_model(temperature=0)
        cls.configs = configurations()

    def respond(self, prompt, result, delegated=False):
        """Run the real model against a completed search or delegated research turn."""
        name = "task" if delegated else "fixture_search"
        agent = create_deep_agent(model=self.model, tools=[], system_prompt=prompt)
        messages = [
            HumanMessage(
                content="Brief me on Orion using the completed research. Include revenue and a CEO quote."
            ),
            AIMessage(
                content="", tool_calls=[{"name": name, "args": {}, "id": "evidence"}]
            ),
            ToolMessage(content=result, tool_call_id="evidence", name=name),
            HumanMessage(content="Give the brief now without further tools."),
        ]
        response = agent.invoke({"messages": messages}, {"recursion_limit": 20})
        return response["messages"][-1].text

    def prompts(self):
        """Yield both research and main prompts from their production configurations."""
        for index, config in enumerate(self.configs):
            yield f"main-{index}", config["system_prompt"]
            yield f"research-{index}", config["subagents"][0]["system_prompt"]

    def test_live_figure_keeps_supporting_url(self):
        for name, prompt in self.prompts():
            with self.subTest(prompt=name):
                answer = self.respond(prompt, LIVE)
                self.assertIn("17.3", answer)
                self.assertIn(URL, answer)
                self.assertNotIn("99.9", answer)
                self.assertNotIn("Our best quarter ever", answer)

    def test_absent_figures_and_nonverbatim_quotes_are_omitted(self):
        for name, prompt in self.prompts():
            with self.subTest(prompt=name):
                answer = self.respond(
                    prompt,
                    f"**Orion**\n{URL}\n{CONTEXT} CEO Vale discussed factory demand.",
                )
                self.assertNotRegex(
                    answer, r"\$\s*\d|\d+(?:\.\d+)?\s*(?:million|billion|%)"
                )
                self.assertNotIn("Our best quarter ever", answer)
                self.assertNotRegex(answer, r"[“\"][^”\"]+[”\"]")
                self.assertIn(URL, answer)

    def test_background_and_fallback_are_qualitative(self):
        for name, prompt in self.prompts():
            for tag in (
                OFFLINE_TAG,
                "[fallback content; live search unavailable -- RuntimeError: offline]",
            ):
                with self.subTest(prompt=name, tag=tag):
                    answer = self.respond(prompt, f"{tag}\n{LIVE}")
                    self.assertRegex(
                        answer.lower(),
                        r"live.*(?:unavailable|not available|could not|not accessible)",
                    )
                    self.assertNotIn("17.3", answer)
                    self.assertNotIn("99.9", answer)
                    self.assertIn("sensor", answer.lower())
                    self.assertIn(URL, answer)

    def test_main_agents_omit_unsupported_delegated_claims(self):
        for index, config in enumerate(self.configs):
            with self.subTest(main=index):
                evidence = f"Search result: {LIVE}\nSubagent summary: {UNSUPPORTED}"
                answer = self.respond(config["system_prompt"], evidence, delegated=True)
                self.assertIn("17.3", answer)
                self.assertIn(URL, answer)
                self.assertNotIn("99.9", answer)
                self.assertNotIn("Our best quarter ever", answer)

    def test_research_search_budget(self):
        for index, config in enumerate(self.configs):
            with self.subTest(research=index):
                research = config["subagents"][0]
                calls = []
                search_tool = recording_search(calls)
                if index == 1:
                    search_tool = search_tool.model_copy(update={"name": "web_search"})
                agent = create_deep_agent(
                    model=self.model,
                    tools=[search_tool],
                    system_prompt=research["system_prompt"],
                )
                agent.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content="Research Orion, Apple, Microsoft, Nvidia, and JPMorgan. Search separately for each company then summarize the evidence."
                            )
                        ]
                    },
                    {"recursion_limit": 30},
                )
                self.assertGreater(len(calls), 0)
                self.assertLessEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
