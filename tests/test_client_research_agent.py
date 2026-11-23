import unittest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agents.client_research_agent import build_client_research_agent


class SearchThenDelegateModel(BaseChatModel):
    available_tools: tuple[str, ...] = ()
    subagent_type: str = "research-agent"

    @property
    def _llm_type(self):
        return "search-then-delegate-test"

    def bind_tools(self, tools, **kwargs):
        names = [
            item["function"]["name"] if isinstance(item, dict) else item.name
            for item in tools
        ]
        return self.model_copy(update={"available_tools": tuple(names)})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if isinstance(last, ToolMessage) and last.name == "task":
            response = AIMessage(content=str(last.content))
        elif "task" in self.available_tools and (
            "web_search" not in self.available_tools or isinstance(last, ToolMessage)
        ):
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Research Example Company",
                            "subagent_type": self.subagent_type,
                        },
                        "id": "delegate",
                    }
                ],
            )
        elif isinstance(last, ToolMessage):
            response = AIMessage(content=str(last.content))
        else:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "Example Company"},
                        "id": "search",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=response)])


class ClientResearchAgentTests(unittest.TestCase):
    def test_research_delegates_without_main_agent_search(self):
        self.assert_delegated_search("research-agent")

    def test_general_purpose_subagent_retains_search_access(self):
        self.assert_delegated_search("general-purpose")

    def assert_delegated_search(self, subagent_type):
        searches = []

        @tool
        def web_search(query: str) -> str:
            """Search for company information."""
            searches.append(query)
            return "Example Company research findings"

        agent = build_client_research_agent(
            model=SearchThenDelegateModel(subagent_type=subagent_type),
            search_tool=web_search,
        )
        result = agent.invoke(
            {"messages": [("user", "Research Example Company in depth")]},
            config={"configurable": {"thread_id": subagent_type}},
        )

        calls = [
            call["name"]
            for message in result["messages"]
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]
        self.assertEqual(calls, ["task"])
        self.assertEqual(searches, ["Example Company"])
        self.assertIn(
            "Example Company research findings", result["messages"][-1].content
        )


if __name__ == "__main__":
    unittest.main()
