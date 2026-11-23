import unittest
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agents.client_research_agent import build_client_research_agent


class ResearchTestModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


with patch("utils.models.make_model", return_value=ResearchTestModel(responses=[AIMessage(content="Ready")])):
    from agents.deployable_agents.client_research.agent import build_deployable_agent


class RootRunRecorder(BaseCallbackHandler):
    def __init__(self):
        self.names = []

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:
            self.names.append(kwargs.get("name"))


class AgentRunNameTests(unittest.TestCase):
    def test_named_graphs_support_invoke_and_stream_with_thread_id(self):
        for builder, expected_name in (
            (build_client_research_agent, "client_research_agent"),
            (build_deployable_agent, "client_research"),
        ):
            for method in ("invoke", "stream"):
                with self.subTest(builder=builder.__name__, method=method):
                    graph = builder(model=ResearchTestModel(responses=[AIMessage(content="Ready")]))
                    self.assertEqual(graph.config["run_name"], expected_name)
                    recorder = RootRunRecorder()
                    config = {"configurable": {"thread_id": method}, "callbacks": [recorder]}
                    inputs = {"messages": [("user", "Hello")]}
                    if method == "invoke":
                        result = graph.invoke(inputs, config=config)
                    else:
                        result = list(graph.stream(inputs, config=config, stream_mode="values"))[-1]
                    self.assertEqual(result["messages"][-1].content, "Ready")
                    self.assertEqual(recorder.names, [expected_name])


if __name__ == "__main__":
    unittest.main()
