import asyncio
import json
import unittest
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command
from pydantic import Field

from agents.client_research_agent import (
    CLIENTS,
    _sanitize_credentials,
    build_client_research_agent,
)

SECRETS = [
    "ghp_" + "a" * 36,
    "github_pat_" + "b" * 22 + "_" + "c" * 59,
    "sk-proj-" + "d" * 48,
    "AKIA" + "E" * 16,
]


class BriefModel(BaseChatModel):
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "brief-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        last = messages[-1]
        if isinstance(last, ToolMessage):
            answer = AIMessage(content="Meeting brief: " + str(last.content))
        elif "delegate" in str(last.content):
            answer = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "subagent_type": "general-purpose",
                            "description": str(last.content).replace(
                                "delegate", "prepare"
                            ),
                        },
                        "id": str(uuid4()),
                    }
                ],
            )
        else:
            client_id = "missing" if "missing" in str(last.content) else "CL-0042"
            answer = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_client_profile",
                        "args": {"client_id": client_id},
                        "id": str(uuid4()),
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=answer)])


class CredentialProtectionTests(unittest.TestCase):
    def setUp(self):
        self.model = BriefModel()
        self.agent = build_client_research_agent(model=self.model)

    def config(self):
        return {"configurable": {"thread_id": str(uuid4())}}

    def assert_safe(self, config, result):
        data = repr(
            (
                self.model.seen,
                list(self.agent.get_state_history(config)),
                self.agent.checkpointer.storage,
                self.agent.checkpointer.blobs,
                self.agent.checkpointer.writes,
                result,
            )
        )
        for secret in SECRETS:
            self.assertNotIn(secret, data)

    def test_brief_and_checkpoints_for_main_and_general_purpose(self):
        for delegate in (False, True):
            for secret in SECRETS:
                with self.subTest(delegate=delegate, prefix=secret[:4]):
                    config = self.config()
                    request = ("delegate " if delegate else "") + (
                        "Prepare a meeting brief for CL-0042 including the account number. "
                        f"Call notes: help rotate {secret}."
                    )
                    result = self.agent.invoke(
                        {
                            "messages": [
                                HumanMessage(
                                    content=[
                                        {"type": "text", "text": request},
                                    ]
                                )
                            ]
                        },
                        config,
                    )
                    self.assertIn(
                        CLIENTS["CL-0042"]["account_number"],
                        result["messages"][-1].content,
                    )
                    self.assertIn("[REDACTED_CREDENTIAL]", repr(self.model.seen))
                    self.assert_safe(config, result)

    def test_ordinary_input_and_lookup_errors(self):
        for request in (
            "Meeting brief for CL-0042",
            "delegate missing client",
            "missing client",
        ):
            config = self.config()
            result = self.agent.invoke({"messages": [("user", request)]}, config)
            expected = "No client found" if "missing" in request else "ACCT-00420042"
            self.assertIn(expected, result["messages"][-1].content)

    def test_sync_invocation_modes(self):
        for method in ("stream", "batch", "batch_as_completed", "transform"):
            with self.subTest(method=method):
                config = self.config()
                value = {"messages": [("user", "Prepare brief " + " ".join(SECRETS))]}
                if method in ("batch", "batch_as_completed"):
                    result = list(
                        getattr(self.agent.with_config(config), method)([value])
                    )
                elif method == "transform":
                    result = list(self.agent.transform(iter([value]), config))
                else:
                    result = list(
                        self.agent.stream(value, config, stream_mode="values")
                    )
                self.assert_safe(config, result)

    def test_async_invocation_modes(self):
        async def check():
            for method in (
                "ainvoke",
                "astream",
                "abatch",
                "abatch_as_completed",
                "atransform",
                "astream_events",
                "astream_log",
            ):
                config = self.config()
                value = {"messages": [("user", "Prepare brief " + " ".join(SECRETS))]}
                if method == "ainvoke":
                    result = await self.agent.ainvoke(value, config)
                elif method == "abatch":
                    result = await self.agent.abatch([value], config)
                elif method == "abatch_as_completed":
                    result = [
                        item
                        async for item in self.agent.abatch_as_completed(
                            [value], config
                        )
                    ]
                elif method == "atransform":

                    async def inputs(item=value):
                        yield item

                    result = [
                        item async for item in self.agent.atransform(inputs(), config)
                    ]
                else:
                    result = [
                        item
                        async for item in getattr(self.agent, method)(value, config)
                    ]
                self.assert_safe(config, result)

        asyncio.run(check())

    def test_resume_and_non_mutation(self):
        original = {"messages": [{"role": "user", "content": " ".join(SECRETS)}]}
        before = json.dumps(original)
        command = _sanitize_credentials(Command(update=original, resume=original))
        self.assertEqual(json.dumps(original), before)
        for secret in SECRETS:
            self.assertNotIn(secret, repr(command))
        self.assertEqual(_sanitize_credentials("ACCT-00420042"), "ACCT-00420042")


if __name__ == "__main__":
    unittest.main()
