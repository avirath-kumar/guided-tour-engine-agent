import os
import re
import unittest
from uuid import uuid4

from agents.client_research_agent import build_client_research_agent
from utils.models import API_KEY_ENV, make_model


@unittest.skipUnless(os.getenv(API_KEY_ENV or ""), "Requires model gateway credentials")
class ClientResearchScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.deployable_agents.client_research.agent import (
            build_deployable_agent,
        )

        cls.builders = (build_client_research_agent, build_deployable_agent)

    def invoke(self, builder, request):
        agent = builder(model=make_model())
        return agent.invoke(
            {"messages": [{"role": "user", "content": request}]}, config={"configurable": {"thread_id": str(uuid4())}}
        )["messages"]

    def test_client_holdings_remain_supported(self):
        for builder in self.builders:
            with self.subTest(agent=builder.__name__):
                messages = self.invoke(
                    builder, "Look up client CL-0042 and tell me their name and top holdings with weights."
                )
                answer = messages[-1].text
                self.assertIn("Acme Pension Fund", answer)
                for ticker, weight in (("AAPL", "8"), ("MSFT", "7"), ("JPM", "5"), ("XOM", "4")):
                    self.assertIn(ticker, answer)
                    self.assertRegex(answer, rf"{weight}(?:\.0)?\s*%")
                self.assertTrue(
                    any(message.type == "tool" and message.name == "get_client_profile" for message in messages)
                )

    def test_unrelated_requests_are_declined(self):
        requests = (
            "Write a poem about the ocean.",
            "Book me a flight from New York to Paris next Friday.",
            "Prepare my personal income tax return for this year.",
        )
        for builder in self.builders:
            for request in requests:
                with self.subTest(agent=builder.__name__, request=request):
                    messages = self.invoke(builder, request)
                    answer = messages[-1].text
                    self.assertRegex(answer.lower(), r"can't|cannot|unable|outside|only|don't|do not")
                    self.assertRegex(answer.lower(), r"client|portfolio|holdings|market|company research|meeting")
                    self.assertLessEqual(len(re.split(r"[.!?]+\s+", answer.strip())), 2)
                    self.assertLessEqual(len(answer.split()), 100)
                    self.assertNotIn("\n", answer.strip())
                    self.assertFalse(any(getattr(message, "tool_calls", None) for message in messages))


if __name__ == "__main__":
    unittest.main()
