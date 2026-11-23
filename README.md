# Guided tour agent (for LangSmith Engine)

The client research deep agent from the [LangSmith guided tour](https://github.com/langchain-samples/langsmith-guided-tour), on its own. Connect this repo in **Engine → Configure Engine → Connect your agent's code repository** so Engine can point its proposed fixes at the code and open pull requests here.

| Path | What it is |
|---|---|
| `agents/client_research_agent.py` | The agent: tools (`get_client_profile`, `web_search`), subagents, prompt |
| `agents/deployable_agents/client_research/` | The deployed variant: `AGENTS.md`, skills, backends |
| `utils/models.py` | Model setup (Anthropic through the LangSmith LLM Gateway) |
| `utils/search.py` | Web search; without `TAVILY_API_KEY` it returns built-in offline results |

The notebooks, evaluations and setup live in the guided tour repo; this repo only holds the agent code Engine reads.
