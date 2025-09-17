"""
Minimal demo: Watsonx.ai (LangChain chat model) + Universal A2A tool.

- Uses your .env (LLM_PROVIDER=watsonx, AGENT_FRAMEWORK=native, WATSONX_* creds).
- Builds the correct LangChain model via provider_api.langchain_llm().
- Uses the built-in a2a_hello tool (no re-decoration).

Requirements:
  pip install -U langchain langchain-ibm python-dotenv httpx
"""

import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType

# 1) Get the right LangChain ChatModel for your provider (watsonx -> ChatWatsonx)
from a2a_universal.provider_api import langchain_llm

# 2) Use the built-in Universal A2A tool (already a valid LangChain Tool)
from a2a_universal.adapters.langchain_tool import a2a_hello

def main() -> None:
    # Load WATSONX_* from .env (project root)
    load_dotenv()

    # Optional: defaults for the A2A server (tool will use this)
    # If your server listens elsewhere, set A2A_BASE_URL in .env.
    os.environ.setdefault("A2A_BASE_URL", "http://localhost:8000")
    # Some HTTP clients expect an API key for OpenAI-style endpoints; a dummy is fine.
    os.environ.setdefault("OPENAI_API_KEY", "sk-local-dummy")

    # Build the LangChain chat model for Watsonx from env
    # (does NOT care that AGENT_FRAMEWORK=native; it just returns a ChatModel).
    llm = langchain_llm()  # -> langchain_ibm.ChatWatsonx

    # Assemble a simple ReAct-style chat agent
    agent = initialize_agent(
        tools=[a2a_hello],                               # Universal A2A tool
        llm=llm,                                         # Watsonx chat model
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    # Demo: ask the agent to use the tool
    out = agent.invoke({"input": "Use the a2a_hello tool to say 'ping'."})
    print("\n[Final Answer]:", out.get("output", ""))

if __name__ == "__main__":
    main()
