# File: examples/crewai_watsonx_duo.py
import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, LLM
from a2a_universal.adapters.crewai_base_tool import A2AHelloTool

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()

# Required environment variables for Watsonx
model_id = os.getenv("MODEL_ID", "ibm/granite-3-8b-instruct")  # ✅ updated default
project_id = os.environ.get("WATSONX_PROJECT_ID")
url = os.environ.get("WATSONX_URL")
api_key = os.environ.get("WATSONX_API_KEY")

if not all([project_id, url, api_key]):
    raise RuntimeError(
        "Missing Watsonx credentials. Please set WATSONX_PROJECT_ID, WATSONX_URL, and WATSONX_API_KEY in your .env file."
    )

# -------------------------------------------------------------------
# Watsonx LLM (CrewAI-native via LiteLLM provider)
# -------------------------------------------------------------------
# NOTE: CrewAI’s `LLM` expects a LiteLLM-compatible model string.
# For watsonx, you must prefix with "watsonx/"
watsonx_llm = LLM(
    model=f"watsonx/{model_id}",
    api_key=api_key,
    base_url=url,           # LiteLLM uses api_base/base_url for endpoint
    temperature=0.0,
    max_tokens=2048,
    project_id=project_id,  # forwarded to watsonx provider
)

# -------------------------------------------------------------------
# Main Crew workflow
# -------------------------------------------------------------------
if __name__ == "__main__":
    topic = "Edge AI for autonomous drones in search & rescue"

    # Shared A2A Tool
    a2a_tool = A2AHelloTool()
    a2a_tool.base_url = os.getenv("A2A_BASE", "http://localhost:8000")

    # Researcher Agent
    researcher = Agent(
        role="Researcher",
        goal="Gather concise, accurate notes and outline the topic.",
        backstory="Methodical analyst who drafts clean bullet-point notes.",
        tools=[a2a_tool],
        llm=watsonx_llm,   # ✅ use CrewAI-native Watsonx LLM
        allow_delegation=False,
        verbose=True,
    )

    # Writer Agent
    writer = Agent(
        role="Writer",
        goal="Turn notes into a tidy LaTeX article (1–2 pages).",
        backstory="Technical writer who produces compilable LaTeX.",
        tools=[a2a_tool],
        llm=watsonx_llm,   # ✅ use CrewAI-native Watsonx LLM
        allow_delegation=False,
        verbose=True,
    )

    # Research Task
    t_research = Task(
        description=(
            f"Research the topic: '{topic}'. "
            "Use the a2a_hello tool to produce a concise outline with bullet points, "
            "covering background, key challenges, approaches, and example applications. "
            "Output: a Markdown outline."
        ),
        agent=researcher,
        expected_output="A clean Markdown outline of findings.",
    )

    # Writing Task (depends on research)
    t_write = Task(
        description=(
            "Using the outline from the Researcher, write a compilable LaTeX article. "
            "Use the a2a_hello tool to help with prose and LaTeX formatting. "
            "Return only the final .tex content."
        ),
        agent=writer,
        context=[t_research],
        expected_output="A single LaTeX .tex string, compilable.",
    )

    # Crew assembly & execution
    crew = Crew(agents=[researcher, writer], tasks=[t_research, t_write])
    result = crew.kickoff()

    print("\n=== FINAL LATEX ===\n")
    print(result)
