# examples/crewai_example.py
from __future__ import annotations

import os
import sys

# Load environment variables FIRST, before importing any local modules that use them.
from dotenv import load_dotenv
load_dotenv()


from crewai import Agent, Task, Crew

from a2a_universal.provider_api import crew_llm   # <-- import directly from provider_api
from a2a_universal.adapters.crewai_tool import A2ATool
from a2a_universal.provider_api import llm as model  # auto-chooses CrewAI LLM by env

if __name__ == "__main__":
    # Get CrewAI LLM via active provider (e.g., LLM_PROVIDER=watsonx)
    try:
        #llm = crew_llm()
        llm = model()  # -> CrewAI LLM with model="watsonx/<MODEL_ID>", creds from env
    except Exception as e:
        sys.stderr.write(f"[fatal] Could not initialize LLM: {e}\n")
        sys.exit(1)

    city = os.getenv("CITY", "Genova, Italy")

    # Tool (A2A backend)
    weather_tool = A2ATool(
        name="a2a_weather",
        description="Get the current weather forecast via Universal A2A.",
        skill="weather",
    )

    # Agents
    meteorologist = Agent(
        role="Meteorologist",
        goal=f"Summarize actionable weather for {city}",
        backstory="A meticulous forecaster who turns raw data into practical guidance.",
        tools=[weather_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    planner = Agent(
        role="Itinerary Planner",
        goal=f"Create an enjoyable one-day plan in {city} adapted to the weather.",
        backstory="A savvy local guide who balances walking, food, and culture.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # Tasks
    t_weather = Task(
        description=(
            f"Use the weather tool to gather today's conditions for {city}. "
            "Return a concise bullet list including: high/low temp (°C), rain likelihood, wind, "
            "UV level, and any alerts. End with 3 clothing/gear recommendations."
        ),
        agent=meteorologist,
        expected_output=(
            "Markdown bullets: temps, precipitation chance, wind, UV, alerts; plus 3-item gear list."
        ),
    )

    t_plan = Task(
        description=(
            "Based on the meteorologist summary, write a practical one-day plan: breakfast, morning walk, "
            "lunch, afternoon museum/indoor option if rainy, coffee break, sunset spot, dinner. "
            "Add weather-aware advice (e.g., indoor swaps if rain, shaded routes if hot). "
            "Return a tidy Markdown itinerary with timestamps."
        ),
        agent=planner,
        context=[t_weather],
        expected_output="A Markdown day-plan with times, locations, and weather-aware alternatives.",
    )

    crew = Crew(agents=[meteorologist, planner], tasks=[t_weather, t_plan], verbose=True)
    result = crew.kickoff()

    print("\n=== FINAL ITINERARY ===\n")
    print(result)
