from crewai import Agent, Task, Crew
from a2a_universal.adapters.crewai_tool import a2a_hello

researcher = Agent(role="Researcher", goal="Use the a2a_hello tool", backstory="Enjoys pinging services", tools=[a2a_hello])
result = Crew(agents=[researcher], tasks=[Task(description="Ping", agent=researcher)]).kickoff()
print(result)
