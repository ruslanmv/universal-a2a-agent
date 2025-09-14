from crewai import Agent, Task, Crew
from a2a_universal.adapters.crewai_base_tool import A2AHelloTool

a2a_tool = A2AHelloTool()
planner = Agent(role="Planner", goal="Plan a greeting", backstory="", tools=[a2a_tool])
executor = Agent(role="Executor", goal="Send the greeting", backstory="", tools=[a2a_tool])

crew = Crew(agents=[planner, executor], tasks=[
    Task(description="Draft the greeting", agent=planner),
    Task(description="Send greeting via A2A", agent=executor)
])
print(crew.kickoff())
