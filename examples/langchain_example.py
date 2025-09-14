from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI  # or any LangChain LLM
from a2a_universal.adapters.langchain_tool import a2a_hello

llm = ChatOpenAI(model="gpt-4o-mini")  # replace with your provider/model
agent = initialize_agent(tools=[a2a_hello], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print(agent.run("Call a2a_hello with 'ping'."))
