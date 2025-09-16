from langgraph.graph import StateGraph, END
from a2a_universal.adapters.langgraph_node import A2ANode

sg = StateGraph(dict)
sg.add_node("call_a2a", A2ANode())
sg.set_entry_point("call_a2a")
sg.add_edge("call_a2a", END)
app = sg.compile()
print(app.invoke({"input": "What is the capital of Italy?"}))
