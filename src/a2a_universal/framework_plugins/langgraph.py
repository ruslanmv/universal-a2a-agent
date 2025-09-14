from __future__ import annotations
from typing import Any

from ..frameworks import FrameworkBase, _call_provider, _extract_last_user_text

class Framework(FrameworkBase):
    id = "langgraph"
    name = "LangGraph Framework"

    def __init__(self, provider, **kwargs):
        super().__init__(provider)
        # Try to build a minimal graph if langgraph is installed; otherwise fallback at execute time
        try:
            from langgraph.graph import StateGraph, END, MessagesState  # type: ignore
            from langchain_core.messages import HumanMessage, AIMessage  # type: ignore

            sg = StateGraph(MessagesState)

            async def node(state: dict[str, Any]) -> dict[str, Any]:
                last = state["messages"][-1]
                user_text = getattr(last, "content", "")
                reply = await _call_provider(self.provider, user_text, [])
                return {"messages": [AIMessage(content=reply)]}

            sg.add_node("a2a", node)
            sg.add_edge("__start__", "a2a")
            sg.add_edge("a2a", END)
            self._app = sg.compile()
            self.ready = True
            self.reason = ""
        except Exception as e:  # langgraph not installed or error
            self._app = None
            self.ready = True  # still usable via fallback
            self.reason = f"LangGraph unavailable, falling back to direct calls: {e}"

    async def execute(self, messages: list[dict[str, Any]]) -> str:
        # If we have a compiled app, run it; else direct provider call
        if getattr(self, "_app", None) is not None:
            try:
                from langchain_core.messages import HumanMessage  # type: ignore
                out = await self._app.ainvoke({"messages": [HumanMessage(content=_extract_last_user_text(messages))]})
                return out["messages"][-1].content
            except Exception as e:
                return f"[langgraph error] {e}"
        # Fallback
        text = _extract_last_user_text(messages)
        return await _call_provider(self.provider, text, messages)
