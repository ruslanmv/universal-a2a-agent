<!-- =======================================================================
  Universal A2A Agent (Python) — README
  Professional edition with badges, logos, and end-to-end instructions
  License: Apache-2.0
======================================================================= -->

# Universal A2A Agent (Python)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/FastAPI-Ready-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Kubernetes-Helm_Chart-326CE5?logo=kubernetes&logoColor=white" alt="Kubernetes">
  <img src="https://img.shields.io/badge/LangChain-Tool_Adapter-1BB46E?logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/LangGraph-Native_MessagesState-000000?logo=diagram&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/CrewAI-Tool_%26_BaseTool-5E5E5E?logo=readme&logoColor=white" alt="CrewAI">
  <img src="https://img.shields.io/badge/AutoGen-Function_Tool-8A2BE2?logo=python&logoColor=white" alt="AutoGen">
  <img src="https://img.shields.io/badge/BeeAI-A2A_Agent-111111?logo=apache&logoColor=white" alt="BeeAI">
  <img src="https://img.shields.io/badge/MCP-Gateway_Ready-0A84FF?logo=protocol&logoColor=white" alt="MCP">
  <a href="https://github.com/agent-matrix/matrix-hub"><img src="https://img.shields.io/badge/MatrixHub-Ready-brightgreen?logo=matrix&logoColor=white" alt="MatrixHub Ready"></a>
</p>

<p align="center">
  <!-- Replace "ruslanmv/universal-a2a-agent" with your real GitHub org/repo -->
  <a href="https://github.com/ruslanmv/universal-a2a-agent/actions/workflows/ci.yml">
    <img alt="Build Status" src="https://github.com/ruslanmv/universal-a2a-agent/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img alt="License" src="https://img.shields.io/badge/license-Apache_2.0-blue.svg">
  </a>
  <a href="https://github.com/ruslanmv/universal-a2a-agent/releases/latest">
    <img alt="Latest Release" src="https://img.shields.io/github/v/release/ruslanmv/universal-a2a-agent">
  </a>
</p>

---

*A production-ready, framework-agnostic A2A service with first-class support for MatrixHub, MCP Gateway, LangChain, **LangGraph (native MessagesState)**, CrewAI, BeeAI Framework, AutoGen, and Watsonx Orchestrate. A private enterprise adapter is included behind env flags (undocumented by design).*

> **What this gives you in 60 seconds**
>
> *One tiny service, many ecosystems.* You deploy **one** HTTP agent and plug it into:
>
> * **App frameworks**: LangChain, **LangGraph** (native), CrewAI, AutoGen, Bee/BeeAI.
> * **Orchestrators**: MCP Gateway, Watsonx Orchestrate.
> * **Ops**: Docker image + Helm chart for Kubernetes.
>
> **Mental model**: *Your app sends a message → the service replies → adapters translate glue code so every platform “just works.”*

---

🌐 What is Universal A2A?

Universal A2A (Agent-to-Agent) is a framework-agnostic, provider-agnostic bridge layer that standardizes how agents talk to each other, tools, and orchestrators.

Think of it as a universal adapter for AI agents:

* It abstracts away differences between frameworks (LangGraph, CrewAI, AutoGen, BeeAI, …)
* It abstracts away differences between model providers (OpenAI, Anthropic Claude, Gemini, Ollama, watsonx.ai, Bedrock, Azure OpenAI, …)
* It exposes common APIs (A2A RPC, JSON-RPC, OpenAI-compatible endpoints) so any external client or orchestrator can connect without custom glue code.

---

🎯 The Scope of the Project

* Normalize communication between heterogeneous AI frameworks and providers.
* Provide a standard runtime (FastAPI server) with consistent endpoints:

  * `/a2a` → raw A2A calls
  * `/rpc` → JSON-RPC 2.0
  * `/openai/v1/chat/completions` → OpenAI-compatible
  * `/enterprise/v1/agent` → private/enterprise adapter
* Pluggable system: drop in a new provider or framework via a plugin folder—no core code changes needed.
* Production-ready containerization: ships with Dockerfile, Compose, CI/CD workflows for DockerHub + GHCR.

---

🧩 What Problem Does It Solve?

Today, the agent ecosystem is fragmented:

* Frameworks (LangChain, CrewAI, LangGraph, AutoGen) each expect a different interface.
* Providers (OpenAI, Anthropic, Ollama, Gemini, Bedrock) each return results in different shapes.
* Orchestrators (UI layers, pipelines, workflows) often assume OpenAI’s API.

➡️ Universal A2A solves this by unifying them all into one universal surface.

* Developers can swap providers (e.g. OpenAI → Ollama) without rewriting agents.
* Enterprises can standardize APIs for internal orchestration, regardless of what the underlying LLM is.
* Framework authors can plug in seamlessly without reinventing transport layers.

---

🔑 Why Is It Important?

* **Interoperability** → Agents, frameworks, and providers can talk to each other with minimal friction.
* **Portability** → Build an agent once, run it anywhere (local, cloud, enterprise).
* **Future-proofing** → As new frameworks/providers emerge, they just need a plugin, not a fork of the server.
* **Enterprise readiness** → Health checks, structured logging, auth adapters, Docker CI/CD out of the box.

In short:

👉 Universal A2A is the “universal translator” for AI agents.

It ensures that no matter what model or framework you use, your agents can speak the same language and be deployed in a reliable, production-ready way.

---

## Table of contents

* [Project tree](#project-tree)
* [Quick start](#quick-start)
* [Installation](#installation)
* [Environment](#environment)
* [Endpoints](#endpoints)
* [Agent Card](#agent-card)
* [Create new agents (no provider file needed)](#create-new-agents-no-provider-file-needed)
* [Providers & Frameworks (runtime selection)](#providers--frameworks-runtime-selection)
* [Adapters & examples](#adapters--examples)

  * [LangChain](#langchain)
  * [LangGraph (native MessagesState)](#langgraph-native-messagesstate)
  * [LangGraph (legacy dict-state)](#langgraph-legacy-dict-state)
  * [CrewAI](#crewai)
  * [Bee / BeeAI](#bee--beeai)
  * [AutoGen](#autogen)
* [Quick Start — watsonx.ai backed examples](#quick-start--watsonxai-backed-examples)
* [Testing & CI](#testing--ci)
* [Docker & Helm](#docker--helm)
* [MCP Gateway](#mcp-gateway)
* [watsonx Orchestrate](#watsonx-orchestrate)
* [watsonx.ai (Studio) — Agent example](#watsonxai-studio--agent-example)
* [MatrixHub integration](#matrixhub-integration)
* [Security & ops notes](#security--ops-notes)
* [Versioning](#versioning)
* [Contributing](#contributing)
* [License](#license)

---

## Project tree

```
src/
  a2a_universal/
    adapters/
    frameworks/
    providers/
    client.py
    server.py
    runner.py
    app.py
...
examples/
  tiny_agent.py
  langgraph_agent_example.py
  debugger_graph.py
```

---

## Quick start

```bash
# 1) clone
git clone https://github.com/ruslanmv/universal-a2a-agent.git
cd universal-a2a-agent

# 2) (optional) venv
python -m venv .venv && source .venv/bin/activate

# 3) install core (or extras below)
pip install -e .
# optional bundles
# pip install -e .[all]
# pip install -e .[openai]
# pip install -e .[watsonx]
# pip install -e .[langgraph]
# pip install -e .[crewai]

# 4) pick a Provider + Framework (any combo)
export LLM_PROVIDER=echo             # echo|openai|watsonx|ollama|anthropic|gemini|azure_openai|bedrock
export AGENT_FRAMEWORK=native        # native|langgraph|crewai

# 5) run the server (development)
uvicorn a2a_universal.server:app --host 0.0.0.0 --port 8000 --reload

# 6) smoke test (A2A JSON-RPC)
curl -s http://localhost:8000/rpc -H 'Content-Type: application/json' -d '{
  "jsonrpc":"2.0",
  "id":"1",
  "method":"message/send",
  "params":{ "message": {"role":"user","parts":[{"text":"ping"}]}}
}' | jq

# readiness (shows provider+framework)
curl -s http://localhost:8000/readyz | jq
```

> If you can hit the **smoke test**, you can integrate **any** adapter below. All adapters call into the same `/rpc`/`/a2a` logic.

---

## Installation

```bash
# Core (FastAPI server + CLI + client)
pip install -e .

# Optional extras as needed
pip install -e .[openai]
pip install -e .[watsonx]
pip install -e .[langgraph]
pip install -e .[crewai]
# Everything
pip install -e .[all]
```

---

## Environment

Create a `.env` or export vars:

```env
# Core server
HOST=0.0.0.0
PORT=8000
PUBLIC_URL=http://localhost:8000
A2A_ROOT_PATH=

# Selection (runtime injection)
LLM_PROVIDER=echo                  # Provider id
AGENT_FRAMEWORK=native             # Framework id

# CORS (tighten for prod)
CORS_ALLOW_ORIGINS=*
CORS_ALLOW_CREDENTIALS=false
CORS_ALLOW_METHODS=*
CORS_ALLOW_HEADERS=*

# Provider creds (examples)
OPENAI_API_KEY=
WATSONX_API_KEY=
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=
MODEL_ID=ibm/granite-3-3-8b-instruct

# Private adapter (enterprise, disabled by default)
PRIVATE_ADAPTER_ENABLED=false        # true|false
PRIVATE_ADAPTER_AUTH_SCHEME=NONE     # NONE|BEARER|API_KEY
PRIVATE_ADAPTER_AUTH_TOKEN=
PRIVATE_ADAPTER_PATH=/enterprise/v1/agent
```

> **Production**: set `PUBLIC_URL` to your **HTTPS** origin so the Agent Card advertises the correct `/rpc` endpoint. Use `A2A_ROOT_PATH` if you deploy behind a path prefix.

---

## Endpoints

* `POST /rpc` — **JSON-RPC 2.0**

  ```json
  {
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [ { "text": "hello" } ]
      }
    }
  }
  ```

  * `GET /rpc` returns a small JSON help page (no 405).
  * `HEAD/OPTIONS /rpc` return `204` with `Allow: POST, OPTIONS`.

* `POST /a2a` — **Raw A2A** envelope

  ```json
  {
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "messageId": "m1",
        "parts": [ { "text": "hello" } ]
      }
    }
  }
  ```

* `POST /openai/v1/chat/completions` — **OpenAI-style** chat for UIs/orchestrators

  ```json
  { "model":"universal-a2a-hello", "messages":[{"role":"user","content":"hello"}] }
  ```

* `GET /.well-known/agent-card.json` — **Agent Card** discovery

* `GET /healthz` — liveness

* `GET /readyz` — readiness (provider & framework)

> **Which one should I use?**
>
> * **LangGraph/LangChain/CrewAI/AutoGen**: use `/rpc` or `/a2a` via adapters.
> * **Orchestrators & chat UIs**: `/openai/v1/chat/completions` is the universal bridge.

---

## Agent Card

`/.well-known/agent-card.json` (excerpt):

```json
{
  "protocolVersion": "0.3.0",
  "name": "Universal A2A Hello",
  "description": "Greets the user and echoes input",
  "version": "1.2.0",
  "preferredTransport": "JSONRPC",
  "url": "https://your-host/rpc",
  "capabilities": { "streaming": false, "pushNotifications": false },
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "skills": [
    { "id": "say-hello", "name": "Say Hello", "description": "Responds with a friendly greeting." }
  ]
}
```

> *Card compatibility*: `capabilities` is an **object**; default modes use **MIME** types (`text/plain`).

---

## Create new agents (no provider file needed)

The fastest path is a **single-function handler** — no custom provider class required.

```python
# examples/tiny_agent.py
from a2a_universal import mount, run

async def handle_text(text: str) -> str:
    return f"Hello from my custom agent. You said: {text}"

# Build a full A2A app from one function (card, /a2a, /rpc, /openai, health)
app = mount(handler=handle_text, name="Tiny Agent", description="One function → full A2A")

if __name__ == "__main__":
    # Default: mode='attach' → your app at '/', Universal A2A under '/a2a'
    run("__main__:app", host="0.0.0.0", port=8000, reload=False)
```

**Test (JSON-RPC):**

```bash
curl -s http://localhost:8000/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "method":"message/send",
    "params": {"message": {"role":"user","parts":[{"text":"My name is Ruslan"}]}},
    "id":"1"
  }' | jq
```

**Input flexibility**: the server accepts `{ "text": "..." }`, `{ "kind":"text","text":"..." }`, or `{ "type":"text","text":"..." }`. Responses always return `parts: [{"text":"..."}]`.

---

## Providers & Frameworks (runtime selection)

Pick any **Provider** and any **Framework** at deploy time:

```bash
export LLM_PROVIDER=openai           # or watsonx|ollama|anthropic|gemini|azure_openai|bedrock|echo
export AGENT_FRAMEWORK=langgraph     # or native|crewai
```

* `providers.py` builds the selected Provider plugin.
* `frameworks.py` builds the selected Framework and injects the Provider.
* The server routes requests to `await framework.execute(messages)`.
* `/readyz` reports readiness and reasons for both provider & framework.

3rd-party plugins can register via setuptools **entry points**:

* Providers: `a2a_universal.providers`
* Frameworks: `a2a_universal.frameworks`

---

## Adapters & examples

### LangChain

```python
# src/a2a_universal/adapters/langchain_tool.py
from langchain.tools import tool
from ..client import A2AClient

@tool("a2a_hello", return_direct=False)
def a2a_hello(text: str, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False) -> str:
    return A2AClient(base_url).send(text, use_jsonrpc=use_jsonrpc)
```

Example:

```python
# examples/langchain_example.py
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from a2a_universal.adapters.langchain_tool import a2a_hello

llm = ChatOpenAI(model="gpt-4o-mini")
agent = initialize_agent([a2a_hello], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print(agent.run("Call a2a_hello with 'ping'."))
```

---

### LangGraph (native MessagesState)

```python
# src/a2a_universal/adapters/langgraph_agent.py
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage
from ..client import A2AClient

class A2AAgentNode:
    def __init__(self, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False):
        self.client = A2AClient(base_url); self.use_jsonrpc = use_jsonrpc
    def __call__(self, state: MessagesState) -> dict[str, object]:
        last = state["messages"][-1]
        user_text = getattr(last, "content", "") if last else ""
        reply = self.client.send(user_text, use_jsonrpc=self.use_jsonrpc)
        return {"messages": [AIMessage(content=reply)]}
```

Example:

```python
# examples/langgraph_agent_example.py
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage
from a2a_universal.adapters.langgraph_agent import A2AAgentNode

g = StateGraph(MessagesState)
g.add_node("a2a", A2AAgentNode(base_url="http://localhost:8000"))
g.add_edge("__start__", "a2a")
g.add_edge("a2a", END)
app = g.compile()

out = app.invoke({"messages": [HumanMessage(content="ping from LangGraph")]})
print(out["messages"][-1].content)
```

---

### LangGraph (legacy dict-state)

```python
# src/a2a_universal/adapters/langgraph_node.py
from ..client import A2AClient

class A2ANode:
    def __init__(self, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False):
        self.client = A2AClient(base_url); self.use_jsonrpc = use_jsonrpc
    def __call__(self, state: dict) -> dict:
        reply = self.client.send(state.get("input", ""), use_jsonrpc=self.use_jsonrpc)
        return {**state, "a2a_reply": reply}
```

---

### CrewAI

```python
# src/a2a_universal/adapters/crewai_tool.py
try:
    from crewai_tools import tool
except Exception:
    def tool(fn=None, **kwargs):
        def wrap(f): return f
        return wrap if fn is None else wrap(fn)

from ..client import A2AClient

@tool("a2a_hello")
def a2a_hello(text: str, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False) -> str:
    return A2AClient(base_url).send(text, use_jsonrpc=use_jsonrpc)
```

BaseTool variant:

```python
# src/a2a_universal/adapters/crewai_base_tool.py
from ..client import A2AClient
class A2AHelloTool:
    name = "a2a_hello"
    description = "Send text to the Universal A2A agent and return the reply."
    base_url = "http://localhost:8000"
    use_jsonrpc = False
    def _run(self, text: str) -> str:
        return A2AClient(self.base_url).send(text, use_jsonrpc=self.use_jsonrpc)
```

---

### Bee / BeeAI

```python
# src/a2a_universal/adapters/bee_tool.py
from ..client import A2AClient

def a2a_call(text: str, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False) -> str:
    return A2AClient(base_url).send(text, use_jsonrpc=use_jsonrpc)
```

```python
# src/a2a_universal/adapters/beeai_agent.py
from beeai_framework.adapters.a2a.agents.agent import A2AAgent as BeeA2AAgent
from beeai_framework.memory import UnconstrainedMemory

def make_beeai_agent(base_url: str = "http://localhost:8000") -> BeeA2AAgent:
    return BeeA2AAgent(
        agent_card_url=f"{base_url}/.well-known/agent-card.json",
        memory=UnconstrainedMemory(),
    )
```

---

### AutoGen

```python
# src/a2a_universal/adapters/autogen_tool.py
try:
    from autogen import register_function
except Exception:
    def register_function(*args, **kwargs):
        def deco(f): return f
        return deco

from ..client import A2AClient

@register_function("a2a_hello", description="Send text to A2A agent and return reply")
def a2a_hello(text: str, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False) -> str:
    return A2AClient(base_url).send(text, use_jsonrpc=use_jsonrpc)
```

---

## Quick Start — watsonx.ai backed examples

Run the server with watsonx.ai:

```bash
pip install -e .[watsonx]
export LLM_PROVIDER=watsonx
export WATSONX_API_KEY=YOUR_KEY
export WATSONX_URL=https://us-south.ml.cloud.ibm.com
export WATSONX_PROJECT_ID=YOUR_PROJECT_ID
export MODEL_ID=ibm/granite-3-3-8b-instruct
export AGENT_FRAMEWORK=native
uvicorn a2a_universal.server:app --port 8000
curl -s localhost:8000/readyz | jq
```

**LangChain (tool → A2A)**

```python
# examples/quickstart_langchain_watsonx.py
import httpx
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

BASE = "http://localhost:8000"

def a2a_call(prompt: str) -> str:
    payload = {
        "method": "message/send",
        "params": {"message": {"role": "user", "messageId": "lc-tool", "parts": [{"text": prompt}]}}
    }
    r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    # read canonical result shape
    result = data.get("result", data)
    message = result.get("message", result)
    for p in (message.get("parts") or []):
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            return p["text"]
    return ""

tool = Tool(name="a2a_hello", description="Call Universal A2A (watsonx-backed)", func=a2a_call)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = initialize_agent([tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

if __name__ == "__main__":
    print(agent.run("Use the a2a_hello tool to say hello to LangChain."))
```

**LangGraph (node → A2A)**

```python
# examples/quickstart_langgraph_watsonx.py
import asyncio, httpx
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

BASE = "http://localhost:8000"

async def a2a_send(text: str) -> str:
    payload = {
        "method": "message/send",
        "params": {"message": {"role": "user", "messageId": "lg-node", "parts": [{"text": text}]}}
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{BASE}/a2a", json=payload)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", data)
        message = result.get("message", result)
        for p in (message.get("parts") or []):
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                return p["text"]
    return ""

async def a2a_node(state: dict) -> dict:
    last = state["messages"][-1]
    user_text = getattr(last, "content", "")
    reply = await a2a_send(user_text)
    return {"messages": [AIMessage(content=reply)]}

g = StateGraph(MessagesState)
g.add_node("a2a", a2a_node)
g.add_edge("__start__", "a2a")
g.add_edge("a2a", END)
app = g.compile()

async def main():
    out = await app.ainvoke({"messages": [HumanMessage(content="ping from LangGraph")]})
    print(out["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```

**CrewAI (tool → A2A)**

```python
# examples/quickstart_crewai_watsonx.py
import httpx
from crewai import Agent, Task, Crew

BASE = "http://localhost:8000"

def a2a_call(prompt: str) -> str:
    payload = {
        "method": "message/send",
        "params": {"message": {"role": "user", "messageId": "crewai-tool", "parts": [{"text": prompt}]}}
    }
    r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    result = data.get("result", data)
    message = result.get("message", result)
    for p in (message.get("parts") or []):
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            return p["text"]
    return ""

if __name__ == "__main__":
    researcher = Agent(
        role="Researcher",
        goal="Use the A2A tool (watsonx-backed) to answer user prompts",
        backstory="Loves calling external agents.",
        tools=[a2a_call],
    )

    task = Task(description="Say hello to CrewAI and summarize the word 'ping' in one sentence.", agent=researcher)
    result = Crew(agents=[researcher], tasks=[task]).kickoff()
    print(result)
```

---

## Testing & CI

Local tests:

```bash
pip install pytest httpx
pytest -q
```

GitHub Actions sample: `.github/workflows/ci.yml`

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e .[langgraph]
      - run: uvicorn a2a_universal.server:app --host 0.0.0.0 --port 8000 &
      - run: sleep 2 && pip install pytest httpx && pytest -q
```

---

## Docker & Helm

Build and run:

```bash
docker build -t ruslanmv/universal-a2a-agent:1.2.0 .
docker run --rm -p 8000:8000 -e PUBLIC_URL=http://localhost:8000 ruslanmv/universal-a2a-agent:1.2.0
```

Helm install:

```bash
helm upgrade --install a2a ./deploy/helm/universal-a2a-agent \
  --set image.repository=ruslanmv/universal-a2a-agent \
  --set image.tag=1.2.0 \
  --set env.PUBLIC_URL=https://a2a.example.com
```

> **Ops checklist**
>
> * Start with `/healthz` for both liveness/readiness; split to `/readyz` as needed.
> * Expose only the paths you need at your ingress; put OpenAI route behind auth if Internet-facing.
> * Set `A2A_ROOT_PATH` when serving behind a path prefix.

---

## MCP Gateway

Two common paths:

1. **A2A Registry** (if available): register your agent’s card URL and endpoint.
2. **REST Tool**: post to `/rpc` and map `result.message.parts[].text` to MCP outputs.

If your Gateway supports Agent Cards, point it at `/.well-known/agent-card.json`.

---

## watsonx Orchestrate

Use the **OpenAI-style** endpoint:

```
api_url: https://your-host/openai/v1/chat/completions
auth_scheme: NONE | BEARER_TOKEN | API_KEY
```

Set `stream: false` unless you implement SSE.

**Input**: `{ "messages": [ {"role":"user","content":"..."} ] }` → read `choices[0].message.content`.

---

## watsonx.ai (Studio) — Agent example

Make a watsonx.ai agent call Universal A2A using the **OpenAI-compatible** route.

**Request:**

```json
POST /openai/v1/chat/completions
Content-Type: application/json
Authorization: Bearer <YOUR_TOKEN_IF_ANY>
{
  "model": "universal-a2a-hello",
  "messages": [ { "role": "user", "content": "${user_input}" } ]
}
```

**Response:**

```json
{
  "choices": [ { "message": { "role": "assistant", "content": "Hello, you said: ..." } } ]
}
```

---

## MatrixHub integration

**Ingest** an index:

```bash
export HUB_BASE=${HUB_BASE:-http://localhost:443}
curl -s -X POST "$HUB_BASE/catalog/ingest" \
  -H 'Content-Type: application/json' \
  -d '{ "index_url": "https://raw.githubusercontent.com/your/repo/main/catalog/index.json" }' | jq
```

**Discover**:

```bash
curl "$HUB_BASE/catalog/search?q=universal%20a2a&type=agent&mode=keyword&limit=5" | jq
```

**Install** into a project:

```bash
curl -s -X POST "$HUB_BASE/catalog/install" \
  -H 'Content-Type: application/json' \
  -d '{ "id": "agent:universal-a2a-hello@1.2.0", "target": "/tmp/myapp" }' | jq
```

Optional environment for Hub process:

```env
MCP_GATEWAY_URL=http://localhost:4444
MCP_GATEWAY_TOKEN=Bearer <your-token>
DERIVE_TOOLS_FROM_MCP=true
```

---

## Security & ops notes

* Use TLS in production and set `PUBLIC_URL` accordingly.
* Put the **OpenAI endpoint** behind auth if exposed publicly.
* Rate-limit / IP allow-lists at the ingress if needed.
* Rotate tokens and restrict scopes in your orchestrators.
* Health: `GET /healthz` (liveness), `GET /readyz` (readiness).
* Observability: front with nginx/envoy; add OpenTelemetry to FastAPI if you need traces.

**Troubleshooting**

* **401 Unauthorized**: check `PRIVATE_ADAPTER_*` or your Bearer/API key headers.
* **415/400 on /openai**: ensure `Content-Type: application/json` and valid body.
* **Empty replies**: send a `messages` array (OpenAI route) or a `text` part (A2A/JSON-RPC).
* **CORS**: if calling from browsers, set `CORS_*` envs or call via your backend.

---

## Versioning

* Semantic versioning (`MAJOR.MINOR.PATCH`).
* The Agent Card `version` tracks runtime features; adapters aim to be backward-compatible within a minor series.

---

## Contributing

1. Fork the repo and create a feature branch.
2. Add tests (see `tests/`).
3. Run `pytest -q` locally, ensure CI passes.
4. Open a PR with a clear description and rationale.

---

## License

```
Apache License, Version 2.0

Copyright (c) 2025 ruslanmv.com
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

See the [LICENSE](LICENSE) file for full terms.

---

## 📚 Learn More

For more information, detailed documentation, and additional tutorials, please check the repository:
👉 [universal-a2a-agent-tutorial](https://github.com/ruslanmv/universal-a2a-agent-tutorial)

---

💡 *Universal A2A Agent makes your agents portable, interoperable, and production-ready.*
Build once, run anywhere. 🚀
