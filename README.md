<!-- =======================================================================
  Universal A2A Agent (Python) — README
  Professional edition with badges, logos, and end-to-end instructions
  License: Apache-2.0
======================================================================= -->

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

# Universal A2A Agent (Python)

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

## What’s new in this upgrade (🚀 Production-Ready)

**Pluggable Provider *and* Framework layers** with runtime **injection**, **async execution**, **structured JSON logs**, and **typed Pydantic settings**.

* ✅ **Provider plugins** (OpenAI, watsonx.ai, Anthropic, Gemini, Azure OpenAI, Bedrock, Ollama, Echo, …)
* ✅ **Framework plugins** (Native pass-through, LangGraph, CrewAI) — choose at deploy time
* ✅ **Injection:** `AGENT_FRAMEWORK` receives the built `LLM_PROVIDER`
* ✅ **Async-safe:** sync providers run via an async shim (no event-loop blocking)
* ✅ **/readyz** reflects **both** provider and framework readiness
* ✅ **OpenAI-compatible** endpoint for UIs & orchestrators
* ✅ **Private adapter** (enterprise) with pluggable auth (NONE/BEARER/API\_KEY)

---

## Table of contents

* [Project tree](#project-tree)
* [Quick start](#quick-start)
* [Installation options](#installation-options)
* [Environment](#environment-envexample)
* [Endpoints](#endpoints)
* [Agent Card](#agent-card)
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
* [Watsonx Orchestrate](#watsonx-orchestrate)
* [WatsonX.ai (Studio) — Agent example](#watsonxai-studio--agent-example)
* [MatrixHub integration](#matrixhub-integration)
* [Security & ops notes](#security--ops-notes)
* [Versioning](#versioning)
* [Contributing](#contributing)
* [License](#license)

---

## Project tree

```
universal-a2a-agent/
├─ pyproject.toml
├─ README.md                          # (this document)
├─ LICENSE
├─ .env.example
├─ Dockerfile
├─ run.sh
├─ src/
│  └─ a2a_universal/
│     ├─ __init__.py
│     ├─ config.py                    # NEW: Pydantic settings (env-driven)
│     ├─ logging_config.py            # NEW: Structured JSON logging
│     ├─ server.py                    # FastAPI app: /a2a, /rpc, /openai, private adapter
│     ├─ models.py                    # Pydantic data models (A2A, JSON-RPC)
│     ├─ card.py                      # Agent Card at /.well-known/agent-card.json
│     ├─ client.py                    # Thin HTTP client
│     ├─ cli.py                       # a2a CLI (ping, card)
│     ├─ providers.py                 # Provider registry & factory (auto-discovery)
│     ├─ provider_plugins/            # Built-in providers (echo, openai, watsonx, ...)
│     │  ├─ __init__.py
│     │  ├─ echo.py
│     │  ├─ openai.py
│     │  ├─ watsonx.py
│     │  ├─ ollama.py
│     │  ├─ anthropic.py
│     │  ├─ gemini.py
│     │  ├─ azure_openai.py
│     │  └─ bedrock.py
│     ├─ frameworks.py                # NEW: Framework registry & factory (auto-discovery)
│     ├─ framework_plugins/           # NEW: Built-in frameworks (native, langgraph, crewai)
│     │  ├─ __init__.py
│     │  ├─ native.py
│     │  ├─ langgraph.py
│     │  └─ crewai.py
│     └─ adapters/
│        ├─ langchain_tool.py         # (optional) LangChain tool wrapper
│        ├─ langgraph_node.py         # (legacy dict-state) node
│        ├─ langgraph_agent.py        # (native MessagesState) agent node
│        ├─ crewai_tool.py            # CrewAI function tool
│        ├─ crewai_base_tool.py       # CrewAI BaseTool class
│        ├─ bee_tool.py               # Bee callable
│        ├─ beeai_agent.py            # BeeAI Framework agent wrapper
│        ├─ autogen_tool.py           # AutoGen registered function
│        └─ private_adapter.py        # Private enterprise adapter mapping helpers
├─ examples/
│  ├─ call_direct.py
│  ├─ langchain_example.py
│  ├─ langgraph_example.py            # legacy dict-state
│  ├─ langgraph_agent_example.py      # native MessagesState
│  ├─ crewai_example.py
│  ├─ crewai_deep_example.py
│  ├─ bee_example.py
│  ├─ beeai_framework_agent.py
│  ├─ autogen_example.py
│  ├─ quickstart_langchain_watsonx.py # NEW: watsonx-backed LC example
│  ├─ quickstart_langgraph_watsonx.py # NEW: watsonx-backed LangGraph example
│  └─ quickstart_crewai_watsonx.py    # NEW: watsonx-backed CrewAI example
├─ deploy/
│  └─ helm/
│     └─ universal-a2a-agent/
│        ├─ Chart.yaml
│        ├─ values.yaml
│        └─ templates/
│           ├─ deployment.yaml
│           ├─ service.yaml
│           ├─ ingress.yaml
│           ├─ configmap.yaml
│           └─ secret.yaml
└─ tests/
   ├─ test_server.py
   └─ test_langgraph_agent.py
```

> **Why this layout works**
>
> * **Single runtime**: one FastAPI app exposes multiple integration-friendly endpoints.
> * **Plugins**: providers & frameworks are discovered at runtime (or via entry points).
> * **Deployability**: Dockerfile + Helm chart = drop-in for most clusters.

---

## Quick start

```bash
# 1) clone
git clone https://github.com/ruslanmv/universal-a2a-agent.git
cd universal-a2a-agent

# 2) (optional) venv
python -m venv .venv && source .venv/bin/activate

# 3) install core
pip install -e .
# (optional) everything
pip install -e .[all]

# 4) pick a Provider + Framework (any combo)
export LLM_PROVIDER=echo             # echo|openai|watsonx|ollama|anthropic|gemini|azure_openai|bedrock
export AGENT_FRAMEWORK=native        # native|langgraph|crewai

# 5) run the server
uvicorn a2a_universal.server:app --host 0.0.0.0 --port 8000 --reload

# 6) smoke test (A2A)
curl -s http://localhost:8000/a2a -H 'Content-Type: application/json' -d '{
  "method":"message/send",
  "params":{"message":{"role":"user","messageId":"m1","parts":[{"type":"text","text":"ping"}]}}
}' | jq

# readiness (shows provider+framework)
curl -s http://localhost:8000/readyz | jq
```

> **Fast path for humans**
>
> If you can hit the **smoke test**, you can integrate **any** adapter below. All adapters call into the same `/a2a`/`/rpc` logic.

---

## Installation options

```bash
# Core (FastAPI server + CLI + client)
pip install -e .

# Optional provider/framework extras
pip install -e .[openai]
pip install -e .[watsonx]
pip install -e .[langgraph]
pip install -e .[crewai]
# Everything
pip install -e .[all]
```

> **Tip**: Start with `.[langgraph]` or `.[langchain]` for prototyping, add the rest later.

---

## Environment (.env.example)

```env
# Core server
HOST=0.0.0.0
PORT=8000
PUBLIC_URL=http://localhost:8000

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

# Private adapter (enterprise)
PRIVATE_ADAPTER_ENABLED=false        # true|false
PRIVATE_ADAPTER_AUTH_SCHEME=NONE     # NONE|BEARER|API_KEY
PRIVATE_ADAPTER_AUTH_TOKEN=
PRIVATE_ADAPTER_PATH=/enterprise/v1/agent
```

> **Production note**: Set `PUBLIC_URL` to your public **HTTPS** origin so your Agent Card advertises the correct `/rpc` endpoint.

---

## Endpoints

* `POST /a2a` — **Raw A2A** envelope

  ```json
  {
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "messageId": "m1",
        "parts": [{"type": "text", "text": "hello"}]
      }
    }
  }
  ```

* `POST /rpc` — **JSON-RPC 2.0**

  ```json
  {
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": { "message": { "...": "..." } }
  }
  ```

* `POST /openai/v1/chat/completions` — **OpenAI-style** chat for UIs/orchestrators
  Minimal body:

  ```json
  { "model":"universal-a2a-hello", "messages":[{"role":"user","content":"hello"}] }
  ```

* `GET /.well-known/agent-card.json` — **Agent Card** discovery

* `GET /healthz` — liveness probe

* `GET /readyz` — readiness for **provider & framework**

> **Which one should I use?**
>
> * **LangGraph/LangChain/CrewAI/AutoGen**: prefer `/rpc` **or** `/a2a` via the provided adapters.
> * **Orchestrators & chat UIs**: `/openai/v1/chat/completions` is the universal bridge.
> * **Enterprise gateway**: use the private adapter (see env flags) if you need field/key remapping.

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

> **Why it matters**: Many gateways (MCP, BeeAI, etc.) can auto-discover your agent via this well-known card.

---

## Providers & Frameworks (runtime selection)

Pick any **Provider** and any **Framework** at deploy time:

```bash
# Examples
export LLM_PROVIDER=openai           # or watsonx|ollama|anthropic|gemini|azure_openai|bedrock|echo
export AGENT_FRAMEWORK=langgraph     # or native|crewai
```

Under the hood:

* `providers.py` builds the selected Provider plugin
* `frameworks.py` builds the selected Framework plugin and **injects** the Provider
* The server routes requests to `await framework.execute(messages)` (async); sync providers are offloaded to a worker thread
* `/readyz` shows both `provider_*` and `framework_*` readiness and reasons

> Third-parties can ship providers/frameworks via setuptools **entry points**:
>
> * Providers: `a2a_universal.providers`
> * Frameworks: `a2a_universal.frameworks`

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

llm = ChatOpenAI(model="gpt-4o-mini")   # replace with your preferred LLM
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
        return {"messages": [AIMessage(content=reply)]}  # partial update
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

> **Why “native”**: returns a **partial** update on `messages` so LangGraph’s reducer appends it cleanly.

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

Function tool:

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

BaseTool class:

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
    return BeeA2AAgent(agent_card_url=f"{base_url}/.well-known/agent-card.json", memory=UnconstrainedMemory())
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

> Run the server with watsonx.ai:
>
> ```bash
> pip install -e .[watsonx]
> export LLM_PROVIDER=watsonx
> export WATSONX_API_KEY=YOUR_KEY
> export WATSONX_URL=https://us-south.ml.cloud.ibm.com
> export WATSONX_PROJECT_ID=YOUR_PROJECT_ID
> export MODEL_ID=ibm/granite-3-3-8b-instruct
> export AGENT_FRAMEWORK=native
> uvicorn a2a_universal.server:app --port 8000
> curl -s localhost:8000/readyz | jq
> ```

**1) LangChain — Tool that calls A2A (watsonx-backed)**

```python
# examples/quickstart_langchain_watsonx.py
import httpx
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI  # any LC LLM is fine as the "driver"

BASE = "http://localhost:8000"

def a2a_call(prompt: str) -> str:
    payload = {
        "method": "message/send",
        "params": {"message": {"role": "user", "messageId": "lc-tool", "parts": [{"type": "text", "text": prompt}]}}
    }
    r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    for p in (data.get("message") or {}).get("parts", []):
        if p.get("type") == "text":
            return p.get("text", "")
    return ""

tool = Tool(name="a2a_hello", description="Call Universal A2A (watsonx-backed)", func=a2a_call)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = initialize_agent([tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

if __name__ == "__main__":
    print(agent.run("Use the a2a_hello tool to say hello to LangChain."))
```

**2) LangGraph — Node that posts to A2A (watsonx-backed)**

```python
# examples/quickstart_langgraph_watsonx.py
import asyncio
import httpx
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

BASE = "http://localhost:8000"

async def a2a_send(text: str) -> str:
    payload = {
        "method": "message/send",
        "params": {"message": {"role": "user", "messageId": "lg-node", "parts": [{"type": "text", "text": text}]}}
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{BASE}/a2a", json=payload)
        r.raise_for_status()
        data = r.json()
        for p in (data.get("message") or {}).get("parts", []):
            if p.get("type") == "text":
                return p.get("text", "")
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

**3) CrewAI — Crew tool that calls A2A (watsonx-backed)**

```python
# examples/quickstart_crewai_watsonx.py
import httpx
from crewai import Agent, Task, Crew

BASE = "http://localhost:8000"

def a2a_call(prompt: str) -> str:
    payload = {
        "method": "message/send",
        "params": {"message": {"role": "user", "messageId": "crewai-tool", "parts": [{"type": "text", "text": prompt}]}}
    }
    r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    for p in (data.get("message") or {}).get("parts", []):
        if p.get("type") == "text":
            return p.get("text", "")
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

GitHub Actions (sample): `.github/workflows/ci.yml`

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

> **Pro tip**: In CI, prefer `--fail-under` thresholds on coverage to keep adapters honest.

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
> * Readiness vs liveness probes: start with `/healthz` for both, then split as needed.
> * Expose only the paths you need at your ingress.
> * Put `/openai/v1/chat/completions` behind auth if it’s Internet-facing.

---

## MCP Gateway

Two common paths:

1. **A2A Registry** (if available): register your agent’s card URL and endpoint.
2. **REST Tool**: post to `/rpc` and map `result.message.parts[].text` to MCP text outputs (e.g., via `jq`).

> **Tip**: If your Gateway supports Agent Cards, point it at `/.well-known/agent-card.json` and you’re done.

---

## Watsonx Orchestrate

Use the provided **OpenAI-style** endpoint:

```
api_url: https://your-host/openai/v1/chat/completions
auth_scheme: NONE | BEARER_TOKEN | API_KEY
```

Set `stream: false` (or implement SSE for streaming).

> **Mapping guide**
>
> * **Input**: `{ "messages": [ {"role":"user","content":"..."} ] }`
> * **Output**: read `choices[0].message.content` as the agent reply.

---

## WatsonX.ai (Studio) — Agent example

> This section shows how to make a **watsonx.ai agent** call the Universal A2A Agent using the **OpenAI-compatible** endpoint. You can adapt the same idea in notebooks, prompt lab, or custom tools.

### 1) Prepare your A2A service

* Deploy the service and ensure it’s reachable at `https://your-host`.
* Decide on auth: `NONE`, `BEARER_TOKEN`, or `API_KEY` and configure it on both sides.

### 2) Create a callable tool in watsonx.ai

You can model a simple HTTP tool that sends a user message to your A2A and returns the reply. Here’s the **request/response** shape:

**Request (to your agent):**

```json
POST /openai/v1/chat/completions
Content-Type: application/json
Authorization: Bearer <YOUR_TOKEN_IF_ANY>
{
  "model": "universal-a2a-hello",
  "messages": [ { "role": "user", "content": "${user_input}" } ]
}
```

**Response (from your agent):**

```json
{
  "choices": [ { "message": { "role": "assistant", "content": "Hello, you said: ..." } } ]
}
```

### 3) Minimal Python snippet (works in watsonx.ai notebooks & tools)

```python
import os, requests

BASE = os.getenv("A2A_BASE", "https://your-host")
TOKEN = os.getenv("A2A_TOKEN", "")  # optional

headers = {"Content-Type": "application/json"}
if TOKEN:
    headers["Authorization"] = f"Bearer {TOKEN}"

body = {
    "model": "universal-a2a-hello",
    "messages": [{"role": "user", "content": "ping from watsonx.ai"}]
}

r = requests.post(f"{BASE}/openai/v1/chat/completions", json=body, headers=headers, timeout=20)
r.raise_for_status()
print(r.json()["choices"][0]["message"]["content"])
```

### 4) Make it an Agent behavior

* **Intent**: “Greet users and echo their message.”
* **Tool contract**: one string input (`user_input`), one string output (`agent_reply`).
* **Invocation**: The agent passes `user_input` to the HTTP tool; map the HTTP response to `agent_reply = choices[0].message.content`.

> **Why this works**: watsonx.ai agents can call HTTP tools. Since A2A exposes an **OpenAI-compatible** route, the integration is trivial—no SDK lock‑in.

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
* Put the **OpenAI endpoint** behind auth if it’s exposed publicly.
* Rate-limit and add IP allow-lists at the ingress if needed.
* Rotate tokens and restrict scopes in your orchestrators.
* Health checks: `GET /healthz`. Add readiness probes as your rollout requires.
* Observability: front with nginx/envoy for access logs; add OpenTelemetry to FastAPI if you need traces.

> **Troubleshooting**
>
> * **401 Unauthorized**: check `PRIVATE_ADAPTER_*` or your Bearer/API key headers.
> * **415/400 on /openai**: ensure `Content-Type: application/json` and valid body.
> * **Empty replies**: ensure you’re sending a `messages` array (OpenAI route) or a `text` part (A2A/JSON-RPC).
> * **CORS**: if calling from browsers, set `CORS_*` envs or call via your backend.

---

## Versioning

* Semantic versioning (`MAJOR.MINOR.PATCH`).
* The Agent Card `version` tracks runtime features; adapters are backward-compatible within a minor series.

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

Copyright (c) 2025 YOUR
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

See the [LICENSE](LICENSE) file for full terms.
