from __future__ import annotations
import os
import typer
from .client import A2AClient
from .card import agent_card

app = typer.Typer(add_completion=False, help="A2A CLI")

def _base() -> str:
    return os.getenv("PUBLIC_URL", "http://localhost:8000")

@app.command()
def ping(text: str = "Hello from CLI", jsonrpc: bool = False):
    client = A2AClient(_base())
    reply = client.send(text=text, use_jsonrpc=jsonrpc)
    typer.echo(reply)

@app.command()
def card():
    import json
    typer.echo(json.dumps(agent_card(), indent=2))

if __name__ == "__main__":
    app()
