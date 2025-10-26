import a2a_universal as a2a
from fastapi import FastAPI

app = FastAPI()


@app.get("/hello")
def hello():
    return {"hi": "from my app under /app"}


async def handle_text(text: str) -> str:
    return f"Hello from my custom agent. You said: {text}"


if __name__ == "__main__":
    a2a.run(
        "__main__:app",
        mode="primary",
        user_prefix="/app",
        handler=handle_text,
        name="Tiny Agent",
        description="One function → full A2A",
        host="0.0.0.0",
        port=8080,
    )
