from a2a_universal import mount, run


async def handle_text(text: str) -> str:
    return f"Hello from my agent. You said: {text}"


app = mount(
    handler=handle_text, name="Tiny Agent", description="One function → full A2A"
)

# App at '/', Universal A2A under '/a2a'
run("__main__:app", mode="attach", a2a_prefix="/a2a", host="0.0.0.0", port=8080)
