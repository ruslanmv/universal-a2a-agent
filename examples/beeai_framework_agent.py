import asyncio
from beeai_framework.backend import UserMessage
from a2a_universal.adapters.beeai_agent import make_beeai_agent

async def main():
    agent = make_beeai_agent("http://localhost:8000")
    result = await agent.run(UserMessage("ping from BeeAI"))
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
