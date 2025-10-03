import os
from a2a_universal.providers import build_provider

# Load environment variables from the .env file in the same directory as this script
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / '.env')

def main():
    os.environ.setdefault("LLM_PROVIDER", "watsonx")
    p = build_provider()
    print("ready:", p.ready, "reason:", getattr(p, "reason", ""))
    print("model reply:", p.generate(prompt="Tell me about Genova Italy"))

if __name__ == "__main__":
    main()