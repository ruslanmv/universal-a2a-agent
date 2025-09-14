#!/usr/bin/env python
"""
Universal A2A Agent — Interactive Setup Wizard (Windows/macOS/Linux)

- Creates/uses a virtualenv
- Installs core and selected extras (LangChain, LangGraph, CrewAI, AutoGen, BeeAI, or ALL)
- Optional dev tools (pytest, ruff, black, mypy)
- Optional .env setup
- Optional quick smoke test of the server

Non-interactive usage (examples):
  python scripts/setup_wizard.py --extras langgraph crewai --devtools --env --smoke
  python scripts/setup_wizard.py --all --venv .venv
"""

from __future__ import annotations
import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV = ROOT / ".venv"

EXTRAS = {
    "langchain": "langchain",
    "langgraph": "langgraph",
    "crewai": "crewai",
    "autogen": "autogen",
    "beeai": "beeai",
    "all": "all",
}

DEV_TOOLS = ["pytest", "ruff", "black", "mypy", "httpx"]


def venv_bin(venv_dir: Path) -> Path:
    """Return the venv bin/Scripts directory in a cross-platform way."""
    if platform.system().lower().startswith("win"):
        return venv_dir / "Scripts"
    return venv_dir / "bin"


def run(cmd: List[str], env=None, check=True) -> subprocess.CompletedProcess:
    print(f" » $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(ROOT), env=env, check=check)


def ensure_venv(venv_dir: Path, py: str) -> None:
    if not venv_dir.exists():
        print(f"Creating virtualenv at {venv_dir} …")
        run([py, "-m", "venv", str(venv_dir)])
    else:
        print(f"Using existing virtualenv at {venv_dir}")


def install_core_and_extras(pip: Path, extras: List[str]) -> None:
    # Build extras string like .[langgraph,crewai] or .[all]
    extras = [e for e in extras if e in EXTRAS]
    if "all" in extras:
        extra_str = ".[all]"
    elif extras:
        extra_str = ".[" + ",".join(extras) + "]"
    else:
        extra_str = "."

    print(f"Installing package in editable mode with extras: {extra_str}")
    run([str(pip), "install", "--upgrade", "pip"])
    run([str(pip), "install", "-e", extra_str])


def install_dev_tools(pip: Path) -> None:
    print("Installing dev tools …")
    run([str(pip), "install", *DEV_TOOLS])


def maybe_copy_env():
    env_file = ROOT / ".env"
    env_example = ROOT / ".env.example"
    if env_file.exists():
        print(".env already exists, leaving it unchanged.")
        return
    if env_example.exists():
        print("Creating .env from .env.example …")
        shutil.copy(env_example, env_file)
    else:
        print("Creating minimal .env …")
        env_file.write_text(
            "A2A_HOST=0.0.0.0\nA2A_PORT=8000\nPUBLIC_URL=http://localhost:8000\n",
            encoding="utf-8",
        )


def smoke_test(bin_dir: Path, host="0.0.0.0", port=8000) -> None:
    uvicorn = bin_dir / "uvicorn"
    print("Starting server for a quick smoke test …")
    proc = subprocess.Popen([str(uvicorn), "a2a_universal.server:app", "--host", host, "--port", str(port)])
    try:
        import urllib.request  # stdlib
        health = f"http://localhost:{port}/healthz"

        # wait briefly for server to come up
        for _ in range(30):
            try:
                with urllib.request.urlopen(health, timeout=0.5) as resp:
                    if resp.status == 200:
                        body = resp.read().decode("utf-8")
                        print(f"Health OK: {body}")
                        break
            except Exception:
                time.sleep(0.2)
        else:
            print("WARNING: server health check did not succeed within timeout.")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print("Smoke test finished.")


def ask_multi_choice(prompt: str, options: List[str]) -> List[str]:
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    print("Enter numbers separated by commas (e.g., 1,3). Enter to skip.")
    selected = input("> ").strip()
    if not selected:
        return []
    result = []
    for token in selected.split(","):
        token = token.strip()
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(options):
                result.append(options[idx - 1])
    return result


def main():
    parser = argparse.ArgumentParser(description="Universal A2A Agent — Setup Wizard")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use for venv creation")
    parser.add_argument("--venv", default=str(DEFAULT_VENV), help="Virtualenv directory (default: .venv)")
    parser.add_argument("--extras", nargs="*", default=[], choices=list(EXTRAS.keys()),
                        help="Extras to install (choose from: %(choices)s)")
    parser.add_argument("--all", action="store_true", help="Install all extras")
    parser.add_argument("--devtools", action="store_true", help="Install dev tools (pytest, ruff, black, mypy, httpx)")
    parser.add_argument("--env", action="store_true", help="Create .env (from .env.example if present)")
    parser.add_argument("--smoke", action="store_true", help="Run quick server smoke test")
    parser.add_argument("--non-interactive", action="store_true", help="Skip prompts and use flags only")
    args = parser.parse_args()

    print("\n=== Universal A2A Agent — Setup Wizard ===")
    print(f"OS: {platform.system()}  Python: {platform.python_version()}  CWD: {ROOT}\n")

    chosen_extras = args.extras[:]
    if args.all and "all" not in chosen_extras:
        chosen_extras = ["all"]

    if not args.non_interactive and not chosen_extras:
        # Interactive selection
        base = ask_multi_choice(
            "Select framework extras to install:",
            ["langchain", "langgraph", "crewai", "autogen", "beeai", "all"]
        )
        chosen_extras = base or []

        if not args.devtools:
            print()
            dev = input("Install dev tools (pytest, ruff, black, mypy)? [y/N] ").strip().lower()
            args.devtools = dev in ("y", "yes", "1", "true")

        if not args.env:
            env_ans = input("Create .env from .env.example (or minimal if missing)? [Y/n] ").strip().lower()
            args.env = env_ans not in ("n", "no", "0", "false")

        if not args.smoke:
            smoke_ans = input("Run a quick smoke test (start/stop server)? [y/N] ").strip().lower()
            args.smoke = smoke_ans in ("y", "yes", "1", "true")

    venv_dir = Path(args.venv).resolve()
    ensure_venv(venv_dir, args.python)
    bin_dir = venv_bin(venv_dir)
    pip = bin_dir / "pip"

    install_core_and_extras(pip, chosen_extras)
    if args.devtools:
        install_dev_tools(pip)
    if args.env:
        maybe_copy_env()
    if args.smoke:
        smoke_test(bin_dir)

    print("\nAll set! Next steps:")
    print("  1) Activate venv:")
    if platform.system().lower().startswith("win"):
        print(f"     {venv_dir}\\Scripts\\activate")
    else:
        print(f"     source {venv_dir}/bin/activate")
    print("  2) Run the server:  make run   (or)  uvicorn a2a_universal.server:app --host 0.0.0.0 --port 8000")
    print("  3) Try a ping:      make ping TEXT='hello world'\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
