# src/a2a_universal/__main__.py
from __future__ import annotations

"""
Universal A2A Agent — CLI Runner

Run the universal A2A service standalone or composed with your existing ASGI app.

Examples
--------
# 1) Run Universal A2A alone (echo provider by default)
python -m a2a_universal --host 0.0.0.0 --port 8000

# 2) Attach A2A under your existing FastAPI app at /a2a
python -m a2a_universal --app mypkg.web:app --mode attach --a2a-prefix /a2a

# 3) Make A2A primary at '/' and mount your app at /app
python -m a2a_universal --app mypkg.web:app --mode primary --user-prefix /app

Notes
-----
- Settings are read from environment (and .env when present; see runner.py).
- Providers and frameworks are selected via:
    LLM_PROVIDER=echo|watsonx|openai|anthropic|gemini|azure|ollama|bedrock|...
    AGENT_FRAMEWORK=langgraph|crewai|langchain|native|...
"""

import argparse
import sys

from .runner import run
from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="a2a_universal",
        description="Run Universal A2A as a server (solo or composed with your ASGI app).",
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit.",
    )
    p.add_argument(
        "--app",
        default=None,
        help="ASGI app path in 'module:attr' form (e.g., 'mypkg.web:app'). Optional.",
    )
    p.add_argument(
        "--mode",
        choices=["primary", "attach", "solo"],
        default="primary",
        help=(
            "Composition mode:\n"
            " - primary: A2A at '/', your app under --user-prefix (default: /app)\n"
            " - attach : Your app at '/', A2A under --a2a-prefix (default: /a2a-universal)\n"
            " - solo   : Only A2A (ignore --app)"
        ),
    )
    p.add_argument(
        "--a2a-prefix",
        default="/a2a-universal",
        help="Mount path for A2A when --mode=attach (default: /a2a-universal).",
    )
    p.add_argument(
        "--user-prefix",
        default="/app",
        help="Mount path for your app when --mode=primary (default: /app).",
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host (default: 0.0.0.0).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000).",
    )
    p.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (dev only). Mutually exclusive with --workers.",
    )
    p.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (omit for single-process). Incompatible with --reload.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit(0)

    # Basic CLI validation
    if args.reload and args.workers:
        parser.error("--reload and --workers are mutually exclusive. Drop one of them.")

    # If mode=solo, ignore any --app (but allow it to be supplied)
    app_arg = None if args.mode == "solo" else args.app

    try:
        run(
            app=app_arg,
            mode=args.mode,
            a2a_prefix=args.a2a_prefix,
            user_prefix=args.user_prefix,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            workers=args.workers,
        )
    except KeyboardInterrupt:
        # Graceful termination on Ctrl+C
        print("\nShutting down...", file=sys.stderr)
        sys.exit(130)
    except Exception as e:  # pragma: no cover
        # User-friendly error and non-zero exit
        print(f"[a2a_universal] Fatal error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
