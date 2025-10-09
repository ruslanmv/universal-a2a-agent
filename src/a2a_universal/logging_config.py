# SPDX-License-Identifier: Apache-2.0
"""
Configures structured logging for the application.

This module sets up logging to be environment-aware:
- In 'development' (default), it uses rich, colorful console output
  for readability.
- In 'production', it switches to a JSON renderer for structured,
  machine-readable logs suitable for log aggregation systems.
"""
import logging
import os
import sys

import structlog
from rich.logging import RichHandler


def configure_logging():
    """
    Configure structured, environment-aware logging for the application.
    """
    # Determine the environment to select the appropriate log renderer.
    # Defaults to 'development' if not set.
    app_env = os.getenv("APP_ENV", "development").lower()

    # Define the chain of processors that will enrich the log message.
    # These are used in both development and production.
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        # Add a timestamp in ISO format.
        structlog.processors.TimeStamper(fmt="iso"),
        # Add callsite information for easier debugging.
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
    ]

    # --- Environment-Specific Configuration ---
    if app_env == "production":
        # In production, use a JSON renderer. This output is structured and
        # ideal for log collectors like Fluentd, Logstash, or cloud services.
        final_processors = shared_processors + [
            # Un-comment the next line to include the event dictionary in the log
            # structlog.stdlib.render_to_log_kwargs,
            structlog.processors.JSONRenderer(),
        ]
        handler = logging.StreamHandler(sys.stdout)
        # In production, we don't need a custom formatter on the handler,
        # as the JSONRenderer processor handles the entire output format.
        formatter = logging.Formatter("%(message)s")

    else:
        # In development, use a colorful console renderer for readability.
        final_processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=False),
        ]
        # RichHandler makes logs from other libraries (like uvicorn) beautiful.
        handler = RichHandler(rich_tracebacks=True, markup=False)
        formatter = logging.Formatter(fmt="%(message)s", datefmt="[%X]")

    # --- Core structlog and stdlib logging configuration ---
    structlog.configure(
        processors=final_processors,
        # Use a wrapper class for compatibility with the standard library.
        wrapper_class=structlog.stdlib.BoundLogger,
        # Use a logger factory that's aware of the standard library.
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Enable caching for better performance.
        cache_logger_on_first_use=True,
    )

    handler.setFormatter(formatter)

    # Get the root logger and clear any existing handlers to prevent duplicates.
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # --- THE FIX: Intercept and Redirect Uvicorn Logs ---
    # This is the crucial step to prevent garbled, duplicated log output.
    # It removes Uvicorn's default handlers and tells its loggers to pass
    # messages up to the root logger, which we have configured.
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
    # -------------------------------------------------------------

    # Suppress overly verbose loggers from dependencies if needed.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ibm_watsonx_ai").setLevel(logging.WARNING)

    # Log the current logging mode for clarity on startup.
    structlog.get_logger("a2a.logging").info(
        "Logging configured", mode=app_env
    )
