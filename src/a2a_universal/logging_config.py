import logging
import sys
import structlog
from rich.logging import RichHandler

def configure_logging():
    """
    Configure structured, colorful logging for the application.

    This setup uses `structlog` to process all log records and `rich` to
    render them beautifully in the console, perfect for development.
    """
    # Define the chain of processors that will format the log message.
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    structlog.configure(
        processors=shared_processors + [
            # This processor is for development, making logs colorful and readable.
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        # Use a wrapper class for compatibility with the standard library.
        wrapper_class=structlog.stdlib.BoundLogger,
        # Use a logger factory that's aware of the standard library.
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Enable caching for better performance.
        cache_logger_on_first_use=True,
    )

    # We use RichHandler to make sure logs from other libraries (like uvicorn, httpx)
    # are also beautifully formatted.
    handler = RichHandler(rich_tracebacks=True, markup=True)

    # Set a custom format for the handler.
    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]",
    )
    handler.setFormatter(formatter)

    # --- THE FIX: Disable Uvicorn's default loggers to prevent conflicts ---
    # This ensures that our RichHandler is the ONLY one formatting these logs.
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    # -----------------------------------------------------------------------
    
    # Get the root logger and configure it.
    root_logger = logging.getLogger()
    # Clear any existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Suppress overly verbose loggers from dependencies if needed.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ibm_watsonx_ai").setLevel(logging.WARNING)