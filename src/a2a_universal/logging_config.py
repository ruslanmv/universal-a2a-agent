from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def configure_logging(level: str | int = "INFO") -> logging.Logger:
    lvl = logging.getLevelName(level) if isinstance(level, str) else level
    root = logging.getLogger()
    root.setLevel(lvl)
    # Clear handlers to avoid duplicate logs in reloaders
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    return logging.getLogger("a2a")
