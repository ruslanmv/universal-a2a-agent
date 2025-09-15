from __future__ import annotations

import importlib.metadata

try:
    # Dynamically pull version from installed package metadata
    __version__ = importlib.metadata.version("universal-a2a-agent")
except importlib.metadata.PackageNotFoundError:
    # Fallback when running in dev mode (editable install)
    __version__ = "0.0.0.dev0"

from .client import A2AClient

__all__ = ["A2AClient", "__version__"]
