# SPDX-License-Identifier: Apache-2.0
"""
BeeAI Framework plugin.

Intent:
- If `beeai_framework` is installed, we can optionally wire a minimal BeeAI-based
  flow here in the future. For now, we keep a pragmatic implementation that
  delegates to the active Provider for generation (identical to 'native'), so
  selecting `AGENT_FRAMEWORK=beeai` always works and remains future-proof.

Behavior:
- If more advanced BeeAI graphing/routing is desired, extend this plugin to
  construct a BeeAI agent that uses the Provider via a tool callback.
"""

from __future__ import annotations
from typing import Any

from ..frameworks import FrameworkBase, _call_provider, _extract_last_user_text


class Framework(FrameworkBase):
    id = "beeai"
    name = "BeeAI Framework"

    def __init__(self, provider, **kwargs):
        # Keep it simple and robust: ready even without beeai_framework installed.
        super().__init__(provider)
        self.ready = True
        self.reason = ""

    async def execute(self, messages: list[dict[str, Any]]) -> str:
        text = _extract_last_user_text(messages)
        return await _call_provider(self.provider, text, messages)
