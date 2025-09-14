from __future__ import annotations
from typing import Any
from ..frameworks import FrameworkBase, _call_provider, _extract_last_user_text

class Framework(FrameworkBase):
    id = "native"
    name = "Native PassThrough"

    async def execute(self, messages: list[dict[str, Any]]) -> str:
        text = _extract_last_user_text(messages)
        return await _call_provider(self.provider, text, messages)
