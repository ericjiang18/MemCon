"""Agent-Framework ALFWorld runner — uses MS Agent Framework's OpenAI client.

ORIGINAL VERSION: Uses the Responses API via agent_framework.openai.
This does NOT work with LiteLLM proxy (which only supports chat/completions).
Kept for reference / direct OpenAI API usage.
"""

from __future__ import annotations
import asyncio
from typing import Tuple
from .base import ALFWorldRunner


class AgentFrameworkALFWorld(ALFWorldRunner):
    framework_name = "agent_framework"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from agent_framework.openai import OpenAIChatClient
        self._client = OpenAIChatClient(
            model=self.model, api_key=self.api_key, base_url=self.api_base,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, int]:
        agent = self._client.as_agent(
            name="alfworld_agent",
            instructions=system_prompt,
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(agent.run(user_prompt))
        pt = ct = 0
        ud = getattr(result, "usage_details", None)
        if isinstance(ud, dict):
            pt = ud.get("input_token_count", 0) or 0
            ct = ud.get("output_token_count", 0) or 0
        text = getattr(result, "text", "") or str(result.value) if hasattr(result, "value") else str(result)
        return text, pt, ct
