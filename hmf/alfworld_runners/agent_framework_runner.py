"""Agent-Framework ALFWorld runner — uses OpenAI chat completions API.

The MS Agent Framework's native client uses the Responses API which
LiteLLM doesn't proxy. We use the standard OpenAI chat completions
client instead, matching the lobster runner approach.
"""

from __future__ import annotations
from typing import Tuple
from openai import OpenAI
from .base import ALFWorldRunner


class AgentFrameworkALFWorld(ALFWorldRunner):
    framework_name = "agent_framework"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self._use_new_param = any(
            t in self.model for t in ("gpt-5", "o3", "o4")
        )

    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, int]:
        params = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if self._use_new_param:
            params["max_completion_tokens"] = 256
        else:
            params["max_tokens"] = 256
            params["temperature"] = 0.1

        resp = self._client.chat.completions.create(**params)
        text = resp.choices[0].message.content or ""
        pt = resp.usage.prompt_tokens if resp.usage else 0
        ct = resp.usage.completion_tokens if resp.usage else 0
        return text, pt, ct
