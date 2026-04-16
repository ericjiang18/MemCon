"""Lobster ALFWorld runner — single-agent OpenAI direct call."""

from __future__ import annotations
from typing import Tuple
from .base import ALFWorldRunner


class LobsterALFWorld(ALFWorldRunner):
    framework_name = "lobster"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self._use_new_param = any(t in self.model for t in ("gpt-5", "o3", "o4"))

    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, int]:
        params = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        if self._use_new_param:
            params["max_completion_tokens"] = 256
        else:
            params["max_tokens"] = 256
        resp = self._client.chat.completions.create(**params)
        text = resp.choices[0].message.content or ""
        pt = ct = 0
        if resp.usage:
            pt = resp.usage.prompt_tokens or 0
            ct = resp.usage.completion_tokens or 0
        return text, pt, ct
