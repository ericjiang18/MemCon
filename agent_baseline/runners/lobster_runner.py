"""Runner for Lobster — single-agent baseline.

Serves as the single-turn baseline for comparison against multi-agent
frameworks (AutoGen group chat, LangGraph graph workflow, Agent Framework
pipeline) and agent_q_mix.

Uses a lightweight Node.js worker or falls back to the Python openai library.
"""

import json
import os
import subprocess
from .base_runner import BaseRunner, GenerateResult

_WORKER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lobster", "bench_worker.mjs",
)


def _node_available() -> bool:
    try:
        subprocess.run(
            ["node", "--version"], capture_output=True, check=True, timeout=10
        )
        return True
    except Exception:
        return False


class LobsterRunner(BaseRunner):
    framework_name = "lobster"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._use_node = False  # Always use Python openai client for reliability
        if self._use_node:
            print(f"  [lobster] Using Node.js worker: {_WORKER_PATH}")
        else:
            print("  [lobster] Node.js worker unavailable – falling back to openai Python client")

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerateResult:
        if self._use_node:
            return self._generate_node(system_prompt, user_prompt)
        return await self._generate_openai(system_prompt, user_prompt)

    def _generate_node(self, system_prompt: str, user_prompt: str) -> GenerateResult:
        payload = json.dumps(
            {
                "base_url": self.base_url,
                "api_key": self.api_key,
                "model": self.model,
                "system_prompt": system_prompt,
                "question": user_prompt,
            }
        )
        proc = subprocess.run(
            ["node", _WORKER_PATH],
            input=payload,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Node worker error: {proc.stderr[:500]}")
        resp = json.loads(proc.stdout)
        if "error" in resp:
            raise RuntimeError(resp["error"])
        usage = resp.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        return GenerateResult(
            text=resp["content"],
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=usage.get("total_tokens", pt + ct),
        )

    async def _generate_openai(self, system_prompt: str, user_prompt: str) -> GenerateResult:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        resp = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
        )
        text = resp.choices[0].message.content or ""
        pt = ct = tt = 0
        if resp.usage:
            pt = resp.usage.prompt_tokens or 0
            ct = resp.usage.completion_tokens or 0
            tt = resp.usage.total_tokens or (pt + ct)
        return GenerateResult(
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
        )
