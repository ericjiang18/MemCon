"""
Lobster + HMF  —  Single-agent baseline augmented with hierarchical
memory and MPC-controlled memory access.

Memory context (cached results, past experiences, matching skills) is
injected into the system prompt.  After each task the experience is
stored and skills are evolved.
"""

from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BASELINE = os.path.join(_REPO_ROOT, "agent_baseline")
for _p in (_REPO_ROOT, _BASELINE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runners.base_runner import BaseRunner, GenerateResult
from ..agent.hmf_agent import HMFAgent
from ..config import HMFConfig


class LobsterHMFRunner(BaseRunner):
    framework_name = "lobster_hmf"

    def __init__(self, hmf_config: HMFConfig | None = None, **kwargs):
        super().__init__(**kwargs)

        cfg = hmf_config or HMFConfig(
            model_name=self.model,
            api_base=self.base_url,
            api_key=self.api_key,
        )
        self.hmf = HMFAgent(cfg)
        self._task_idx = 0

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerateResult:
        from openai import AsyncOpenAI

        hmf = self.hmf
        self._task_idx += 1
        task_type = self.domain or "general"

        context = hmf.init_task(user_prompt[:200], task_type, user_prompt)
        mem_ctx = hmf.build_memory_context(context)

        augmented_system = system_prompt
        if mem_ctx:
            augmented_system += f"\n\n{mem_ctx}"

        # MPC may also suggest pre-step memory actions
        mpc_action = hmf.step_decision(user_prompt[:500])
        if mpc_action.action_type.name not in ("NO_OP", "LLM_GENERATE"):
            extra_result = hmf.execute_memory_action(mpc_action.action_type, user_prompt[:300])
            if extra_result.get("data"):
                augmented_system += "\n\n=== Additional Memory Context ===\n"
                for d in extra_result["data"][:2]:
                    augmented_system += d[:400] + "\n"

        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        t0 = time.time()
        resp = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
        )
        latency_ms = (time.time() - t0) * 1000

        text = resp.choices[0].message.content or ""
        pt = ct = tt = 0
        if resp.usage:
            pt = resp.usage.prompt_tokens or 0
            ct = resp.usage.completion_tokens or 0
            tt = resp.usage.total_tokens or (pt + ct)

        hmf.record_step(
            action="generate",
            observation=text[:500],
            token_cost=pt + ct,
            latency_ms=latency_ms,
        )

        hmf.finish_task(
            success=True,
            feedback=text[:300],
        )

        return GenerateResult(
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
        )
