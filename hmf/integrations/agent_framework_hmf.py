"""
Agent-Framework + HMF  —  Multi-agent sequential pipeline augmented with
hierarchical memory and MPC-controlled memory access.

Memory context is injected into each agent's instructions and into the
decision agent.  Post-task, the experience is stored and skills evolved.
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

from runners.base_runner import BaseRunner, GenerateResult, NUM_ROUNDS
from ..agent.hmf_agent import HMFAgent
from ..config import HMFConfig


class AgentFrameworkHMFRunner(BaseRunner):
    framework_name = "agent_framework_hmf"

    def __init__(self, hmf_config: HMFConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        from agent_framework.openai import OpenAIChatClient

        self._client = OpenAIChatClient(
            model_id=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        cfg = hmf_config or HMFConfig(
            model_name=self.model,
            api_base=self.base_url,
            api_key=self.api_key,
        )
        self.hmf = HMFAgent(cfg)
        self._task_idx = 0

    def _extract_usage(self, result) -> tuple:
        pt = ct = tt = 0
        ud = getattr(result, "usage_details", None)
        if isinstance(ud, dict):
            pt = ud.get("input_token_count", 0) or 0
            ct = ud.get("output_token_count", 0) or 0
            tt = ud.get("total_token_count", 0) or (pt + ct)
        return pt, ct, tt

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerateResult:
        role_prompts = self.get_role_prompts()
        hmf = self.hmf

        self._task_idx += 1
        task_type = self.domain or "general"
        context = hmf.init_task(user_prompt[:200], task_type, user_prompt)
        mem_ctx = hmf.build_memory_context(context)

        agents = []
        for role_name, role_desc in role_prompts:
            instructions = (
                f"{role_desc}\n\n"
                "Consider other agents' analysis critically. "
                "Do not simply agree with the majority."
            )
            if mem_ctx:
                instructions += f"\n\n{mem_ctx}"

            agent = self._client.as_agent(
                name=role_name.replace(" ", "_"),
                instructions=instructions,
            )
            agents.append((role_name, agent))

        total_pt = total_ct = 0
        agent_outputs: dict[str, list[str]] = {name: [] for name, _ in agents}

        for round_idx in range(NUM_ROUNDS):
            for role_name, agent in agents:
                prior_text = ""
                for other_name, other_outputs in agent_outputs.items():
                    if other_outputs:
                        prior_text += f"\n--- {other_name} ---\n{other_outputs[-1]}\n"

                prompt = user_prompt
                if prior_text:
                    prompt += f"\n\nOther agents' outputs from previous discussion:\n{prior_text}"

                result = await agent.run(prompt)
                pt, ct, _ = self._extract_usage(result)
                total_pt += pt
                total_ct += ct
                agent_outputs[role_name].append(result.text)

                hmf.record_step(
                    action=f"{role_name}_round{round_idx}",
                    observation=result.text[:300],
                    token_cost=pt + ct,
                )

        decision_instructions = system_prompt
        if mem_ctx:
            decision_instructions += f"\n\n{mem_ctx}"

        decision_agent = self._client.as_agent(
            name="Decision_Maker",
            instructions=decision_instructions,
        )

        synthesis = f"Task:\n{user_prompt}\n\nAgents' responses:\n"
        for role_name, outputs in agent_outputs.items():
            for r_idx, resp in enumerate(outputs):
                synthesis += f"\n--- {role_name} (round {r_idx + 1}) ---\n{resp}\n"
        synthesis += "\nSynthesize the best final answer."

        decision_result = await decision_agent.run(synthesis)
        pt, ct, _ = self._extract_usage(decision_result)
        total_pt += pt
        total_ct += ct

        hmf.finish_task(
            success=True,
            feedback=decision_result.text[:300],
        )

        return GenerateResult(
            text=decision_result.text,
            prompt_tokens=total_pt,
            completion_tokens=total_ct,
            total_tokens=total_pt + total_ct,
        )
