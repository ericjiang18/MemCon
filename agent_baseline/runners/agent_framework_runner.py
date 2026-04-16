"""Runner for Microsoft Agent Framework — multi-agent orchestrated pipeline.

Creates domain-specific agents with different roles and runs them in a
sequential pipeline with inter-agent communication over multiple rounds,
followed by a decision agent that synthesises the final answer.
"""

from .base_runner import BaseRunner, GenerateResult, NUM_ROUNDS


class AgentFrameworkRunner(BaseRunner):
    framework_name = "agent-framework"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from agent_framework.openai import OpenAIChatClient

        self._client = OpenAIChatClient(
            model_id=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

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

        agents = []
        for role_name, role_desc in role_prompts:
            agent = self._client.as_agent(
                name=role_name.replace(" ", "_"),
                instructions=(
                    f"{role_desc}\n\n"
                    "Consider other agents' analysis critically. "
                    "Do not simply agree with the majority."
                ),
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

                if prior_text:
                    prompt = (
                        f"{user_prompt}\n\n"
                        f"Other agents' outputs from previous discussion:\n{prior_text}"
                    )
                else:
                    prompt = user_prompt

                result = await agent.run(prompt)
                pt, ct, _ = self._extract_usage(result)
                total_pt += pt
                total_ct += ct
                agent_outputs[role_name].append(result.text)

        decision_agent = self._client.as_agent(
            name="Decision_Maker",
            instructions=system_prompt,
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

        return GenerateResult(
            text=decision_result.text,
            prompt_tokens=total_pt,
            completion_tokens=total_ct,
            total_tokens=total_pt + total_ct,
        )
