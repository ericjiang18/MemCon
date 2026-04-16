"""Runner for Microsoft AutoGen — multi-agent group chat.

Uses RoundRobinGroupChat with domain-specific agent roles (3 solvers + 1
decision round).  Each agent sees previous agents' outputs, matching the
multi-agent collaboration pattern used in agent_q_mix.
"""

from .base_runner import BaseRunner, GenerateResult, NUM_ROUNDS


class AutoGenRunner(BaseRunner):
    framework_name = "autogen"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        self._model_client = OpenAIChatCompletionClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            },
        )

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerateResult:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_agentchat.conditions import MaxMessageTermination

        role_prompts = self.get_role_prompts()

        agents = []
        for i, (role_name, role_desc) in enumerate(role_prompts):
            safe_name = role_name.replace(" ", "_").replace("-", "_")
            agent = AssistantAgent(
                name=f"{safe_name}_{i}",
                model_client=self._model_client,
                system_message=(
                    f"{role_desc}\n\n"
                    "When responding, consider and critically evaluate the analysis "
                    "from other agents. Do not simply agree with the majority."
                ),
            )
            agents.append(agent)

        n_agents = len(agents)
        max_messages = n_agents * NUM_ROUNDS

        team = RoundRobinGroupChat(
            participants=agents,
            termination_condition=MaxMessageTermination(max_messages=max_messages),
        )

        result = await team.run(task=user_prompt)

        agent_outputs = []
        pt = ct = 0
        for msg in result.messages:
            content = getattr(msg, "content", None)
            if content and isinstance(content, str):
                agent_outputs.append(content)
            mu = getattr(msg, "models_usage", None)
            if mu:
                pt += getattr(mu, "prompt_tokens", 0) or 0
                ct += getattr(mu, "completion_tokens", 0) or 0

        decision_agent = AssistantAgent(
            name="Decision_Maker",
            model_client=self._model_client,
            system_message=system_prompt,
        )

        synthesis_prompt = f"Task:\n{user_prompt}\n\nAgents' responses:\n"
        for i, (role_name, _) in enumerate(role_prompts):
            rounds_for_agent = [
                agent_outputs[j]
                for j in range(i, len(agent_outputs), n_agents)
            ]
            for r_idx, resp in enumerate(rounds_for_agent):
                synthesis_prompt += (
                    f"\n--- {role_name} (round {r_idx + 1}) ---\n{resp}\n"
                )
        synthesis_prompt += "\nSynthesize the best final answer."

        decision_result = await decision_agent.run(task=synthesis_prompt)

        decision_text = ""
        for msg in reversed(decision_result.messages):
            content = getattr(msg, "content", None)
            if content and isinstance(content, str):
                decision_text = content
                break
        for msg in decision_result.messages:
            mu = getattr(msg, "models_usage", None)
            if mu:
                pt += getattr(mu, "prompt_tokens", 0) or 0
                ct += getattr(mu, "completion_tokens", 0) or 0

        return GenerateResult(
            text=decision_text,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
        )
