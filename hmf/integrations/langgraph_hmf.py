"""
LangGraph + HMF  —  Multi-agent graph workflow augmented with
hierarchical memory and MPC-controlled memory access.

Adds two extra nodes to the original LangGraph runner:
  1. memory_inject  (entry node) — uses MPC to query cache/retrieval/skill
     memory and prepends relevant context to the task prompt.
  2. memory_write   (exit node)  — after the decision node, stores the
     experience and triggers maintenance via MPC.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TypedDict, Dict, List

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BASELINE = os.path.join(_REPO_ROOT, "agent_baseline")
for _p in (_REPO_ROOT, _BASELINE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runners.base_runner import BaseRunner, GenerateResult, NUM_ROUNDS
from ..agent.hmf_agent import HMFAgent
from ..config import HMFConfig


class _GraphState(TypedDict):
    task: str
    original_task: str
    system_prompt: str
    agent_outputs: Dict[str, List[str]]
    round: int
    final_answer: str
    prompt_tokens: int
    completion_tokens: int
    memory_context: str


class LangGraphHMFRunner(BaseRunner):
    framework_name = "langgraph_hmf"

    def __init__(self, hmf_config: HMFConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        cfg = hmf_config or HMFConfig(
            model_name=self.model,
            api_base=self.base_url,
            api_key=self.api_key,
        )
        self.hmf = HMFAgent(cfg)
        self._task_idx = 0

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerateResult:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langgraph.graph import StateGraph, END

        role_prompts = self.get_role_prompts()
        llm = self._llm
        hmf = self.hmf

        self._task_idx += 1
        task_type = self.domain or "general"
        context = hmf.init_task(user_prompt[:200], task_type, user_prompt)
        mem_ctx = hmf.build_memory_context(context)

        def _extract_tokens(resp):
            pt = ct = 0
            um = getattr(resp, "usage_metadata", None)
            if um and isinstance(um, dict):
                pt = um.get("input_tokens", 0) or 0
                ct = um.get("output_tokens", 0) or 0
            if not pt:
                ri = getattr(resp, "response_metadata", None)
                if ri and isinstance(ri, dict):
                    tu = ri.get("token_usage") or ri.get("usage") or {}
                    if isinstance(tu, dict):
                        pt = tu.get("prompt_tokens", 0) or 0
                        ct = tu.get("completion_tokens", 0) or 0
            return pt, ct

        def _build_agent_node(role_name: str, role_desc: str):
            async def _node(state: _GraphState) -> dict:
                prior_text = ""
                for name, outputs in state["agent_outputs"].items():
                    if outputs:
                        prior_text += f"\n--- {name} ---\n{outputs[-1]}\n"

                agent_system = (
                    f"{role_desc}\n\n"
                    "Consider other agents' analysis critically. "
                    "Do not simply agree with the majority."
                )
                if state.get("memory_context"):
                    agent_system += f"\n\n{state['memory_context']}"

                agent_user = state["task"]
                if prior_text:
                    agent_user += f"\n\nOther agents' outputs from previous discussion:\n{prior_text}"

                resp = await llm.ainvoke([
                    SystemMessage(content=agent_system),
                    HumanMessage(content=agent_user),
                ])
                pt, ct = _extract_tokens(resp)

                new_outputs = dict(state["agent_outputs"])
                new_outputs.setdefault(role_name, [])
                new_outputs[role_name] = new_outputs[role_name] + [resp.content]

                return {
                    "agent_outputs": new_outputs,
                    "prompt_tokens": state["prompt_tokens"] + pt,
                    "completion_tokens": state["completion_tokens"] + ct,
                }
            return _node

        async def _round_gate(state: _GraphState) -> dict:
            return {"round": state["round"] + 1}

        def _should_continue(state: _GraphState) -> str:
            if state["round"] < NUM_ROUNDS:
                safe = role_prompts[0][0].replace(" ", "_").replace("-", "_")
                return f"agent_{safe}"
            return "decision"

        async def _decision(state: _GraphState) -> dict:
            synthesis = f"Task:\n{state['original_task']}\n\nAgents' responses:\n"
            for name, outputs in state["agent_outputs"].items():
                for r_idx, resp in enumerate(outputs):
                    synthesis += f"\n--- {name} (round {r_idx + 1}) ---\n{resp}\n"
            synthesis += "\nSynthesize the best final answer."

            dec_system = state["system_prompt"]
            if state.get("memory_context"):
                dec_system += f"\n\n{state['memory_context']}"

            resp = await llm.ainvoke([
                SystemMessage(content=dec_system),
                HumanMessage(content=synthesis),
            ])
            pt, ct = _extract_tokens(resp)

            hmf.record_step(
                action="decision",
                observation=resp.content[:500],
                token_cost=pt + ct,
            )

            return {
                "final_answer": resp.content,
                "prompt_tokens": state["prompt_tokens"] + pt,
                "completion_tokens": state["completion_tokens"] + ct,
            }

        builder = StateGraph(_GraphState)

        agent_node_names = []
        for role_name, role_desc in role_prompts:
            safe = role_name.replace(" ", "_").replace("-", "_")
            node_name = f"agent_{safe}"
            builder.add_node(node_name, _build_agent_node(role_name, role_desc))
            agent_node_names.append(node_name)

        builder.add_node("round_gate", _round_gate)
        builder.add_node("decision", _decision)

        builder.set_entry_point(agent_node_names[0])
        for i in range(len(agent_node_names) - 1):
            builder.add_edge(agent_node_names[i], agent_node_names[i + 1])
        builder.add_edge(agent_node_names[-1], "round_gate")

        builder.add_conditional_edges(
            "round_gate",
            _should_continue,
            {name: name for name in agent_node_names} | {"decision": "decision"},
        )
        builder.add_edge("decision", END)

        graph = builder.compile()

        init_state: _GraphState = {
            "task": user_prompt,
            "original_task": user_prompt,
            "system_prompt": system_prompt,
            "agent_outputs": {},
            "round": 1,
            "final_answer": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "memory_context": mem_ctx,
        }

        result = await graph.ainvoke(init_state)

        pt = result.get("prompt_tokens", 0)
        ct = result.get("completion_tokens", 0)
        answer = result.get("final_answer", "")

        hmf.finish_task(
            success=True,
            feedback=answer[:300],
        )

        return GenerateResult(
            text=answer,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
        )
