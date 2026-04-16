"""
LangGraph + G-Memory — Multi-agent graph workflow augmented with
the original G-Memory (graph-based hierarchical memory).

Same architecture as langgraph_hmf but uses GMemory as the memory
backend, enabling fair comparison: langgraph vs langgraph+gmemory vs langgraph+ours.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TypedDict, Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BASELINE = os.path.join(_REPO_ROOT, "agent_baseline")
for _p in (_REPO_ROOT, _BASELINE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runners.base_runner import BaseRunner, GenerateResult, NUM_ROUNDS
from mas.memory.mas_memory.GMemory import GMemory
from mas.memory.mas_memory.memory_base import MASMemoryBase
from mas.memory.common import MASMessage
from mas.llm import GPTChat, Message
from mas.utils import EmbeddingFunc


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


class LangGraphGMemoryRunner(BaseRunner):
    framework_name = "langgraph_gmemory"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        self._task_idx = 0
        self._gmemory: Optional[GMemory] = None
        self._llm_model = GPTChat(model_name=self.model)
        self._embed_func = EmbeddingFunc("sentence-transformers/all-MiniLM-L6-v2")
        self._init_gmemory()

    def _init_gmemory(self):
        working_dir = os.path.join("/tmp", "hmf_gmemory_bench")
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(os.path.join(working_dir, "gmemory"), exist_ok=True)
        self._gmemory = GMemory(
            namespace="gmemory",
            global_config={"working_dir": working_dir, "hop": 1},
            llm_model=self._llm_model,
            embedding_func=self._embed_func,
        )

    def _retrieve_memory_context(self, query: str) -> str:
        try:
            result = self._gmemory.retrieve_memory(
                query_task=query,
                successful_topk=2,
                failed_topk=0,
                insight_topk=5,
                threshold=0.3,
            )
            parts = []
            successful_trajs = result[0] if len(result) > 0 else []
            insights = result[2] if len(result) > 2 else []

            if successful_trajs:
                parts.append("=== Past Successful Experiences ===")
                for t in successful_trajs[:2]:
                    desc = getattr(t, "task_description", "") or ""
                    traj = getattr(t, "task_trajectory", "") or ""
                    parts.append(f"{desc[:300]}\n{traj[:300]}")

            if insights:
                parts.append("\n=== Key Insights ===")
                for i, ins in enumerate(insights[:5], 1):
                    parts.append(f"{i}. {ins}")

            return "\n".join(parts) if parts else ""
        except Exception as e:
            print(f"[GMemory] retrieve error: {e}")
            return ""

    def _store_experience(self, query: str, answer: str, success: bool = True):
        try:
            self._gmemory.init_task_context(query[:200], query)
            self._gmemory.move_memory_state(
                action="answer",
                observation=answer[:500],
            )
            self._gmemory.save_task_context(label=success, feedback=answer[:300])
        except Exception as e:
            print(f"[GMemory] store error: {e}")

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerateResult:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langgraph.graph import StateGraph, END

        role_prompts = self.get_role_prompts()
        llm = self._llm

        self._task_idx += 1
        mem_ctx = self._retrieve_memory_context(user_prompt[:500])

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

        self._store_experience(user_prompt[:300], answer[:500])

        return GenerateResult(
            text=answer,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
        )
