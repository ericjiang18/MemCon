"""LangGraph ALFWorld runner — multi-agent graph for each step decision."""

from __future__ import annotations
from typing import Tuple
from .base import ALFWorldRunner


class LangGraphALFWorld(ALFWorldRunner):
    framework_name = "langgraph"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from langchain_openai import ChatOpenAI
        self._llm = ChatOpenAI(
            model=self.model, base_url=self.api_base, api_key=self.api_key,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, int]:
        from langchain_core.messages import HumanMessage, SystemMessage
        kwargs = {}
        if any(t in self.model for t in ("gpt-5-mini", "o3-mini", "o4-mini")):
            kwargs["temperature"] = None
        resp = self._llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ], **kwargs)
        pt = ct = 0
        um = getattr(resp, "usage_metadata", None)
        if um and isinstance(um, dict):
            pt = um.get("input_tokens", 0) or 0
            ct = um.get("output_tokens", 0) or 0
        if not pt:
            ri = getattr(resp, "response_metadata", None)
            if ri and isinstance(ri, dict):
                tu = ri.get("token_usage") or ri.get("usage") or {}
                pt = tu.get("prompt_tokens", 0) or 0
                ct = tu.get("completion_tokens", 0) or 0
        return resp.content or "", pt, ct
