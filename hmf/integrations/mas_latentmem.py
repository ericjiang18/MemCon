"""
LatentMem — Re-implementation of LatentMem's GMemory baseline.

Uses MemCon's own infrastructure (MASMemoryBase, GPTChat, EmbeddingFunc)
but implements LatentMem's algorithmic differences:
  - Simpler retrieval (no LLM relevance scoring, same as MemCon's G-Memory fork)
  - LatentMem-specific insight generation prompts
  - No merge_insights (LatentMem only does finetune, not periodic merge)
  - Different insight update interval defaults (20 instead of 5)

This avoids importing LatentMem's package (which requires trl, flash_attention_2,
and local HF models). All methods use the same LLM API as other baselines for
a fair comparison.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mas.memory.mas_memory.GMemory import GMemory
from mas.memory.common import MASMessage
from mas.llm import LLMCallable, Message
from mas.utils import EmbeddingFunc


# ── LatentMem-specific prompts (from LatentMem/latentmem/mas_core/memory/backbone/prompt.py) ──

_COMPARE_SYSTEM = (
    "You are an advanced reasoning agent that derives general rules from examples.\n"
    "You will receive one successful trial and one failed trial.\n\n"
    "Your goal:\n"
    "- Compare the positive and negative examples to extract insights that help avoid similar mistakes.\n"
    "- The insights must be concise and expressed as high-level reasoning principles, not tied to specific items or tasks."
)

_FORMAT_RULES = (
    "<OPERATION> <RULE NUMBER>: <RULE> (e.g. ADD: xxx, EDIT/REMOVE/AGREE 1: xxx)\n\n"
    "The available operations are: **AGREE (if the existing rule is strongly relevant for the task), "
    "REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), "
    "EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), "
    "ADD (add new rules that are very different from existing rules and relevant for other tasks). "
    "Each needs to CLOSELY follow their corresponding formatting below "
    "(any existing rule not edited, not agreed, nor removed is considered copied)**:\n\n"
    "AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\n"
    "REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\n"
    "EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\n"
    "ADD: <NEW RULE>\n\n"
    "Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. "
    "Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. "
    "Do at most 4 operations and each existing rule can only get a maximum of 1 operation."
)

_COMPARE_USER = (
    "## Successful Trial\n{pos_shot}\n\n"
    "## Failed Trial\n{neg_shot}\n\n"
    "## Existing Rules\n{existing_rules}\n\n"
    "## Your task:\n"
    "Compare the successful and failed trials. Update the rule list by adding, editing, "
    "removing, or agreeing so that the rules become general, high-level reasoning guidelines "
    "for avoiding similar failures.\nOutput only in the required format:\n" + _FORMAT_RULES
)

_SUMMARIZE_SYSTEM = (
    "You are an advanced reasoning agent capable of adding, editing, or removing rules "
    "from an existing rule set by forming new critiques of past task trajectories.\n"
    "You will receive a set of successful trajectories.\n\n"
    "Your goal:\n"
    "- Summarize general insights from these successful trajectories to guide future problem solving.\n"
    "- Ensure the insights are concise, expressed as high-level reasoning principles, "
    "and not tied to specific items or tasks."
)

_SUMMARIZE_USER = (
    "## Successful Trials\n{pos_shots}\n\n"
    "## Existing Rules\n{existing_rules}\n\n"
    "Your task:\n"
    "Based on the successful trials and existing rules, update the rule set "
    "(add, edit, remove, or agree).\n"
    "Ensure the final rules are high-level, general insights that guide better "
    "Thought and Action across diverse tasks.\n"
    "Follow the required output format:\n" + _FORMAT_RULES
)

_FINETUNE_SUFFIX = {
    "full": "Focus on REMOVE or EDIT or AGREE rules first, and stop ADD rule unless "
            "the new rule is VERY insightful and different from EXISTING RULES.",
    "not_full": "",
}


@dataclass
class LatentMemMemory(GMemory):
    """
    LatentMem GMemory variant.

    Inherits from MemCon's GMemory but applies LatentMem's specific differences:
      - Higher insight activation threshold (20 vs 5)
      - Higher insight update interval (20 vs 5)
      - No periodic merge_insights (only finetune)
      - LatentMem-specific prompts for insight generation
      - Simpler retrieval (inherited from MemCon's GMemory, no LLM scoring)
    """

    def __post_init__(self):
        # Override defaults to match LatentMem's config before calling super
        self.global_config.setdefault("start_insights_threshold", 20)
        self.global_config.setdefault("rounds_per_insights", 20)
        self.global_config.setdefault("insights_point_num", 5)

        super().__post_init__()
        print(f"[LatentMem] initialized (threshold={self._start_insights_threshold}, "
              f"interval={self._rounds_per_insights})")

    def add_memory(self, mas_message: MASMessage) -> None:
        """
        Same as GMemory but WITHOUT periodic merge_insights.
        LatentMem only does finetune, never merge.
        """
        mas_message = self._extract_mas_message(mas_message=mas_message)

        self.task_layer.add_task_node(mas_message.task_main)

        from langchain_core.documents import Document
        meta_data = MASMessage.to_dict(mas_message)
        memory_doc = Document(page_content=mas_message.task_main, metadata=meta_data)
        if mas_message.label is True or mas_message.label is False:
            self.main_memory.add_documents([memory_doc])
        else:
            raise ValueError("The mas_message must have label!")

        # LatentMem: finetune only, NO merge_insights
        if (self.memory_size >= self._start_insights_threshold
                and self.memory_size % self._rounds_per_insights == 0):
            self._latentmem_finetune_insights(self._insights_point_num)

        self._index_done()

    def _latentmem_finetune_insights(self, num_points: int):
        """
        Finetune insights using LatentMem's prompts instead of GMemory's.
        """
        import random
        import re

        all_ids = self.main_memory.get()["ids"]
        if not all_ids:
            return

        for _ in range(num_points):
            random_id = random.choice(all_ids)
            random_entry = self.main_memory.get(ids=[random_id])
            if "metadatas" not in random_entry or not random_entry["metadatas"]:
                continue
            mas_message = MASMessage.from_dict(random_entry["metadatas"][0])

            true_trajs, false_trajs = self.insights_layer._retrieve_memory(
                query_task=mas_message.task_main, successful_topk=3, failed_topk=1
            )
            if mas_message.label is True:
                true_trajs.append(mas_message)
            else:
                false_trajs.append(mas_message)

            all_task_mains = [t.task_main for t in true_trajs + false_trajs]
            related_ids, related_insights = self.insights_layer._find_related_insights(
                all_task_mains, len(all_task_mains) / 2
            )

            # Use LatentMem prompts
            rule_list = [self.insights_layer.insights_memory[i] for i in related_ids]
            existing_rules = [r["rule"] for r in rule_list]
            rule_text = "\n".join(f"{i}. {r}" for i, r in enumerate(existing_rules, 1)) if existing_rules else "No insights."

            suffix = _FINETUNE_SUFFIX["full"] if len(self.insights_layer.insights_memory) > 10 else _FINETUNE_SUFFIX["not_full"]

            # Compare pairs
            for idx, fail_task in enumerate(false_trajs):
                if idx >= len(true_trajs):
                    break
                succ_task = true_trajs[idx]
                pos_text = succ_task.task_description + "\n" + (succ_task.task_trajectory or "")
                neg_text = fail_task.task_description + "\n" + (fail_task.task_trajectory or "")

                prompt = _COMPARE_USER.format(
                    pos_shot=pos_text, neg_shot=neg_text, existing_rules=rule_text
                )
                if suffix:
                    prompt += "\n" + suffix

                response = self.llm_model(
                    [Message("system", _COMPARE_SYSTEM), Message("user", prompt)],
                    temperature=0.1,
                )
                ops = self.insights_layer._parse_rules(response)
                ops = self._map_ops(ops, related_ids)
                self.insights_layer._update_rules(
                    [succ_task.task_main, fail_task.task_main], ops, 10
                )

            # Success-only summarization
            if true_trajs:
                history = "\n\n".join(
                    t.task_description + "\n" + (t.task_trajectory or "")
                    for t in true_trajs[:5]
                )
                prompt = _SUMMARIZE_USER.format(pos_shots=history, existing_rules=rule_text)
                if suffix:
                    prompt += "\n" + suffix

                response = self.llm_model(
                    [Message("system", _SUMMARIZE_SYSTEM), Message("user", prompt)],
                    temperature=0.1,
                )
                ops = self.insights_layer._parse_rules(response)
                ops = self._map_ops(ops, related_ids)
                self.insights_layer._update_rules(
                    [t.task_main for t in true_trajs], ops, 10
                )

        self.insights_layer.clear_insights()
        self.insights_layer._index_done()

    @staticmethod
    def _map_ops(operations, insight_ids):
        """Map parsed operation indices back to global insight indices."""
        processed = []
        for operation, text in operations:
            parts = operation.split(" ")
            if "ADD" in parts:
                operation = "ADD"
            elif len(parts) == 2:
                if not insight_ids:
                    continue
                try:
                    idx = int(parts[1]) - 1
                except ValueError:
                    continue
                if idx < 0 or idx >= len(insight_ids):
                    continue
                parts[1] = str(insight_ids[idx] + 1)
                operation = " ".join(parts)
            processed.append((operation, text))
        return processed
