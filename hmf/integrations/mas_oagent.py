"""OAgent — Insight-based learning without task graph (simpler than G-Memory)."""

from __future__ import annotations
import json, os, sys, re, random
from collections import defaultdict
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_core.documents import Document

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mas.memory.mas_memory.memory_base import MASMemoryBase
from mas.memory.common import MASMessage
from mas.llm import Message

_COMPARE_SYSTEM = (
    "You are an advanced reasoning agent that derives general rules from examples.\n"
    "You will receive one successful trial and one failed trial.\n\n"
    "Your goal:\n"
    "- Compare the positive and negative examples to extract insights.\n"
    "- The insights must be concise and expressed as high-level reasoning principles."
)

_FORMAT = (
    "AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\n"
    "REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\n"
    "EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\n"
    "ADD: <NEW RULE>\n\n"
    "Do at most 4 operations and each existing rule can only get a maximum of 1 operation."
)

_COMPARE_USER = (
    "## Successful Trial\n{pos_shot}\n\n"
    "## Failed Trial\n{neg_shot}\n\n"
    "## Existing Rules\n{existing_rules}\n\n"
    "Compare the trials. Update rules using:\n" + _FORMAT
)

_SUCCESS_SYSTEM = (
    "You are an advanced reasoning agent that summarizes insights from successful trajectories."
)

_SUCCESS_USER = (
    "## Successful Trials\n{pos_shots}\n\n"
    "## Existing Rules\n{existing_rules}\n\n"
    "Update rules using:\n" + _FORMAT
)


def _basic_retrieve(main_memory, query_task, successful_topk, failed_topk):
    true_docs = main_memory.similarity_search_with_score(
        query=query_task, k=successful_topk, filter={"label": True}
    ) if successful_topk > 0 else []
    false_docs = main_memory.similarity_search_with_score(
        query=query_task, k=failed_topk, filter={"label": False}
    ) if failed_topk > 0 else []
    succ = [MASMessage.from_dict(d[0].metadata) for d in sorted(true_docs, key=lambda x: x[1])]
    fail = [MASMessage.from_dict(d[0].metadata) for d in sorted(false_docs, key=lambda x: x[1])]
    return succ, fail


@dataclass
class OAgentMemory(MASMemoryBase):

    def __post_init__(self):
        super().__post_init__()
        self.main_memory = Chroma(
            embedding_function=self.embedding_func,
            persist_directory=self.persist_dir,
        )
        self._insights_path = os.path.join(self.persist_dir, "oagent_insights.json")
        self._insights: list[dict] = []
        if os.path.exists(self._insights_path):
            with open(self._insights_path) as f:
                self._insights = json.load(f)
        self._insights_cache: list[str] = []
        self._threshold = self.global_config.get("start_insights_threshold", 20)
        self._interval = self.global_config.get("rounds_per_insights", 20)
        self._iters = self.global_config.get("insights_point_num", 5)

    def add_memory(self, mas_message: MASMessage) -> None:
        meta_data = MASMessage.to_dict(mas_message)
        doc = Document(page_content=mas_message.task_main, metadata=meta_data)
        if mas_message.label is True or mas_message.label is False:
            self.main_memory.add_documents([doc])
        else:
            raise ValueError("mas_message must have label!")
        if self.memory_size >= self._threshold and self.memory_size % self._interval == 0:
            self._finetune_insights()
        self._save_insights()

    def retrieve_memory(self, query_task="", successful_topk=2, failed_topk=1,
                        insight_topk=10, **kw):
        succ, fail = _basic_retrieve(self.main_memory, query_task, successful_topk, failed_topk)
        scored = defaultdict(float)
        for tm in [t.task_main for t in succ + fail] + [query_task]:
            for ins in self._insights:
                if tm in ins.get("positive_correlation_tasks", []):
                    scored[ins["rule"]] += 1
        sorted_ins = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        insights = [r for r, _ in sorted_ins[:insight_topk]]
        self._insights_cache = insights
        return succ, fail, insights

    def backward(self, reward, **kwargs):
        for text in self._insights_cache:
            for ins in self._insights:
                if text in ins.get("rule", ""):
                    ins["score"] += (-2 if reward is False else 1)
        self._insights = [i for i in self._insights if i["score"] > 0]
        self._save_insights()
        self._insights_cache = []

    def _finetune_insights(self):
        all_ids = self.main_memory.get()["ids"]
        if not all_ids:
            return
        for _ in range(self._iters):
            rid = random.choice(all_ids)
            entry = self.main_memory.get(ids=[rid])
            if not entry.get("metadatas"):
                continue
            msg = MASMessage.from_dict(entry["metadatas"][0])
            succ, fail = _basic_retrieve(self.main_memory, msg.task_main, 3, 1)
            if msg.label is True:
                succ.append(msg)
            else:
                fail.append(msg)

            rules = [i["rule"] for i in self._insights]
            rule_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules)) or "No insights."

            for idx, f_task in enumerate(fail):
                if idx >= len(succ):
                    break
                pos = (succ[idx].task_description or "") + "\n" + (succ[idx].task_trajectory or "")
                neg = (f_task.task_description or "") + "\n" + (f_task.task_trajectory or "")
                prompt = _COMPARE_USER.format(pos_shot=pos[:500], neg_shot=neg[:500], existing_rules=rule_text)
                resp = self.llm_model([Message("system", _COMPARE_SYSTEM), Message("user", prompt)], temperature=0.1)
                self._apply_ops(resp, [s.task_main for s in succ] + [f_task.task_main])

            if succ:
                history = "\n\n".join((s.task_description or "") + "\n" + (s.task_trajectory or "") for s in succ[:5])
                prompt = _SUCCESS_USER.format(pos_shots=history[:1000], existing_rules=rule_text)
                resp = self.llm_model([Message("system", _SUCCESS_SYSTEM), Message("user", prompt)], temperature=0.1)
                self._apply_ops(resp, [s.task_main for s in succ])

        self._insights = [i for i in self._insights if i["score"] > 0]
        self._save_insights()

    def _apply_ops(self, llm_text, task_mains):
        pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)'
        for operation, text in re.findall(pattern, llm_text):
            text = text.strip()
            if not text or not text.endswith("."):
                continue
            op_type = operation.split(" ")[0]
            if op_type == "ADD":
                self._insights.append({"rule": text, "score": 2,
                                       "positive_correlation_tasks": list(task_mains),
                                       "negative_correlation_tasks": []})
            elif " " in operation:
                try:
                    idx = int(operation.split(" ")[1]) - 1
                    if 0 <= idx < len(self._insights):
                        if op_type == "AGREE":
                            self._insights[idx]["score"] += 1
                        elif op_type == "REMOVE":
                            self._insights[idx]["score"] -= 3
                        elif op_type == "EDIT":
                            self._insights[idx]["rule"] = text
                            self._insights[idx]["score"] += 1
                except (ValueError, IndexError):
                    pass

    def _save_insights(self):
        with open(self._insights_path, "w") as f:
            json.dump(self._insights, f, indent=2)

    @property
    def memory_size(self):
        return len(self.main_memory.get()["ids"])
