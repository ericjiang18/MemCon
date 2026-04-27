"""ExperienceBank — LLM-scored retrieval with generative re-ranking."""

from __future__ import annotations
import os, sys, re
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_core.documents import Document

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mas.memory.mas_memory.memory_base import MASMemoryBase
from mas.memory.common import MASMessage
from mas.llm import Message

_SYSTEM = "You are an agent designed to score the relevance between two pieces of text."
_USER = (
    "You will be given a past case and an ongoing task. "
    "Evaluate how relevant the past case is for the ongoing task, on a scale of 0-10.\n"
    "Past case:\n{trajectory}\n"
    "Ongoing task:\n{query}\n"
    "Score (just the number): "
)


@dataclass
class ExperienceBankMemory(MASMemoryBase):

    def __post_init__(self):
        super().__post_init__()
        self.main_memory = Chroma(
            embedding_function=self.embedding_func,
            persist_directory=self.persist_dir,
        )

    def add_memory(self, mas_message: MASMessage) -> None:
        meta_data = MASMessage.to_dict(mas_message)
        doc = Document(page_content=mas_message.task_main, metadata=meta_data)
        if mas_message.label is True or mas_message.label is False:
            self.main_memory.add_documents([doc])
        else:
            raise ValueError("mas_message must have label!")
        self._index_done()

    def _score_tasks(self, tasks, query_task):
        scored = []
        for t in tasks:
            traj = (t.task_description or "") + "\n" + (t.task_trajectory or "")
            prompt = _USER.format(trajectory=traj[:500], query=query_task[:200])
            try:
                resp = self.llm_model(
                    [Message("system", _SYSTEM), Message("user", prompt)], temperature=0.1,
                )
                match = re.fullmatch(r"\s*(\d+)\s*", resp.strip())
                s = int(match.group(1)) if match else 0
            except Exception:
                s = 0
            scored.append((s, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored]

    def retrieve_memory(self, query_task="", successful_topk=2, failed_topk=1, **kw):
        true_docs = self.main_memory.similarity_search_with_score(
            query=query_task, k=2 * successful_topk, filter={"label": True}
        ) if successful_topk > 0 else []
        false_docs = self.main_memory.similarity_search_with_score(
            query=query_task, k=2 * failed_topk, filter={"label": False}
        ) if failed_topk > 0 else []

        succ_2x = [MASMessage.from_dict(d[0].metadata) for d in true_docs]
        fail_2x = [MASMessage.from_dict(d[0].metadata) for d in false_docs]

        succ = self._score_tasks(succ_2x, query_task)[:successful_topk]
        fail = self._score_tasks(fail_2x, query_task)[:failed_topk]
        return succ, fail, []

    @property
    def memory_size(self):
        return len(self.main_memory.get()["ids"])
