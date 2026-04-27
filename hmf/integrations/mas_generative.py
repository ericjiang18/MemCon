"""Generative — LLM scores relevance of retrieved trajectories, returns top-k.
Faithful re-implementation of /home/ubuntu/workplace/GMemory/mas/memory/mas_memory/generative.py
"""

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

# Original prompts from GMemory/mas/memory/mas_memory/prompt.py
_SYSTEM = """You are an agent designed to score the relevance between two pieces of text."""
_USER = '''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Score: '''


@dataclass
class GenerativeMemory(MASMemoryBase):
    def __post_init__(self):
        super().__post_init__()
        self.main_memory = Chroma(
            embedding_function=self.embedding_func,
            persist_directory=self.persist_dir,
        )

    def add_memory(self, mas_message: MASMessage) -> None:
        meta_data: dict = MASMessage.to_dict(mas_message)
        memory_doc = Document(
            page_content=mas_message.task_main,
            metadata=meta_data,
        )
        if mas_message.label == True or mas_message.label == False:
            self.main_memory.add_documents([memory_doc])
        else:
            raise ValueError('The mas_message must have label!')
        self._index_done()

    def _retrieve_memory_raw(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
    ) -> tuple[list[MASMessage], list[MASMessage]]:

        true_tasks_doc: list = []
        false_tasks_doc: list = []

        if successful_topk != 0:
            true_tasks_doc = self.main_memory.similarity_search_with_score(
                query=query_task, k=successful_topk, filter={'label': True}
            )
        if failed_topk != 0:
            false_tasks_doc = self.main_memory.similarity_search_with_score(
                query=query_task, k=failed_topk, filter={'label': False}
            )
        sorted(true_tasks_doc, key=lambda x: x[1])
        sorted(false_tasks_doc, key=lambda x: x[1])

        true_task_messages: list[MASMessage] = []
        false_task_messages: list[MASMessage] = []
        for doc in true_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            true_task_messages.append(mas_message)

        for doc in false_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            false_task_messages.append(mas_message)

        return true_task_messages, false_task_messages

    def retrieve_memory(
        self,
        query_task: str = "",
        successful_topk: int = 1,
        failed_topk: int = 1,
        **kargs
    ) -> tuple[list, list, list]:
        successful_task_trajectories, failed_task_trajectories = self._retrieve_memory_raw(
            query_task, 2 * successful_topk, 2 * failed_topk)

        importance_score: list[float] = []
        for success_task in successful_task_trajectories:
            prompt: str = _USER.format(
                trajectory=success_task.task_description + '\n' + success_task.task_trajectory,
                query_scenario=query_task
            )
            response: str = self.llm_model(messages=[Message('system', _SYSTEM), Message('user', prompt)])
            score = int(re.search(r'\d+', response).group()) if re.search(r'\d+', response) else 0
            importance_score.append(score)

        sorted_success_tasks = [task for _, task in sorted(zip(importance_score, successful_task_trajectories),
                                                           key=lambda x: x[0], reverse=True)]
        top_success_task_trajectories = sorted_success_tasks[:successful_topk]
        top_fail_task_trajectories = failed_task_trajectories[:failed_topk]

        return top_success_task_trajectories, top_fail_task_trajectories, []

    @property
    def memory_size(self):
        return len(self.main_memory.get()["ids"])
