"""Voyager — LLM summarizes trajectory before storage, retrieval by similarity.
Faithful re-implementation of /home/ubuntu/workplace/GMemory/mas/memory/mas_memory/voyager.py
"""

from __future__ import annotations
import os, sys
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
_SYSTEM = (
    "You are a helpful assistant that writes a description of the task resolution trajectory.\n"
    "1) Try to summarize the trajectory in no more than 6 sentences.\n"
    "2) Your response should be a single line of text."
)

_USER = """For example:
Trajectory:
You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a diningtable 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: heat some egg and put it in diningtable.
> think: To solve the task, I need to find and take an egg, then heat it with microwave, then put it in diningtable.
OK.

Summary: The agent found an egg in the fridge, heated it in the microwave, and placed it on the dining table to complete the task.

You are given the overall MAS trajectory on a task:
{task_trajectory}

Your job is to output a single-line summary describing the task and the trajectory (do not output anything irrelevant).
Your output:"""


@dataclass
class VoyagerMemory(MASMemoryBase):
    def __post_init__(self):
        super().__post_init__()
        self.main_memory = Chroma(
            embedding_function=self.embedding_func,
            persist_directory=self.persist_dir,
        )

    def add_memory(self, mas_message: MASMessage) -> None:
        prompt: str = _USER.format(
            task_trajectory=mas_message.task_description + mas_message.task_trajectory
        )
        messages: list[Message] = [Message('system', _SYSTEM), Message('user', prompt)]
        response: str = self.llm_model(messages, temperature=0.1)
        mas_message.task_main = response

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

    def retrieve_memory(
        self,
        query_task: str = "",
        successful_topk: int = 1,
        failed_topk: int = 1,
        **kargs
    ) -> tuple[list, list, list]:
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

        return true_task_messages, false_task_messages, []

    @property
    def memory_size(self):
        return len(self.main_memory.get()["ids"])
