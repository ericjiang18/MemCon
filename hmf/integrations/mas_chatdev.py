"""ChatDev — Periodic LLM summarization every 10 steps. No cross-task retrieval.
Faithful re-implementation of /home/ubuntu/workplace/GMemory/mas/memory/mas_memory/chatdev.py
"""

from __future__ import annotations
import os, sys
from dataclasses import dataclass

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mas.memory.mas_memory.memory_base import MASMemoryBase
from mas.memory.common import MASMessage
from mas.llm import Message

# Original prompts from GMemory/mas/memory/mas_memory/prompt.py
_SYSTEM = """You are an agent skilled in summarization. Your task is to generate **phase-based summaries** from given execution records of an agent's task. These summaries help the agent efficiently utilize existing information, avoid redundant computations, and ensure task continuity.

## **Requirements for Your Summary:**
1. **Phase-based summarization**: Organize execution records into logical phases and extract key steps.
2. **Task relevance**: Ensure the summary helps the agent understand what has been completed and what needs to be done next.
3. **Clarity and conciseness**: Use clear and precise language to summarize the information while avoiding unnecessary details.

## **Additional Guidelines:**
- Maintain **contextual consistency** so that the agent can seamlessly continue the task.
- If there are incorrect intermediate states or irrelevant information, filter or correct them to make the summary more accurate."""

_USER = """You will be given a partial execution record of an agent's task. Your job is to generate a **phase-based summary** that the agent can understand and use to continue the task.

## **Your Summary Should Follow These Guidelines:**
1. **Phase-based summarization**: Break the record into logical steps, ensuring that each phase's key tasks are captured.
2. **Efficient information transfer**:
   - Document key task objectives, executed actions, and the current state.
   - Identify unfinished parts to help the agent determine the next steps.
3. **Prevent information loss**:
   - Include critical decision points, state changes, and key computation processes.
   - If there are uncertainties, retain relevant details for future judgment.

Task: {task}

Execution Record:
{task_trajectory}

Summary:"""


@dataclass
class ChatDevMemory(MASMemoryBase):
    def __post_init__(self):
        super().__post_init__()
        os.makedirs(self.persist_dir, exist_ok=True)
        self.counter: int = 0

    def summarize(self, **kargs) -> str:
        self.counter += 1
        if self.counter % 10 != 0:
            return super().summarize()

        mas_message: MASMessage = self.current_task_context

        if self.current_task_context is None:
            raise RuntimeError('The current task memory is empty.')

        user_prompt: str = _USER.format(
            task=mas_message.task_description,
            task_trajectory=mas_message.task_trajectory
        )
        messages: list[Message] = [Message('system', _SYSTEM), Message('user', user_prompt)]

        response: str = self.llm_model(messages)
        return self.current_task_context.task_description + '\n' + response

    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        self.counter = 0
        return super().save_task_context(label, feedback=feedback)
