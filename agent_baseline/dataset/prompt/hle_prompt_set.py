"""HLE (Humanity's Last Exam) prompt set.

Inherits AgenticPromptSet's multi-role setup (Knowlegable Expert, Critic,
Mathematician, etc.) but overrides constraint and postprocessing to handle
both multiple-choice AND exact-match question types.
"""

from typing import Union, List
import re
from .prompt_set_registry import PromptSetRegistry
from .agentic_prompt_set import AgenticPromptSet


@PromptSetRegistry.register("hle")
class HLEPromptSet(AgenticPromptSet):

    @staticmethod
    def get_constraint(role=None):
        return (
            "You will be given a very challenging question. It may be multiple choice or open-ended.\n"
            "For multiple choice questions: select the correct option letter.\n"
            "For open-ended questions: provide a precise, concise answer.\n"
            "Your answer can refer to the answers of other agents provided to you. "
            "Please think critically and not just follow the answers of the majority.\n"
            "Think step by step. Your reply must be less than 200 words.\n"
            "The LAST line of your reply must contain ONLY your final answer "
            "(a single letter for multiple choice, or the exact short answer for open-ended)."
        )

    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a very challenging question and other agents' analyses.\n"
            "Synthesize their reasoning and provide the best final answer.\n"
            "For multiple choice: reply with ONLY the correct letter.\n"
            "For open-ended: reply with ONLY the exact answer, as concise as possible.\n"
            "Your reply must contain ONLY the final answer and nothing else."
        )

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            return ""

        answer = answer.strip()
        if not answer:
            return ""

        lines = [l.strip() for l in answer.split("\n") if l.strip()]
        if not lines:
            return ""

        last = lines[-1]

        mc = re.match(r"^([A-J])\.?\s*$", last, re.IGNORECASE)
        if mc:
            return mc.group(1).upper()

        for prefix in ["The answer is: ", "The answer is ", "Answer: ", "ANSWER: ",
                        "Final answer: ", "Final Answer: "]:
            if last.lower().startswith(prefix.lower()):
                last = last[len(prefix):].strip().rstrip(".")
                break

        mc2 = re.match(r"^([A-J])\.?\s*$", last, re.IGNORECASE)
        if mc2:
            return mc2.group(1).upper()

        return last
