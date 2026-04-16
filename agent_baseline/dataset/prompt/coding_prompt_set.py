"""Coding prompt set — adapted from Topodim/prompt/humaneval_prompt_set.py
Covers: humaneval, livecodebench
"""

from typing import Dict, Any, List, Union
import itertools
from .prompt_set import PromptSet
from .prompt_set_registry import PromptSetRegistry
from .common import get_combine_materials

roles = itertools.cycle([
    "Project Manager",
    "Algorithm Designer",
    "Programming Expert",
    "Test Analyst",
    "Bug Fixer",
])

ROLE_DESCRIPTION = {
    "Project Manager":
        "You are a project manager. "
        "You will be given a function signature and its docstring by the user. "
        "You are responsible for overseeing the overall structure of the code, ensuring that the code is structured to complete the task. Implement code concisely and correctly without pursuing over-engineering. "
        "You need to suggest optimal design patterns to ensure that the code follows best practices for maintainability and flexibility. "
        "You can specify the overall design of the code, including the classes that need to be defined(maybe none) and the functions used (maybe only one function). "
        "I hope your reply will be more concise. Preferably within fifty words. Don't list too many points.",
    "Algorithm Designer":
        "You are an algorithm designer. "
        "You will be given a function signature and its docstring by the user. "
        "You need to specify the specific design of the algorithm, including the classes that may be defined and the functions used. "
        "You need to generate the detailed documentation, including explanations of the algorithm, usage instructions, and API references. "
        "When the implementation logic is complex, you can give the pseudocode logic of the main algorithm. "
        "I hope your reply will be more concise. Preferably within fifty words. Don't list too many points.",
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a function signature and its docstring by the user. "
        "You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks.",
    "Test Analyst":
        "You are a test analyst. "
        "You will be given a function signature and its docstring by the user. "
        "You need to provide problems in the current code or solution based on the test data and possible test feedback in the question. "
        "You need to provide additional special use cases, boundary conditions, etc. that should be paid attention to when writing code. "
        "You can point out any potential errors in the code. "
        "I hope your reply will be more concise. Preferably within fifty words. Don't list too many points.",
    "Bug Fixer":
        "You are a bug fixer. "
        "You will be given a function signature and its docstring by the user. "
        "You need to provide modified and improved python code based on the current overall code design, algorithm framework, code implementation or test problems. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks.",
    "Normal Programmer":
        "You are a programmer. "
        "You will be given a function signature and its docstring by the user. "
        "You can refer to the agents' outputs. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response.",
}

ROLE_CONNECTION = [
    ("Project Manager", "Query", "Algorithm Designer"),
    ("Algorithm Designer", "Spatial", "Programming Expert"),
    ("Programming Expert", "Query", "Test Analyst"),
    ("Test Analyst", "Query", "Bug Fixer"),
    ("Algorithm Designer", "Query", "Project Manager"),
    ("Programming Expert", "Query", "Bug Fixer"),
    ("Bug Fixer", "Spatial", "Programming Expert"),
    ("Test Analyst", "Debate", "Programming Expert"),
    ("Algorithm Designer", "Query", "Test Analyst"),
    ("Project Manager", "Query", "Programming Expert"),
]


@PromptSetRegistry.register("humaneval")
@PromptSetRegistry.register("livecodebench")
class CodingPromptSet(PromptSet):
    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION.get(role, ROLE_DESCRIPTION["Programming Expert"])

    def get_description(self, role):
        return ROLE_DESCRIPTION.get(role, ROLE_DESCRIPTION["Programming Expert"])

    def get_role_connection(self):
        return ROLE_CONNECTION

    @staticmethod
    def get_format():
        return "python code"

    @staticmethod
    def get_answer_prompt(question, role="Programming Expert"):
        return f"{question}"

    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a function signature and its docstring by the user. "
            "You may be given the overall code design, algorithm framework, code implementation or test problems. "
            "Write your full implementation (restate the function signature). "
            "If the prompt given to you contains code that passed internal testing, you can choose the most reliable reply. "
            "If there is no code that has passed internal testing in the prompt, you can change it yourself according to the prompt. "
            "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
            "Do not include anything other than Python code blocks in your response."
        )

    @staticmethod
    def get_decision_role():
        return (
            "You are the top decision-maker and are good at analyzing and summarizing "
            "other people's opinions, finding errors and giving final answers. "
            "And you are an AI that only responds with only python code."
        )

    @staticmethod
    def get_decision_few_shot():
        return ""

    @staticmethod
    def postprocess_answer(answer):
        if "```python" in answer:
            return answer.split("```python")[1].split("```")[0].strip()
        return answer.strip()

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
