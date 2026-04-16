from abc import ABC, abstractmethod
from typing import List, Tuple


class PromptSet(ABC):
    @staticmethod
    @abstractmethod
    def get_role() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_constraint(role: str) -> str:
        pass

    @abstractmethod
    def get_description(self, role: str) -> str:
        pass

    @abstractmethod
    def get_role_connection(self) -> List[Tuple[str, str]]:
        pass

    @staticmethod
    @abstractmethod
    def get_format() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_answer_prompt(question: str, role: str = "") -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_decision_constraint() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_decision_role() -> str:
        pass

    @staticmethod
    def get_decision_few_shot() -> str:
        return ""

    @staticmethod
    def postprocess_answer(answer: str) -> str:
        return answer.strip()
