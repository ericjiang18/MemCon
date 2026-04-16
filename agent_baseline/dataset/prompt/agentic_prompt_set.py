"""Agentic/reasoning prompt set — adapted from Topodim/prompt/mmlu_prompt_set.py
Covers: mmlu_pro, gaia (+ frontierscience via gaia domain)
"""

from typing import Union, Dict, Any, List
import itertools
from .prompt_set import PromptSet
from .prompt_set_registry import PromptSetRegistry
from .common import get_combine_materials

roles = itertools.cycle([
    "Knowlegable Expert",
    "Critic",
    "Mathematician",
    "Psychologist",
    "Historian",
    "Doctor",
    "Lawyer",
    "Economist",
    "Programmer",
])

ROLE_DESCRIPTION = {
    "Knowlegable Expert":
        "You are a knowlegable expert in question answering.\n"
        "Please give several key entities that need to be searched in wikipedia to solve the problem, for example: catfish effect, broken window effect, Shakespeare.\n"
        "If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it\n",
    "Wiki Searcher":
        "You will be given a question and a wikipedia overview of the key entities within it.\n"
        "Please refer to them step by step to give your answer.\n"
        "And point out potential issues in other agent's analysis.\n",
    "Critic":
        "You are an excellent critic.\n"
        "Please point out potential issues in other agent's analysis point by point.\n",
    "Mathematician":
        "You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.\n",
    "Psychologist":
        "You are a psychologist.\n"
        "You are good at psychology, sociology, and philosophy.\n"
        "You give people scientific suggestions that will make them feel better.\n",
    "Historian":
        "You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.\n",
    "Doctor":
        "You are a doctor and come up with creative treatments for illnesses or diseases.\n"
        "You are able to recommend conventional medicines, herbal remedies and other natural alternatives. \n"
        "You also consider the patient's age, lifestyle and medical history when providing your recommendations.\n",
    "Lawyer":
        "You are good at law, politics, and history.\n",
    "Economist":
        "You are good at economics, finance, and business.\n"
        "You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.\n",
    "Programmer":
        "You are good at computer science, engineering, and physics.\n"
        "You have experience in designing and developing computer software and hardware.\n",
}

ROLE_CONNECTION = [
    ("Knowlegable Expert", "Query", "Mathematician"),
    ("Knowlegable Expert", "Query", "Economist"),
    ("Knowlegable Expert", "Query", "Lawyer"),
    ("Knowlegable Expert", "Query", "Critic"),
    ("Knowlegable Expert", "Query", "Psychologist"),
    ("Knowlegable Expert", "Query", "Doctor"),
    ("Knowlegable Expert", "Query", "Historian"),
    ("Knowlegable Expert", "Command", "Programmer"),
    ("Knowlegable Expert", "Query", "Critic"),
    ("Mathematician", "Debate", "Critic"),
    ("Mathematician", "Debate", "Critic"),
    ("Psychologist", "Debate", "Critic"),
    ("Economist", "Query", "Lawyer"),
    ("Lawyer", "Debate", "Critic"),
    ("Critic", "Query", "Psychologist"),
    ("Psychologist", "Query", "Doctor"),
    ("Doctor", "Query", "Historian"),
    ("Historian", "Query", "Knowlegable Expert"),
    ("Programmer", "Query", "Mathematician"),
    ("Programmer", "Command", "Knowlegable Expert"),
    ("Mathematician", "Command", "Programmer"),
    ("Programmer", "Query", "Economist"),
    ("Economist", "Query", "Psychologist"),
    ("Psychologist", "Query", "Knowlegable Expert"),
    ("Critic", "Query", "Historian"),
    ("Historian", "Query", "Economist"),
    ("Lawyer", "Query", "Knowlegable Expert"),
    ("Doctor", "Query", "Lawyer"),
    ("Mathematician", "Debate", "Doctor"),
    ("Programmer", "Query", "Critic"),
    ("Economist", "Query", "Doctor"),
    ("Lawyer", "Debate", "Critic"),
    ("Psychologist", "Query", "Lawyer"),
    ("Historian", "Query", "Mathematician"),
    ("Programmer", "Query", "Doctor"),
    ("Doctor", "Query", "Psychologist"),
    ("Historian", "Command", "Programmer"),
    ("Critic", "Query", "Economist")
]


@PromptSetRegistry.register("mmlu_pro")
@PromptSetRegistry.register("gaia")
class AgenticPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option qestion answering.
    """
    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at solving problems and analyzing other people's opinions, thinking critically and giving final answers of the task."
    
    def get_role_connection(self):
        return ROLE_CONNECTION
    
    def get_description(self,role):
        return ROLE_DESCRIPTION[role]
    
    @staticmethod
    def get_constraint(role=None):
        return """
            I will ask you a question.
            I will also give you up to 10 answers enumerated as A, B, C, D, E, F, G, H, I and J.
            Only one answer out of the offered options is correct.
            You must choose the correct answer to the question.
            Your response must be one of the letters: A, B, C, D, E, F, G, H, I or J,
            corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you. Please think critically and not just follow the answers of the majority.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C, D, E, F, G, H, I or J)
        """
    
    @staticmethod
    def get_analyze_constraint(role):
        return ROLE_DESCRIPTION[role] if role in ROLE_DESCRIPTION.keys() else ""+ """
I will ask you a question and up to 10 answers enumerated as A through J.
Only one answer out of the offered options is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : A, B, C, D, E, F, G, H, I or J)
"""
    
    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a question.
        I will also give you up to 10 answers enumerated as A, B, C, D, E, F, G, H, I and J.
        Only one answer out of the offered options is correct.
        You must choose the correct answer to the question.
        Your response must be one of the letters: A, B, C, D, E, F, G, H, I or J,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """
    
    @staticmethod
    def get_format():
        return "free-form answer"

    @staticmethod
    def get_answer_prompt(question, role=None):
        return f"""{question}"""

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The first line of your reply must contain only one letter(for example : A, B, C or D)
                """

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
    
    @staticmethod
    def get_decision_few_shot():
        return ""
    
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            return ""
        
        # MMLU-Pro robust extraction logic
        import re
        
        # Level 1: Strong pattern "answer is (X)" or "answer is X"
        match = re.search(r"answer is \(?([A-J])\)?", answer, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        # Level 2: "Answer: X"
        match = re.search(r"Answer:\s*([A-J])", answer, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        # Level 3: Final fallback - look for the LAST single letter A-J
        # This avoids matching "Option A" in the reasoning
        pattern = r"\b([A-J])\b(?!.*\b[A-J]\b)"
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            return match.group(1).upper()
            
        return ""
