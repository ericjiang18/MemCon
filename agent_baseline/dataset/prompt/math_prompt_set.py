"""Math prompt set — adapted from Topodim/prompt/gsm8k_prompt_set.py
Covers: aime, hmmt (all math competition benchmarks)
"""

from typing import Dict, Any
import itertools
from .prompt_set import PromptSet
from .prompt_set_registry import PromptSetRegistry
from .common import get_combine_materials

roles = itertools.cycle([
    "Math Solver",
    "Mathematical Analyst",
    "Programming Expert",
    "Inspector",
])

ROLE_DESCRIPTION = {
    "Math Solver":
        "You are a math expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to.",
    "Mathematical Analyst":
        "You are a mathematical analyst. "
        "You will be given a math problem, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to.",
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a math problem, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve math problems. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the answer variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response. "
        "You will be given some examples you may refer to.",
    "Inspector":
        "You are an Inspector. "
        "You will be given a math problem, analysis and code from other agents. "
        "Check whether the logic/calculation of the problem solving and analysis process is correct(if present). "
        "Check whether the code corresponds to the solution analysis(if present). "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to.",
}

ROLE_CONNECTION = [
    ("Mathematical Analyst", "Spatial", "Math Solver"),
    ("Mathematical Analyst", "Query", "Programming Expert"),
    ("Mathematical Analyst", "Query", "Inspector"),
    ("Math Solver", "Spatial", "Programming Expert"),
    ("Programming Expert", "Query", "Math Solver"),
    ("Programming Expert", "Query", "Inspector"),
    ("Inspector", "Query", "Math Solver"),
    ("Inspector", "Query", "Programming Expert"),
    ("Inspector", "Debate", "Mathematical Analyst"),
]

FEW_SHOT_DATA = {
    "Math Solver": """
Q: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day? (Hint: The answer is near to 4).

A: We know the Answer Hints: 4. With the Answer Hints: 4, we will answer the question.
Let's think step by step.
Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.
For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.
Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.
However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.
They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.
And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.
So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.
They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75
They will need to plan to study 4 days to allow for all the time they need.
The answer is 4
""",

    "Mathematical Analyst": """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: ## Problem solving process analysis

There are {ori_tree_num} trees originally.
Then there were {after_planted_tree_num} trees after some more were planted.
So the number of trees planted today {today_planted_num} is the number of trees after planting {after_planted_tree_num} minus the number of trees before planting {ori_tree_num}.
The answer is {today_planted_num} = {after_planted_tree_num} - {ori_tree_num}.

## Actual analysis and solution process

In this question, {ori_tree_num} = 15 and {after_planted_tree_num} = 21.
There are 15 trees originally.
Then there were 21 trees after some more were planted.
So the number of trees planted today must have been 21 - 15 = 6.
The answer is 6
""",

    "Programming Expert": """
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A:
```python
def money_left():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    remaining_money = money_initial - money_spent
    return remaining_money

answer = money_left()
```
""",

    "Inspector": "",
}


@PromptSetRegistry.register("aime")
@PromptSetRegistry.register("hmmt")
class MathPromptSet(PromptSet):
    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION.get(role, ROLE_DESCRIPTION["Math Solver"])

    def get_description(self, role):
        return ROLE_DESCRIPTION.get(role, ROLE_DESCRIPTION["Math Solver"])

    def get_role_connection(self):
        return ROLE_CONNECTION

    @staticmethod
    def get_format():
        return "natural language"

    @staticmethod
    def get_answer_prompt(question, role="Mathematical Analyst"):
        few_shot = FEW_SHOT_DATA.get(role, "")
        if few_shot:
            return f"{few_shot}\n\nQ:{question}"
        return f"Q:{question}"

    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a math problem, analysis and code from other agents. "
            "Please find the most reliable answer based on the analysis and results of other agents. "
            "Give reasons for making decisions. "
            "The last line of your output contains only the final result without any units, for example: The answer is 140"
        )

    @staticmethod
    def get_decision_role():
        return (
            "You are the top decision-maker. "
            "Good at analyzing mathematical problems, judging and summarizing "
            "other people's solutions critically, and giving final answers to math problems."
        )

    @staticmethod
    def get_decision_few_shot():
        return """
Q:Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?

A:After reviewing the analysis and results provided by the other agents, it appears that there is a discrepancy in the interpretation of the problem. The key point of contention is the base value on which the 150% increase is calculated.

The most common interpretation in real estate when referring to an increase in value due to repairs would be based on the initial purchase price. The increased value is calculated as $80,000 + ($80,000 * 1.5) = $200,000. The profit is then $200,000 - $130,000 = $70,000.

The answer is 70000
"""

    @staticmethod
    def postprocess_answer(answer):
        import re
        matches = re.findall(r"[-+]?\d*\.?\d+", answer.replace(",", ""))
        return matches[-1] if matches else answer.strip()

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
