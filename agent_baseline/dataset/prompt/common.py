import re
from typing import Dict, Any


def get_combine_materials(materials: Dict[str, Any], avoid_vague=True) -> str:
    question = materials.get('task', 'No problem provided')

    for key, value in materials.items():
        if "No useful information from WebSearch" in str(value):
            continue
        if isinstance(value, list):
            value = "\n".join(value)
        if not (isinstance(value, str) and isinstance(key, str)):
            continue
        value = value.strip("\n").strip()
        if key != 'task' and value:
            question += f"\n\nReference information for {key}:" + \
                        "\n----------------------------------------------\n" + \
                        f"{value}" + \
                        "\n----------------------------------------------\n\n"

    if avoid_vague:
        question += "\nProvide a specific answer. For questions with known answers, ensure to provide accurate and factual responses. " + \
                    "Avoid vague responses or statements like 'unable to...' that don't contribute to a definitive answer."

    return question
