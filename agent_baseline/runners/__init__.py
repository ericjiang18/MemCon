"""Framework runner factory – lazy imports to avoid pulling every framework."""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from .base_runner import BaseRunner


def get_runner(name: str, *, base_url: str, api_key: str, model: str) -> BaseRunner:
    if name == "agent-framework":
        from .agent_framework_runner import AgentFrameworkRunner
        return AgentFrameworkRunner(base_url=base_url, api_key=api_key, model=model)
    elif name == "autogen":
        from .autogen_runner import AutoGenRunner
        return AutoGenRunner(base_url=base_url, api_key=api_key, model=model)
    elif name == "langgraph":
        from .langgraph_runner import LangGraphRunner
        return LangGraphRunner(base_url=base_url, api_key=api_key, model=model)
    elif name == "lobster":
        from .lobster_runner import LobsterRunner
        return LobsterRunner(base_url=base_url, api_key=api_key, model=model)
    elif name == "langgraph_hmf":
        from hmf.integrations.langgraph_hmf import LangGraphHMFRunner
        return LangGraphHMFRunner(base_url=base_url, api_key=api_key, model=model)
    elif name == "agent_framework_hmf":
        from hmf.integrations.agent_framework_hmf import AgentFrameworkHMFRunner
        return AgentFrameworkHMFRunner(base_url=base_url, api_key=api_key, model=model)
    elif name == "lobster_hmf":
        from hmf.integrations.lobster_hmf import LobsterHMFRunner
        return LobsterHMFRunner(base_url=base_url, api_key=api_key, model=model)
    elif name == "langgraph_gmemory":
        from hmf.integrations.langgraph_gmemory import LangGraphGMemoryRunner
        return LangGraphGMemoryRunner(base_url=base_url, api_key=api_key, model=model)
    else:
        raise ValueError(
            f"Unknown framework: {name}. "
            f"Available: agent-framework, autogen, langgraph, lobster, "
            f"langgraph_hmf, agent_framework_hmf, lobster_hmf, langgraph_gmemory"
        )
