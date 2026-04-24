from typing import Type

from .reasoning import ReasoningBase, ReasoningIO
from .memory import *
from .memory.mas_memory.gmemory_plus import GMemoryPlus
from .memory.mas_memory.skill_memory import SkillMemory

def _get_hmf_memory(static=False):
    """Lazy import to avoid pulling hmf deps when not needed."""
    import sys, os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if static:
        from hmf.integrations.mas_hmf_static import HMFStaticMemory
        return HMFStaticMemory
    from hmf.integrations.mas_hmf_memory import HMFMemory
    return HMFMemory

def module_map(
    reasoning: str, mas_memory: str = None
) -> tuple[Type[ReasoningBase], Type[MASMemoryBase]]:
    
    reasoning_map = {
        'io': ReasoningIO,
    }
    mas_memory_map = {
        'empty': MASMemoryBase,
        'g-memory': GMemory,
        'g-memory-plus': GMemoryPlus,
        'gmemory-plus': GMemoryPlus,
        'gm+': GMemoryPlus,
        'skill-memory': SkillMemory,
        'skill-rl': SkillMemory,
        'srl': SkillMemory,
        # backward compat
        'goal-rl': SkillMemory,
        'goal-rl-memory': SkillMemory,
        'grl': SkillMemory,
        # HMF (Hierarchical Memory Framework with MPC)
        'hmf': None,
        'hmf-mpc': None,
        # HMF-Static (same layers, no MPC — ablation)
        'hmf-static': None,
        # HMF-Learned (same layers, learned bandit controller)
        'hmf-learned': None,
        # HMF v2 (graph-enhanced + scored insights + cache + skill)
        'hmf-v2': None,
        # G-Memory Enhanced (G-Memory + retrieval cache + success patterns)
        'gmemory-enhanced': None,
        'gm-e': None,
        # AMC (Anticipatory Memory Control — our main contribution)
        'amc': None,
        # MemCon (Memory as Controlled Process — paper method)
        'memcon': None,
        # External baselines
        'g-memory-orig': None,
        'gm-orig': None,
        'latentmem': None,
        # Additional baselines
        'metagpt': None,
        'voyager': None,
        'generative': None,
        'chatdev': None,
        'memorybank': None,
        'oagent': None,
        'experiencebank': None,
    }

    if reasoning not in reasoning_map:
        raise ValueError(f"Invalid reasoning type '{reasoning}'. Allowed values: {list(reasoning_map.keys())}")

    if mas_memory is not None and mas_memory not in mas_memory_map:
        raise ValueError(f"Invalid MAS memory type '{mas_memory}'. Allowed values: {list(mas_memory_map.keys())}")

    # Lazy-load HMF to avoid import overhead for non-HMF runs
    if mas_memory in ('hmf', 'hmf-mpc'):
        mas_memory_map[mas_memory] = _get_hmf_memory()
    elif mas_memory == 'hmf-static':
        mas_memory_map[mas_memory] = _get_hmf_memory(static=True)
    elif mas_memory == 'hmf-learned':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_hmf_learned import HMFLearnedMemory
        mas_memory_map[mas_memory] = HMFLearnedMemory
    elif mas_memory == 'hmf-v2':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_hmf_v2 import HMFv2Memory
        mas_memory_map[mas_memory] = HMFv2Memory
    elif mas_memory in ('gmemory-enhanced', 'gm-e'):
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_gmemory_enhanced import GMemoryEnhanced
        mas_memory_map[mas_memory] = GMemoryEnhanced
    elif mas_memory == 'amc':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_amc import AMCMemory
        mas_memory_map[mas_memory] = AMCMemory
    elif mas_memory == 'memcon':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_memcon import MemConMemory
        mas_memory_map[mas_memory] = MemConMemory
    elif mas_memory in ('g-memory-orig', 'gm-orig'):
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_gmemory_orig import GMemoryOrig
        mas_memory_map[mas_memory] = GMemoryOrig
    elif mas_memory == 'latentmem':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_latentmem import LatentMemMemory
        mas_memory_map[mas_memory] = LatentMemMemory
    elif mas_memory == 'metagpt':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_metagpt import MetaGPTMemory
        mas_memory_map[mas_memory] = MetaGPTMemory
    elif mas_memory == 'voyager':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_voyager import VoyagerMemory
        mas_memory_map[mas_memory] = VoyagerMemory
    elif mas_memory == 'generative':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_generative import GenerativeMemory
        mas_memory_map[mas_memory] = GenerativeMemory
    elif mas_memory == 'chatdev':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_chatdev import ChatDevMemory
        mas_memory_map[mas_memory] = ChatDevMemory
    elif mas_memory == 'memorybank':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_memorybank import MemoryBankMemory
        mas_memory_map[mas_memory] = MemoryBankMemory
    elif mas_memory == 'oagent':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_oagent import OAgentMemory
        mas_memory_map[mas_memory] = OAgentMemory
    elif mas_memory == 'experiencebank':
        import sys as _sys, os as _os
        _rr = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        if _rr not in _sys.path:
            _sys.path.insert(0, _rr)
        from hmf.integrations.mas_experiencebank import ExperienceBankMemory
        mas_memory_map[mas_memory] = ExperienceBankMemory

    return (
        reasoning_map[reasoning],
        mas_memory_map.get(mas_memory, None)
    )
    
    