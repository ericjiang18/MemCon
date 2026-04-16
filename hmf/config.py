"""
Hierarchical Memory Framework — Configuration.

Central config for all HMF components: memory substrates,
MPC controller, cost weights, and integration settings.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CacheMemoryConfig:
    max_entries: int = 256
    ttl_seconds: float = 600.0
    max_tokens_per_entry: int = 2048
    semantic_threshold: float = 0.85
    lru_enabled: bool = True


@dataclass
class RetrievalMemoryConfig:
    collection_name: str = "hmf_retrieval"
    max_documents: int = 2000
    top_k: int = 5
    relevance_threshold: float = 0.3
    decay_factor: float = 0.95
    importance_weight: float = 0.4
    recency_weight: float = 0.3
    frequency_weight: float = 0.3


@dataclass
class SkillMemoryConfig:
    max_skills: int = 100
    min_trajectory_success_rate: float = 0.5
    consolidation_threshold: int = 3
    skill_decay_rate: float = 0.01
    evolution_enabled: bool = True


@dataclass
class MPCConfig:
    horizon: int = 3
    num_candidates: int = 5
    alpha_token: float = 0.3
    beta_latency: float = 0.2
    gamma_uncertainty: float = 0.2
    delta_utility: float = 0.3
    token_budget: int = 8000
    latency_budget_ms: float = 30000.0
    safety_threshold: float = 0.1
    replan_every: int = 1
    use_llm_scoring: bool = True
    fallback_to_heuristic: bool = True


@dataclass
class HMFConfig:
    cache: CacheMemoryConfig = field(default_factory=CacheMemoryConfig)
    retrieval: RetrievalMemoryConfig = field(default_factory=RetrievalMemoryConfig)
    skill: SkillMemoryConfig = field(default_factory=SkillMemoryConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)

    working_dir: str = ".hmf_db"
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_base: Optional[str] = None
    api_key: Optional[str] = None

    verbose: bool = True
