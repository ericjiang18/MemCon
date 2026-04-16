"""
Memory as a Controlled Process (MemCon)

A general framework that models agent memory operations as a Markov Decision
Process and learns an online policy to control WHEN, WHAT, and HOW MUCH
to retrieve, encode, consolidate, and forget.

Can wrap ANY underlying memory backend (G-Memory, flat vector store, etc.).
"""

from .memory_mdp import MemoryMDP, MemoryState, MemoryActionSpace
from .policy import MemoryPolicy
from .wrapper import MemConWrapper
