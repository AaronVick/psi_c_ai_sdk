"""
ΨC-AI SDK: A cognitive framework for building self-reflective AI systems.

Core modules:
- memory: Memory storage and management
- embedding: Vector embedding generation and caching
- coherence: Coherence measurement between memories
- contradiction: Contradiction detection in memory sets
- reflection: Reflection cycles for self-improvement
- schema: Schema graph building and evolution
- beliefs: Belief revision and trust calibration
- entropy: Entropy measurement and management
"""

__version__ = "0.1.0"

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.embedding.embedding import EmbeddingEngine
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer
from psi_c_ai_sdk.contradiction.contradiction import ContradictionDetector
from psi_c_ai_sdk.beliefs.toolkit import BeliefToolkit
from psi_c_ai_sdk.beliefs.contradiction import ContradictionType
from psi_c_ai_sdk.beliefs.revision import (
    BeliefRevisionSystem, TrustLevel, DecisionOutcome, DecisionStrategy
)
from psi_c_ai_sdk.schema.toolkit import SchemaToolkit
from psi_c_ai_sdk.schema.schema import SchemaGraph, SchemaNode
from psi_c_ai_sdk.schema.mutation import MutationType

from psi_c_ai_sdk import memory
from psi_c_ai_sdk import embedding
from psi_c_ai_sdk import coherence
from psi_c_ai_sdk import contradiction
from psi_c_ai_sdk import schema
from psi_c_ai_sdk import reflection
from psi_c_ai_sdk import psi_c
from psi_c_ai_sdk import runtime
from psi_c_ai_sdk import beliefs
from psi_c_ai_sdk import entropy

__all__ = [
    # Modules
    "memory",
    "embedding", 
    "coherence",
    "contradiction",
    "schema",
    "reflection",
    "psi_c",
    "runtime",
    "beliefs",
    "entropy",
    
    # Core classes
    "Memory",
    "MemoryStore",
    "EmbeddingEngine",
    "CoherenceScorer",
    "ContradictionDetector",
    "BeliefToolkit",
    "BeliefRevisionSystem",
    "TrustLevel",
    "DecisionOutcome",
    "DecisionStrategy",
    "ContradictionType",
    "SchemaToolkit",
    "SchemaGraph",
    "SchemaNode",
    "MutationType"
] 