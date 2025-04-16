"""
Justification module for the ΨC-AI SDK.

This module provides tools for explaining and tracing the lineage of memories and beliefs:
1. Explanation generation for beliefs and decisions
2. Ancestry chain visualization showing how memories evolved
3. Source tracing to track the origins of knowledge
"""

from psi_c_ai_sdk.justification.justification_engine import (
    JustificationEngine,
    ExplanationFormat,
    BeliefExplanation,
    AncestryNode,
    SourceReference
)

__all__ = [
    'JustificationEngine',
    'ExplanationFormat',
    'BeliefExplanation',
    'AncestryNode',
    'SourceReference'
]
