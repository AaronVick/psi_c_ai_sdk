"""
Belief management systems for handling contradictions, trust, and revision.

This package provides components for managing beliefs, including contradiction
detection, belief revision, and trust calibration mechanisms.
"""

from psi_c_ai_sdk.beliefs.contradiction import ContradictionDetector, ContradictionType
from psi_c_ai_sdk.beliefs.revision import (
    BeliefRevisionSystem,
    DecisionStrategy,
    DecisionOutcome,
    TrustLevel
)

__all__ = [
    'ContradictionDetector',
    'ContradictionType',
    'BeliefRevisionSystem',
    'DecisionStrategy',
    'DecisionOutcome',
    'TrustLevel'
] 