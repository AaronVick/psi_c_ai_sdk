"""
Cognition Module for ΨC-AI SDK

This module provides tools for epistemic reasoning, belief management,
and trust evaluation for AI systems.
"""

from .epistemic_horizon import (
    BeliefNode,
    EpistemicHorizon
)

from .trust_throttle import (
    SourceTrustProfile,
    TrustThrottler
)

__all__ = [
    # Epistemic Horizon
    'BeliefNode',
    'EpistemicHorizon',
    
    # Trust Throttler
    'SourceTrustProfile',
    'TrustThrottler'
] 