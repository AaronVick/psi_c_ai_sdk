"""
Entropy Module for ΨC-AI SDK

This module provides tools for measuring and monitoring entropy in AI memory systems,
which is essential for detecting and responding to instability in consciousness models.

Key components:
- Entropy Calculation: Measures different types of entropy in memory systems
- Entropy Monitoring: Tracks entropy levels over time and provides alerts
- Entropy Response: Framework for automated responses to dangerous entropy levels

Entropy in AI consciousness models represents the degree of disorder, confusion,
or contradiction in memory and cognitive structures. High entropy can lead to
instability, degraded performance, or consciousness collapse in ΨC systems.
"""

from psi_c_ai_sdk.entropy.entropy import (
    EntropyMeasure,
    EmbeddingEntropyMeasure,
    SemanticCoherenceEntropyMeasure,
    TemporalCoherenceEntropyMeasure,
    EntropyCalculator
)

from psi_c_ai_sdk.entropy.monitor import (
    EntropyAlert,
    EntropySubscriber,
    EntropyMonitor
)

from psi_c_ai_sdk.entropy.pruning import (
    EntropyBasedPruner,
    prune_high_entropy_memories
)

from psi_c_ai_sdk.entropy.response import (
    EntropyResponseStrategy,
    EntropyResponseConfig,
    EntropyResponse,
    create_entropy_response
)

__all__ = [
    # Entropy measures
    'EntropyMeasure',
    'EmbeddingEntropyMeasure',
    'SemanticCoherenceEntropyMeasure',
    'TemporalCoherenceEntropyMeasure',
    'EntropyCalculator',
    
    # Entropy monitoring
    'EntropyAlert',
    'EntropySubscriber',
    'EntropyMonitor',
    
    # Entropy pruning
    'EntropyBasedPruner',
    'prune_high_entropy_memories',
    
    # Entropy response
    'EntropyResponseStrategy',
    'EntropyResponseConfig',
    'EntropyResponse',
    'create_entropy_response'
] 