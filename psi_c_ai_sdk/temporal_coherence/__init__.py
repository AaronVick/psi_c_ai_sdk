"""
Temporal Coherence Module

This module provides tools for maintaining temporal coherence in memory systems, including:
- Temporal pattern detection (recurring themes, access patterns, causal relationships)
- Timeline consistency checking (contradictions, impossible sequences, gaps, anachronisms)
"""

from .temporal_pattern import TemporalPattern, TemporalPatternDetector
from .timeline_consistency import TimelineInconsistency, TimelineConsistencyChecker

__all__ = [
    'TemporalPattern',
    'TemporalPatternDetector',
    'TimelineInconsistency',
    'TimelineConsistencyChecker',
] 