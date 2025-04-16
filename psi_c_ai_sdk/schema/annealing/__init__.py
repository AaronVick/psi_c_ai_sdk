"""
Schema Annealing Module for the ΨC-AI SDK.

This module provides temperature-based control for schema mutations, driving
the schema graph toward convergence over time. Key features include:

1. Temperature-based mutation control with exponential cooling
2. Convergence tracking for measuring schema stability
3. Stability metrics for evaluating schema maturity

Schema annealing helps the system settle into reliable cognitive structures
rather than evolving indefinitely, mimicking the way human knowledge
structures stabilize over time.
"""

from psi_c_ai_sdk.schema.annealing.annealing import (
    SchemaAnnealer,
    AnnealingSchedule,
    ConvergenceTracker,
    StabilityMetrics
)

__all__ = [
    'SchemaAnnealer',
    'AnnealingSchedule',
    'ConvergenceTracker',
    'StabilityMetrics'
] 