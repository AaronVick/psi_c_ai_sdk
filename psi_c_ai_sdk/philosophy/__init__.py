"""
Philosophy module for ΨC-AI SDK

This module provides components for philosophical reasoning, boundaries, and ethical
decision-making in cognitive AI systems. It implements core philosophical principles,
contradiction management, and belief systems.

Components:
- Core Philosophy System: Defines axioms and principles that guide system behavior
- Contradiction Warning System: Detects and manages logical and philosophical contradictions
- Philosophical Boundary Enforcement: Prevents actions that violate core principles

This module is central to maintaining the philosophical integrity and ethical
behavior of ΨC-AI systems.
"""

from psi_c_ai_sdk.philosophy.core_philosophy import (
    PhilosophicalDomain,
    PrincipleType,
    PhilosophicalPrinciple,
    CorePhilosophySystem
)

from psi_c_ai_sdk.philosophy.contradiction_warning import (
    ContradictionType,
    ContradictionSeverity,
    ContradictionWarning,
    ContradictionWarningSystem
)

from psi_c_ai_sdk.philosophy.boundary_enforcement import (
    BoundaryAction,
    EnforcementDomain,
    PhilosophicalBoundary,
    BoundaryEnforcementSystem,
    BoundaryViolation,
    check_operation,
    enforce_operation,
    get_boundary_enforcement_system
)

__all__ = [
    'PhilosophicalDomain',
    'PrincipleType',
    'PhilosophicalPrinciple',
    'CorePhilosophySystem',
    'ContradictionType',
    'ContradictionSeverity',
    'ContradictionWarning',
    'ContradictionWarningSystem',
    'BoundaryAction',
    'EnforcementDomain',
    'PhilosophicalBoundary',
    'BoundaryEnforcementSystem',
    'BoundaryViolation',
    'check_operation',
    'enforce_operation',
    'get_boundary_enforcement_system'
] 