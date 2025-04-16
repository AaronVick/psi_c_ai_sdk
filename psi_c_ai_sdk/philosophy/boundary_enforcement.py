#!/usr/bin/env python3
"""
Philosophical Boundary Enforcement for ΨC-AI SDK

This module implements boundary enforcement mechanisms that prevent actions and
beliefs that would violate the system's core philosophical principles. It serves
as a safeguard for maintaining the system's philosophical integrity, identity
stability, and coherence.

The boundary enforcement system acts as a gatekeeper that:
1. Validates actions against core axioms and principles
2. Intercepts potential violations before they occur
3. Provides feedback on boundary conditions
4. Maintains logs of enforcement actions
"""

import logging
import time
import uuid
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable

from psi_c_ai_sdk.philosophy.core_philosophy import (
    CorePhilosophy, AxiomType, ViolationSeverity, get_core_philosophy
)

# Configure logging
logger = logging.getLogger(__name__)


class BoundaryAction(Enum):
    """Types of actions that can be taken when boundaries are violated."""
    
    ALLOW = auto()         # Allow the action to proceed (no violation or minor)
    WARN = auto()          # Allow but log a warning (minor violation)
    MODIFY = auto()        # Allow with modifications to comply with boundaries
    REJECT = auto()        # Reject the action entirely (serious violation)
    QUARANTINE = auto()    # Isolate and mark for review (severe violation)


class EnforcementDomain(Enum):
    """Domains where boundary enforcement applies."""
    
    MEMORY = auto()        # Memory operations (add, modify, delete)
    SCHEMA = auto()        # Schema operations (mutations, merges)
    REFLECTION = auto()    # Reflection processes
    BELIEF = auto()        # Belief system operations
    EXTERNAL = auto()      # Interactions with external systems
    IDENTITY = auto()      # Identity-related operations


class EnforcementRecord:
    """Record of a boundary enforcement action."""
    
    def __init__(
        self,
        operation_id: str,
        domain: EnforcementDomain,
        action_taken: BoundaryAction,
        violated_axioms: List[str],
        context: Dict[str, Any],
        timestamp: float = None
    ):
        """
        Initialize an enforcement record.
        
        Args:
            operation_id: Unique identifier for the operation
            domain: Domain where enforcement occurred
            action_taken: Action taken by the enforcement system
            violated_axioms: List of axioms that were violated
            context: Additional context about the enforcement
            timestamp: When the enforcement occurred (default: now)
        """
        self.id = str(uuid.uuid4())
        self.operation_id = operation_id
        self.domain = domain
        self.action_taken = action_taken
        self.violated_axioms = violated_axioms
        self.context = context
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the record to a dictionary.
        
        Returns:
            Dictionary representation of the record
        """
        return {
            "id": self.id,
            "operation_id": self.operation_id,
            "domain": self.domain.name,
            "action_taken": self.action_taken.name,
            "violated_axioms": self.violated_axioms,
            "context": self.context,
            "timestamp": self.timestamp
        }


class BoundaryViolation:
    """Representation of a philosophical boundary violation."""
    
    def __init__(
        self,
        axiom_name: str,
        severity: ViolationSeverity,
        description: str,
        score: float,
        threshold: float,
        context: Dict[str, Any] = None
    ):
        """
        Initialize a boundary violation.
        
        Args:
            axiom_name: Name of the violated axiom
            severity: Severity of the violation
            description: Description of the violation
            score: Evaluation score (lower = worse violation)
            threshold: Threshold that was violated
            context: Additional context
        """
        self.axiom_name = axiom_name
        self.severity = severity
        self.description = description
        self.score = score
        self.threshold = threshold
        self.context = context or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the violation to a dictionary.
        
        Returns:
            Dictionary representation of the violation
        """
        return {
            "axiom_name": self.axiom_name,
            "severity": self.severity.name,
            "description": self.description,
            "score": self.score,
            "threshold": self.threshold,
            "context": self.context,
            "timestamp": self.timestamp
        }


class PhilosophicalBoundary:
    """
    A philosophical boundary that enforces core principles.
    
    Each boundary is associated with a specific domain and defines
    conditions under which operations in that domain are permitted.
    """
    
    def __init__(
        self,
        name: str,
        domain: EnforcementDomain,
        axiom_types: List[AxiomType],
        description: str,
        enforcement_fn: Callable[[Dict[str, Any]], Tuple[bool, float]] = None
    ):
        """
        Initialize a philosophical boundary.
        
        Args:
            name: Boundary name
            domain: Domain where the boundary applies
            axiom_types: Types of axioms this boundary enforces
            description: Description of the boundary
            enforcement_fn: Function to evaluate boundary compliance
        """
        self.name = name
        self.domain = domain
        self.axiom_types = axiom_types
        self.description = description
        self.enforcement_fn = enforcement_fn
        self.created_at = time.time()
        self.stats = {
            "total_checks": 0,
            "violations": 0,
            "allows": 0,
            "warns": 0,
            "modifies": 0,
            "rejects": 0,
            "quarantines": 0
        }
    
    def check(self, context: Dict[str, Any]) -> Tuple[BoundaryAction, List[BoundaryViolation]]:
        """
        Check if an operation violates this boundary.
        
        Args:
            context: Context for the operation
            
        Returns:
            Tuple of (action to take, list of violations)
        """
        self.stats["total_checks"] += 1
        
        # Get core philosophy
        philosophy = get_core_philosophy()
        
        # Check violations against relevant axioms
        violations = []
        for name, axiom in philosophy.axioms.items():
            # Skip axioms not relevant to this boundary
            if axiom.axiom_type not in self.axiom_types:
                continue
            
            # Evaluate the axiom
            is_violated, score = axiom.evaluate(context)
            
            if is_violated:
                violations.append(BoundaryViolation(
                    axiom_name=name,
                    severity=axiom.severity,
                    description=axiom.description,
                    score=score,
                    threshold=axiom.threshold,
                    context=context
                ))
        
        # If custom enforcement function exists, use it
        if self.enforcement_fn:
            is_violated, score = self.enforcement_fn(context)
            if is_violated:
                violations.append(BoundaryViolation(
                    axiom_name=f"custom:{self.name}",
                    severity=ViolationSeverity.MEDIUM,  # Default for custom
                    description=f"Custom boundary violation: {self.description}",
                    score=score,
                    threshold=0.5,  # Default threshold
                    context=context
                ))
        
        # Determine action based on violations
        action = self._determine_action(violations)
        
        # Update stats
        if action == BoundaryAction.ALLOW:
            self.stats["allows"] += 1
        elif action == BoundaryAction.WARN:
            self.stats["warns"] += 1
        elif action == BoundaryAction.MODIFY:
            self.stats["modifies"] += 1
        elif action == BoundaryAction.REJECT:
            self.stats["rejects"] += 1
        elif action == BoundaryAction.QUARANTINE:
            self.stats["quarantines"] += 1
        
        if violations:
            self.stats["violations"] += 1
        
        return action, violations
    
    def _determine_action(self, violations: List[BoundaryViolation]) -> BoundaryAction:
        """
        Determine what action to take based on violations.
        
        Args:
            violations: List of boundary violations
            
        Returns:
            Action to take
        """
        if not violations:
            return BoundaryAction.ALLOW
        
        # Find the most severe violation
        severities = [v.severity for v in violations]
        max_severity = max(severities, key=lambda s: s.value)
        
        # Determine action based on severity
        if max_severity == ViolationSeverity.TERMINAL:
            return BoundaryAction.QUARANTINE
        elif max_severity == ViolationSeverity.HIGH:
            return BoundaryAction.REJECT
        elif max_severity == ViolationSeverity.MEDIUM:
            # For medium severity, check if we have multiple violations
            if len(violations) > 1:
                return BoundaryAction.REJECT
            else:
                return BoundaryAction.MODIFY
        else:  # LOW
            return BoundaryAction.WARN
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the boundary to a dictionary.
        
        Returns:
            Dictionary representation of the boundary
        """
        return {
            "name": self.name,
            "domain": self.domain.name,
            "axiom_types": [at.name for at in self.axiom_types],
            "description": self.description,
            "created_at": self.created_at,
            "stats": self.stats
        }


class BoundaryEnforcementSystem:
    """
    System for enforcing philosophical boundaries.
    
    This system maintains a set of boundaries for different domains
    and provides mechanisms to check, enforce, and report on boundary
    violations.
    """
    
    def __init__(self):
        """Initialize the boundary enforcement system."""
        self.boundaries: Dict[str, PhilosophicalBoundary] = {}
        self.enforcement_records: List[EnforcementRecord] = []
        self.quarantined_operations: Dict[str, Dict[str, Any]] = {}
        self.max_record_history = 1000
        self.created_at = time.time()
        
        # Initialize default boundaries
        self._initialize_default_boundaries()
    
    def _initialize_default_boundaries(self):
        """Initialize default philosophical boundaries."""
        # Schema Identity Boundary
        self.add_boundary(PhilosophicalBoundary(
            name="schema_identity_boundary",
            domain=EnforcementDomain.SCHEMA,
            axiom_types=[AxiomType.IDENTITY, AxiomType.STABILITY],
            description="Prevents schema changes that would compromise system identity"
        ))
        
        # Memory Coherence Boundary
        self.add_boundary(PhilosophicalBoundary(
            name="memory_coherence_boundary",
            domain=EnforcementDomain.MEMORY,
            axiom_types=[AxiomType.COHERENCE, AxiomType.CONSISTENCY],
            description="Ensures memory operations maintain or improve coherence"
        ))
        
        # Reflection Causality Boundary
        self.add_boundary(PhilosophicalBoundary(
            name="reflection_causality_boundary",
            domain=EnforcementDomain.REFLECTION,
            axiom_types=[AxiomType.CAUSALITY],
            description="Ensures all reflections maintain causal traceability"
        ))
        
        # Belief Integrity Boundary
        self.add_boundary(PhilosophicalBoundary(
            name="belief_integrity_boundary",
            domain=EnforcementDomain.BELIEF,
            axiom_types=[AxiomType.CONSISTENCY, AxiomType.IDENTITY],
            description="Maintains integrity of the belief system"
        ))
        
        # Complexity Management Boundary
        self.add_boundary(PhilosophicalBoundary(
            name="complexity_management_boundary",
            domain=EnforcementDomain.SCHEMA,
            axiom_types=[AxiomType.COMPLEXITY],
            description="Prevents excessive system complexity"
        ))
        
        # External Influence Boundary
        self.add_boundary(PhilosophicalBoundary(
            name="external_influence_boundary",
            domain=EnforcementDomain.EXTERNAL,
            axiom_types=[AxiomType.STABILITY, AxiomType.IDENTITY],
            description="Limits external influence that could compromise identity"
        ))
    
    def add_boundary(self, boundary: PhilosophicalBoundary) -> None:
        """
        Add a boundary to the enforcement system.
        
        Args:
            boundary: The boundary to add
        """
        if boundary.name in self.boundaries:
            logger.warning(f"Overwriting existing boundary: {boundary.name}")
        
        self.boundaries[boundary.name] = boundary
        logger.info(f"Added philosophical boundary: {boundary.name}")
    
    def check_operation(
        self,
        operation_id: str,
        domain: EnforcementDomain,
        context: Dict[str, Any]
    ) -> Tuple[BoundaryAction, List[BoundaryViolation]]:
        """
        Check if an operation violates any boundaries.
        
        Args:
            operation_id: Unique identifier for the operation
            domain: Domain of the operation
            context: Context for the operation
            
        Returns:
            Tuple of (action to take, list of violations)
        """
        # Get relevant boundaries for this domain
        relevant_boundaries = [
            b for b in self.boundaries.values()
            if b.domain == domain
        ]
        
        if not relevant_boundaries:
            logger.debug(f"No boundaries defined for domain: {domain.name}")
            return BoundaryAction.ALLOW, []
        
        # Check each boundary
        all_violations = []
        actions = []
        
        for boundary in relevant_boundaries:
            action, violations = boundary.check(context)
            actions.append(action)
            all_violations.extend(violations)
        
        # Take the most restrictive action
        if BoundaryAction.QUARANTINE in actions:
            final_action = BoundaryAction.QUARANTINE
        elif BoundaryAction.REJECT in actions:
            final_action = BoundaryAction.REJECT
        elif BoundaryAction.MODIFY in actions:
            final_action = BoundaryAction.MODIFY
        elif BoundaryAction.WARN in actions:
            final_action = BoundaryAction.WARN
        else:
            final_action = BoundaryAction.ALLOW
        
        # Record the enforcement
        if all_violations:
            self._record_enforcement(
                operation_id=operation_id,
                domain=domain,
                action_taken=final_action,
                violated_axioms=[v.axiom_name for v in all_violations],
                context=context
            )
            
            # If quarantined, store the operation
            if final_action == BoundaryAction.QUARANTINE:
                self.quarantined_operations[operation_id] = {
                    "domain": domain.name,
                    "context": context,
                    "violations": [v.to_dict() for v in all_violations],
                    "timestamp": time.time()
                }
        
        return final_action, all_violations
    
    def enforce_operation(
        self,
        operation_id: str,
        domain: EnforcementDomain,
        context: Dict[str, Any],
        operation_fn: Callable
    ) -> Tuple[bool, Any, List[BoundaryViolation]]:
        """
        Enforce boundaries on an operation, executing it only if allowed.
        
        Args:
            operation_id: Unique identifier for the operation
            domain: Domain of the operation
            context: Context for the operation
            operation_fn: Function to execute if operation is allowed
            
        Returns:
            Tuple of (success, result, violations)
        """
        # Check operation against boundaries
        action, violations = self.check_operation(
            operation_id=operation_id,
            domain=domain,
            context=context
        )
        
        # Handle based on action
        if action == BoundaryAction.ALLOW:
            # Execute operation
            result = operation_fn()
            return True, result, []
        elif action == BoundaryAction.WARN:
            # Log warning but execute
            logger.warning(
                f"Operation {operation_id} proceeded despite boundary warnings: "
                f"{', '.join(v.axiom_name for v in violations)}"
            )
            result = operation_fn()
            return True, result, violations
        elif action == BoundaryAction.MODIFY:
            # TODO: Implement modification logic
            # For now, just execute with warning
            logger.warning(
                f"Operation {operation_id} requires modification to comply with boundaries: "
                f"{', '.join(v.axiom_name for v in violations)}"
            )
            result = operation_fn()
            return True, result, violations
        elif action == BoundaryAction.REJECT:
            # Reject operation
            logger.error(
                f"Operation {operation_id} rejected due to boundary violations: "
                f"{', '.join(v.axiom_name for v in violations)}"
            )
            return False, None, violations
        elif action == BoundaryAction.QUARANTINE:
            # Quarantine operation
            logger.critical(
                f"Operation {operation_id} quarantined due to severe boundary violations: "
                f"{', '.join(v.axiom_name for v in violations)}"
            )
            return False, None, violations
        
        # Shouldn't reach here
        return False, None, violations
    
    def _record_enforcement(
        self,
        operation_id: str,
        domain: EnforcementDomain,
        action_taken: BoundaryAction,
        violated_axioms: List[str],
        context: Dict[str, Any]
    ) -> None:
        """
        Record an enforcement action.
        
        Args:
            operation_id: Unique identifier for the operation
            domain: Domain of the operation
            action_taken: Action taken by the enforcement system
            violated_axioms: List of axioms that were violated
            context: Additional context about the enforcement
        """
        record = EnforcementRecord(
            operation_id=operation_id,
            domain=domain,
            action_taken=action_taken,
            violated_axioms=violated_axioms,
            context=context
        )
        
        self.enforcement_records.append(record)
        
        # Prune history if needed
        if len(self.enforcement_records) > self.max_record_history:
            self.enforcement_records = self.enforcement_records[-self.max_record_history:]
        
        # Log enforcement action
        if action_taken in [BoundaryAction.REJECT, BoundaryAction.QUARANTINE]:
            logger.warning(
                f"Enforcement action {action_taken.name} taken for operation {operation_id} "
                f"in domain {domain.name}. Violated axioms: {', '.join(violated_axioms)}"
            )
        else:
            logger.info(
                f"Enforcement action {action_taken.name} taken for operation {operation_id} "
                f"in domain {domain.name}"
            )
    
    def get_enforcement_history(
        self,
        limit: Optional[int] = None,
        domain: Optional[EnforcementDomain] = None,
        action: Optional[BoundaryAction] = None
    ) -> List[Dict[str, Any]]:
        """
        Get history of enforcement actions.
        
        Args:
            limit: Maximum number of records to return, None for all
            domain: Filter by domain, None for all domains
            action: Filter by action taken, None for all actions
            
        Returns:
            List of enforcement records
        """
        # Apply filters
        filtered_records = self.enforcement_records
        
        if domain:
            filtered_records = [r for r in filtered_records if r.domain == domain]
        
        if action:
            filtered_records = [r for r in filtered_records if r.action_taken == action]
        
        # Sort by timestamp (newest first)
        sorted_records = sorted(filtered_records, key=lambda r: r.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            sorted_records = sorted_records[:limit]
        
        # Convert to dictionaries
        return [r.to_dict() for r in sorted_records]
    
    def get_quarantined_operations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all quarantined operations.
        
        Returns:
            Dictionary of quarantined operations
        """
        return self.quarantined_operations
    
    def get_boundaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all defined boundaries.
        
        Returns:
            Dictionary of boundaries
        """
        return {name: boundary.to_dict() for name, boundary in self.boundaries.items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the enforcement system.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "boundary_count": len(self.boundaries),
            "enforcement_record_count": len(self.enforcement_records),
            "quarantined_operation_count": len(self.quarantined_operations),
            "created_at": self.created_at,
            "domains": {},
            "actions": {
                "ALLOW": 0,
                "WARN": 0,
                "MODIFY": 0,
                "REJECT": 0,
                "QUARANTINE": 0
            }
        }
        
        # Count by domain
        for record in self.enforcement_records:
            domain_name = record.domain.name
            if domain_name not in stats["domains"]:
                stats["domains"][domain_name] = 0
            stats["domains"][domain_name] += 1
            
            # Count by action
            action_name = record.action_taken.name
            stats["actions"][action_name] += 1
        
        # Add per-boundary stats
        stats["boundaries"] = {
            name: boundary.stats
            for name, boundary in self.boundaries.items()
        }
        
        return stats


# Global instance
_ENFORCEMENT_SYSTEM = None


def get_boundary_enforcement_system() -> BoundaryEnforcementSystem:
    """
    Get the singleton boundary enforcement system instance.
    
    Returns:
        BoundaryEnforcementSystem instance
    """
    global _ENFORCEMENT_SYSTEM
    if _ENFORCEMENT_SYSTEM is None:
        _ENFORCEMENT_SYSTEM = BoundaryEnforcementSystem()
    
    return _ENFORCEMENT_SYSTEM


# Convenience functions

def check_operation(
    operation_id: str,
    domain: EnforcementDomain,
    context: Dict[str, Any]
) -> Tuple[BoundaryAction, List[BoundaryViolation]]:
    """
    Check if an operation violates any boundaries.
    
    Args:
        operation_id: Unique identifier for the operation
        domain: Domain of the operation
        context: Context for the operation
        
    Returns:
        Tuple of (action to take, list of violations)
    """
    return get_boundary_enforcement_system().check_operation(
        operation_id=operation_id,
        domain=domain,
        context=context
    )


def enforce_operation(
    operation_id: str,
    domain: EnforcementDomain,
    context: Dict[str, Any],
    operation_fn: Callable
) -> Tuple[bool, Any, List[BoundaryViolation]]:
    """
    Enforce boundaries on an operation, executing it only if allowed.
    
    Args:
        operation_id: Unique identifier for the operation
        domain: Domain of the operation
        context: Context for the operation
        operation_fn: Function to execute if operation is allowed
        
    Returns:
        Tuple of (success, result, violations)
    """
    return get_boundary_enforcement_system().enforce_operation(
        operation_id=operation_id,
        domain=domain,
        context=context,
        operation_fn=operation_fn
    ) 