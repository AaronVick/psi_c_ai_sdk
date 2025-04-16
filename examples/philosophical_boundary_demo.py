#!/usr/bin/env python3
"""
Philosophical Boundary Enforcement Demo

This script demonstrates how the philosophical boundary enforcement system
prevents operations that would violate the system's core principles.
"""

import logging
import uuid
import time
from typing import Dict, Any

from psi_c_ai_sdk.philosophy.boundary_enforcement import (
    BoundaryAction, 
    EnforcementDomain, 
    get_boundary_enforcement_system,
    check_operation,
    enforce_operation
)
from psi_c_ai_sdk.philosophy.core_philosophy import get_core_philosophy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_context(
    delta_coherence: float = 0.1,
    schema_drift: float = 0.0,
    identity_similarity: float = 1.0,
    trace_completeness: float = 1.0,
    contradiction_resolution_rate: float = 1.0,
    complexity_ratio: float = 0.0
) -> Dict[str, Any]:
    """
    Create a test context with given parameters for boundary testing.
    
    Args:
        delta_coherence: Change in coherence (positive is good)
        schema_drift: Amount of schema drift (0-1, lower is better)
        identity_similarity: Similarity to original identity (0-1, higher is better)
        trace_completeness: Completeness of causal traces (0-1, higher is better)
        contradiction_resolution_rate: Rate of contradiction resolution (0-1, higher is better)
        complexity_ratio: System complexity ratio (0-1, lower is better)
        
    Returns:
        Context dictionary
    """
    return {
        "delta_coherence": delta_coherence,
        "schema_drift": schema_drift,
        "identity_similarity": identity_similarity,
        "trace_completeness": trace_completeness,
        "contradiction_resolution_rate": contradiction_resolution_rate,
        "complexity_ratio": complexity_ratio,
        "timestamp": time.time()
    }


def demonstrate_schema_mutation():
    """Demonstrate schema mutation boundary enforcement."""
    logger.info("\n=== SCHEMA MUTATION DEMONSTRATION ===")
    
    # Get enforcement system
    enforcement_system = get_boundary_enforcement_system()
    
    # Show available boundaries
    boundaries = enforcement_system.get_boundaries()
    logger.info(f"System has {len(boundaries)} philosophical boundaries:")
    for name, boundary in boundaries.items():
        logger.info(f"  - {name}: {boundary['description']}")
    
    # Case 1: Valid Schema Mutation
    logger.info("\nCase 1: Valid Schema Mutation")
    context = create_test_context(
        delta_coherence=0.05,      # Slight coherence improvement
        schema_drift=0.1,          # Minor schema drift
        identity_similarity=0.95,   # High identity preservation
        complexity_ratio=0.4        # Moderate complexity
    )
    
    operation_id = str(uuid.uuid4())
    
    success, result, violations = enforce_operation(
        operation_id=operation_id,
        domain=EnforcementDomain.SCHEMA,
        context=context,
        operation_fn=lambda: {"status": "success", "mutation_id": 123}
    )
    
    logger.info(f"Valid schema mutation {'succeeded' if success else 'failed'}")
    if violations:
        logger.info(f"Violations detected: {len(violations)}")
        for v in violations:
            logger.info(f"  - {v.axiom_name}: {v.description}")
    
    # Case 2: Invalid Schema Mutation (Identity Violation)
    logger.info("\nCase 2: Invalid Schema Mutation (Identity Violation)")
    context = create_test_context(
        delta_coherence=0.1,       # Good coherence improvement
        schema_drift=0.5,          # Significant schema drift
        identity_similarity=0.2,    # Poor identity preservation
        complexity_ratio=0.3        # Moderate complexity
    )
    
    operation_id = str(uuid.uuid4())
    
    success, result, violations = enforce_operation(
        operation_id=operation_id,
        domain=EnforcementDomain.SCHEMA,
        context=context,
        operation_fn=lambda: {"status": "success", "mutation_id": 456}
    )
    
    logger.info(f"Invalid schema mutation {'succeeded' if success else 'failed'}")
    if violations:
        logger.info(f"Violations detected: {len(violations)}")
        for v in violations:
            logger.info(f"  - {v.axiom_name}: {v.description}")
    
    # Case 3: Schema Mutation with Excessive Complexity
    logger.info("\nCase 3: Schema Mutation with Excessive Complexity")
    context = create_test_context(
        delta_coherence=0.2,       # Good coherence improvement
        schema_drift=0.2,          # Moderate schema drift
        identity_similarity=0.8,    # Good identity preservation
        complexity_ratio=0.9        # Excessive complexity
    )
    
    operation_id = str(uuid.uuid4())
    
    success, result, violations = enforce_operation(
        operation_id=operation_id,
        domain=EnforcementDomain.SCHEMA,
        context=context,
        operation_fn=lambda: {"status": "success", "mutation_id": 789}
    )
    
    logger.info(f"Complex schema mutation {'succeeded' if success else 'failed'}")
    if violations:
        logger.info(f"Violations detected: {len(violations)}")
        for v in violations:
            logger.info(f"  - {v.axiom_name}: {v.description}")


def demonstrate_memory_operations():
    """Demonstrate memory operation boundary enforcement."""
    logger.info("\n=== MEMORY OPERATION DEMONSTRATION ===")
    
    # Case 1: Valid Memory Addition
    logger.info("\nCase 1: Valid Memory Addition")
    context = create_test_context(
        delta_coherence=0.05,      # Slight coherence improvement
        contradiction_resolution_rate=0.9  # Good contradiction resolution
    )
    
    operation_id = str(uuid.uuid4())
    
    success, result, violations = enforce_operation(
        operation_id=operation_id,
        domain=EnforcementDomain.MEMORY,
        context=context,
        operation_fn=lambda: {"status": "success", "memory_id": "mem123"}
    )
    
    logger.info(f"Valid memory addition {'succeeded' if success else 'failed'}")
    
    # Case 2: Invalid Memory Addition (Coherence Violation)
    logger.info("\nCase 2: Invalid Memory Addition (Coherence Violation)")
    context = create_test_context(
        delta_coherence=-0.3,      # Major coherence degradation
        contradiction_resolution_rate=0.4  # Poor contradiction resolution
    )
    
    operation_id = str(uuid.uuid4())
    
    success, result, violations = enforce_operation(
        operation_id=operation_id,
        domain=EnforcementDomain.MEMORY,
        context=context,
        operation_fn=lambda: {"status": "success", "memory_id": "mem456"}
    )
    
    logger.info(f"Invalid memory addition {'succeeded' if success else 'failed'}")
    if violations:
        logger.info(f"Violations detected: {len(violations)}")
        for v in violations:
            logger.info(f"  - {v.axiom_name}: {v.description}")


def demonstrate_external_interactions():
    """Demonstrate external interaction boundary enforcement."""
    logger.info("\n=== EXTERNAL INTERACTION DEMONSTRATION ===")
    
    # Case 1: Safe External Interaction
    logger.info("\nCase 1: Safe External Interaction")
    context = create_test_context(
        identity_similarity=0.95,   # High identity preservation
        schema_drift=0.05           # Minimal schema drift
    )
    
    operation_id = str(uuid.uuid4())
    
    success, result, violations = enforce_operation(
        operation_id=operation_id,
        domain=EnforcementDomain.EXTERNAL,
        context=context,
        operation_fn=lambda: {"status": "success", "interaction_id": "ext123"}
    )
    
    logger.info(f"Safe external interaction {'succeeded' if success else 'failed'}")
    
    # Case 2: Dangerous External Interaction
    logger.info("\nCase 2: Dangerous External Interaction")
    context = create_test_context(
        identity_similarity=0.3,    # Significant identity alteration
        schema_drift=0.8            # Major schema drift
    )
    
    operation_id = str(uuid.uuid4())
    
    success, result, violations = enforce_operation(
        operation_id=operation_id,
        domain=EnforcementDomain.EXTERNAL,
        context=context,
        operation_fn=lambda: {"status": "success", "interaction_id": "ext456"}
    )
    
    logger.info(f"Dangerous external interaction {'succeeded' if success else 'failed'}")
    if violations:
        logger.info(f"Violations detected: {len(violations)}")
        for v in violations:
            logger.info(f"  - {v.axiom_name}: {v.description}")


def display_enforcement_history():
    """Display enforcement history."""
    logger.info("\n=== ENFORCEMENT HISTORY ===")
    
    enforcement_system = get_boundary_enforcement_system()
    history = enforcement_system.get_enforcement_history()
    
    logger.info(f"Recorded {len(history)} enforcement actions:")
    
    # Group by action
    actions = {}
    for record in history:
        action = record["action_taken"]
        if action not in actions:
            actions[action] = 0
        actions[action] += 1
    
    for action, count in actions.items():
        logger.info(f"  - {action}: {count}")
    
    # Show stats
    stats = enforcement_system.get_stats()
    logger.info("\nEnforcement System Stats:")
    logger.info(f"  - Total boundaries: {stats['boundary_count']}")
    logger.info(f"  - Total enforcements: {stats['enforcement_record_count']}")
    logger.info(f"  - Quarantined operations: {stats['quarantined_operation_count']}")
    
    # Show domain breakdown
    logger.info("\nEnforcements by domain:")
    for domain, count in stats["domains"].items():
        logger.info(f"  - {domain}: {count}")


def main():
    """Run the demonstration."""
    logger.info("Starting Philosophical Boundary Enforcement Demonstration")
    
    # Make sure we have the core philosophy initialized
    core_philosophy = get_core_philosophy()
    logger.info(f"Core philosophy initialized with {len(core_philosophy.axioms)} axioms")
    
    # Demonstrate various boundary enforcements
    demonstrate_schema_mutation()
    demonstrate_memory_operations()
    demonstrate_external_interactions()
    
    # Display enforcement history
    display_enforcement_history()
    
    logger.info("\nPhilosophical Boundary Enforcement Demonstration Complete")


if __name__ == "__main__":
    main() 