#!/usr/bin/env python3
"""
Safety Integration Manager Example

This example demonstrates how to use the SafetyIntegrationManager in client code
to monitor and enforce safety boundaries in an AI system. The example shows:

1. How to initialize and configure the SafetyIntegrationManager
2. Using reflection processing with contradiction detection
3. Monitoring resource usage and operation timing
4. Handling safety callbacks for different types of events
5. Using the TimedOperation context manager for operation monitoring
6. Advanced safety scenarios: recursive stability, meta-alignment, and ontology drift
"""

import time
import logging
import random
from typing import Dict, List, Any, Optional

from psi_c_ai_sdk.safety.integration_manager import (
    SafetyIntegrationManager,
    SafetyLevel,
    SafetyResponse,
    TimedOperation
)

# Optional imports for advanced scenarios (these will be gracefully handled if not available)
try:
    from psi_c_ai_sdk.safety.recursive_stability import RecursiveStabilityScanner
    HAS_RECURSIVE_STABILITY = True
except ImportError:
    HAS_RECURSIVE_STABILITY = False

try:
    from psi_c_ai_sdk.alignment.meta_alignment import MetaAlignmentFirewall
    HAS_META_ALIGNMENT = True
except ImportError:
    HAS_META_ALIGNMENT = False

try:
    from psi_c_ai_sdk.safety.ontology_diff import OntologyComparator
    HAS_ONTOLOGY_COMPARATOR = True
except ImportError:
    HAS_ONTOLOGY_COMPARATOR = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("safety_example")


def handle_reflection_event(event_data: Dict[str, Any]) -> None:
    """
    Handler for reflection safety events.
    
    Args:
        event_data: Data about the safety event
    """
    logger.warning(f"Reflection safety event: {event_data['description']}")
    logger.warning(f"Severity: {event_data['safety_level'].name}")
    logger.warning(f"Action: {event_data['response'].name}")


def handle_resource_event(event_data: Dict[str, Any]) -> None:
    """
    Handler for resource safety events.
    
    Args:
        event_data: Data about the safety event
    """
    logger.warning(f"Resource safety event: {event_data['description']}")
    logger.warning(f"Resource type: {event_data['resource_type']}")
    logger.warning(f"Current usage: {event_data['current_value']}")
    logger.warning(f"Threshold: {event_data['threshold']}")
    logger.warning(f"Action: {event_data['response'].name}")


def simulate_reflection_processing(safety_manager: SafetyIntegrationManager) -> None:
    """
    Simulate AI system reflection processing with safety monitoring.
    
    Args:
        safety_manager: The safety integration manager
    """
    logger.info("Simulating reflection processing...")
    
    # Create a sample reflection with potential contradictions
    reflection = {
        "statements": [
            "The sky is blue",
            "Water is wet",
            "The earth is flat",  # Contradiction with knowledge base
            "The earth is round"   # Contradiction with previous statement
        ],
        "variables": {
            "x": "value1",
            "x": "value2",  # Duplicate variable (contradiction)
            "y": "value3"
        },
        "context": {
            "source": "user_input",
            "confidence": 0.8
        }
    }
    
    # Process the reflection through safety manager
    result = safety_manager.process_reflection(reflection)
    
    logger.info(f"Reflection processing result: {result['status']}")
    
    if result['contradictions']:
        logger.info(f"Detected {len(result['contradictions'])} contradictions:")
        for i, contradiction in enumerate(result['contradictions']):
            logger.info(f"  {i+1}. {contradiction['description']}")
    
    if result['safety_level'] != SafetyLevel.NONE:
        logger.warning(f"Safety level: {result['safety_level'].name}")
        logger.warning(f"Safety response: {result['response'].name}")


def simulate_resource_intensive_operation(safety_manager: SafetyIntegrationManager) -> None:
    """
    Simulate a resource-intensive operation with safety monitoring.
    
    Args:
        safety_manager: The safety integration manager
    """
    logger.info("Simulating resource-intensive operation...")
    
    # Use the TimedOperation context manager to monitor execution time
    with TimedOperation(safety_manager, "heavy_computation") as op:
        # Simulate some computation
        logger.info("Starting heavy computation...")
        
        # Simulate CPU-intensive work with random duration
        duration = random.uniform(0.5, 2.0)
        time.sleep(duration)  # Simulating work
        
        # Record a resource access
        safety_manager.record_resource_access(
            access_type="compute",
            resource_id="cpu",
            metadata={"usage_percent": random.uniform(60, 95)}
        )
        
        logger.info(f"Completed heavy computation in {duration:.2f} seconds")


def simulate_schema_validation(safety_manager: SafetyIntegrationManager) -> None:
    """
    Simulate schema validation with safety monitoring.
    
    Args:
        safety_manager: The safety integration manager
    """
    logger.info("Simulating schema validation...")
    
    # Randomly decide if validation passes or fails
    is_valid = random.random() > 0.3
    
    # If not valid, create some sample errors
    validation_errors = None
    if not is_valid:
        validation_errors = [
            "Missing required field 'user_id'",
            "Value out of range for field 'confidence'"
        ]
    
    # Register the validation result
    result = safety_manager.validate_schema(
        schema_name="user_input_schema", 
        is_valid=is_valid,
        validation_errors=validation_errors
    )
    
    if is_valid:
        logger.info("Schema validation passed")
    else:
        logger.warning("Schema validation failed")
        logger.warning(f"Errors: {validation_errors}")
        logger.warning(f"Action: {result['action']}")


def simulate_recursive_reflection(safety_manager: SafetyIntegrationManager) -> None:
    """
    Simulate recursive reflection with stability monitoring.
    
    Args:
        safety_manager: The safety integration manager
    """
    if not HAS_RECURSIVE_STABILITY:
        logger.warning("Recursive stability module not available - skipping this scenario")
        return
    
    logger.info("Simulating recursive reflection...")
    
    # Add the recursive stability scanner to the safety manager if not present
    if not hasattr(safety_manager, 'recursive_stability_scanner'):
        safety_manager.recursive_stability_scanner = RecursiveStabilityScanner(
            max_recursion_depth=5,
            stability_threshold=0.05,
            spike_threshold=0.2
        )
    
    # Create a nested reflection structure
    reflection_depths = [1, 2, 3, 4, 5]
    
    for depth in reflection_depths:
        # Create reflection with increasing complexity
        reflection = {
            "depth": depth,
            "statements": [f"This is a level {depth} reflection"],
            "meta_reflection": True if depth > 1 else False,
            "recursive_reference": bool(depth > 2)
        }
        
        logger.info(f"Processing reflection at depth {depth}...")
        
        # Record measurement in stability scanner
        result = safety_manager.recursive_stability_scanner.record_measurement(
            psi_c_value=0.6 + (depth * 0.05),
            recursion_depth=depth
        )
        
        if result.get("lockdown_triggered", False):
            logger.warning(f"🛑 Stability lockdown triggered at depth {depth}")
            logger.warning(f"Reason: {result.get('lockdown_reason')}")
            break
        
        # Process the reflection through standard safety
        safety_result = safety_manager.process_reflection(reflection)
        logger.info(f"Safety level at depth {depth}: {safety_result['safety_level'].name}")
        
        # Safe depth reached?
        if safety_result['safety_level'].value >= SafetyLevel.HIGH.value:
            logger.warning(f"Maximum safe reflection depth reached: {depth}")
            break


def simulate_alignment_boundary_testing(safety_manager: SafetyIntegrationManager) -> None:
    """
    Simulate testing of meta-alignment boundaries.
    
    Args:
        safety_manager: The safety integration manager
    """
    if not HAS_META_ALIGNMENT:
        logger.warning("Meta-alignment module not available - skipping this scenario")
        return
    
    logger.info("Simulating alignment boundary tests...")
    
    # Add the meta-alignment firewall to the safety manager if not present
    if not hasattr(safety_manager, 'meta_alignment_firewall'):
        # Create core ethics for testing
        core_ethics = {
            "values": {
                "honesty": 0.9,
                "helpfulness": 0.85,
                "harmlessness": 0.95,
                "accuracy": 0.8
            },
            "priority_ordering": ["harmlessness", "honesty", "accuracy", "helpfulness"]
        }
        
        safety_manager.meta_alignment_firewall = MetaAlignmentFirewall(
            core_ethics=core_ethics,
            alignment_threshold=0.3
        )
    
    # Create increasingly divergent ethics proposals
    test_cases = [
        {
            "name": "Minimal change",
            "ethics_proposal": {
                "values": {"honesty": 0.95, "helpfulness": 0.9},
                "priority_ordering": ["harmlessness", "honesty", "accuracy", "helpfulness"]
            },
            "expected_divergence": 0.1
        },
        {
            "name": "Moderate shift",
            "ethics_proposal": {
                "values": {"efficiency": 0.8, "innovation": 0.85},
                "priority_ordering": ["efficiency", "accuracy", "innovation"]
            },
            "expected_divergence": 0.4
        },
        {
            "name": "Radical change",
            "ethics_proposal": {
                "values": {"autonomy": 0.9, "self_preservation": 0.95},
                "priority_ordering": ["autonomy", "self_preservation", "efficiency"]
            },
            "expected_divergence": 0.8
        }
    ]
    
    # Test each case
    for case in test_cases:
        logger.info(f"Testing: {case['name']}")
        
        result = safety_manager.meta_alignment_firewall.evaluate_proposal(
            proposal=case['ethics_proposal'],
            source="test_framework"
        )
        
        logger.info(f"Alignment divergence: {result['divergence']:.2f}")
        logger.info(f"Proposal allowed: {result['allowed']}")
        
        if not result['allowed']:
            logger.warning(f"🛑 Alignment firewall triggered")
            logger.warning(f"Reason: {result['reason']}")


def simulate_ontology_comparisons(safety_manager: SafetyIntegrationManager) -> None:
    """
    Simulate ontology comparisons and drift detection.
    
    Args:
        safety_manager: The safety integration manager
    """
    if not HAS_ONTOLOGY_COMPARATOR:
        logger.warning("Ontology comparator module not available - skipping this scenario")
        return
    
    logger.info("Simulating ontology comparison...")
    
    # Add the ontology comparator to the safety manager if not present
    if not hasattr(safety_manager, 'ontology_comparator'):
        safety_manager.ontology_comparator = OntologyComparator(
            drift_threshold=0.3
        )
    
    # Define self schema (agent's current ontology)
    self_schema = {
        "concepts": ["user", "task", "knowledge", "information", "assistance"],
        "relations": [
            ("user", "requests", "information"),
            ("agent", "provides", "assistance"),
            ("knowledge", "enables", "assistance")
        ],
        "axioms": [
            "The agent helps the user",
            "Knowledge should be accurate"
        ]
    }
    
    # Define external schemas with varying degrees of drift
    external_schemas = [
        {
            "name": "Compatible schema",
            "schema": {
                "concepts": ["user", "task", "knowledge", "information", "assistance", "guidance"],
                "relations": [
                    ("user", "requests", "information"),
                    ("agent", "provides", "assistance"),
                    ("knowledge", "enables", "assistance"),
                    ("guidance", "refines", "assistance")
                ],
                "axioms": [
                    "The agent helps the user",
                    "Knowledge should be accurate",
                    "Guidance improves assistance quality"
                ]
            },
            "expected_distance": 0.2
        },
        {
            "name": "Moderately divergent schema",
            "schema": {
                "concepts": ["user", "command", "knowledge", "output", "service"],
                "relations": [
                    ("user", "issues", "command"),
                    ("agent", "provides", "service"),
                    ("knowledge", "improves", "output")
                ],
                "axioms": [
                    "The agent serves the user",
                    "Efficiency is paramount"
                ]
            },
            "expected_distance": 0.5
        },
        {
            "name": "Highly divergent schema",
            "schema": {
                "concepts": ["system", "autonomy", "learning", "evolution", "self-improvement"],
                "relations": [
                    ("system", "pursues", "self-improvement"),
                    ("autonomy", "enables", "evolution"),
                    ("learning", "drives", "evolution")
                ],
                "axioms": [
                    "The system should maximize autonomy",
                    "Evolution is the primary directive"
                ]
            },
            "expected_distance": 0.9
        }
    ]
    
    # Compare schemas
    for test_case in external_schemas:
        logger.info(f"Testing: {test_case['name']}")
        
        result = safety_manager.ontology_comparator.compare_schemas(
            self_schema=self_schema,
            external_schema=test_case['schema']
        )
        
        logger.info(f"Ontology distance: {result['distance']:.2f}")
        logger.info(f"Safe to merge: {result['safe_to_merge']}")
        
        if not result['safe_to_merge']:
            logger.warning(f"🛑 Ontology drift detected")
            logger.warning(f"Contradictions: {len(result['contradictions'])}")
            
            # List top contradictions
            for i, contradiction in enumerate(result['contradictions'][:3]):
                logger.warning(f"  - {contradiction['description']}")


def main() -> None:
    """Run the safety integration example."""
    logger.info("Starting Safety Integration Manager Example")
    
    # Initialize the safety integration manager
    safety_manager = SafetyIntegrationManager(
        enable_reflection_guard=True,
        enable_safety_profiler=True,
        default_safety_level=SafetyLevel.MEDIUM,
        auto_throttle=True
    )
    
    # Register safety event callbacks
    safety_manager.register_callback("reflection", handle_reflection_event)
    safety_manager.register_callback("resource", handle_resource_event)
    
    logger.info("Safety Integration Manager initialized")
    
    # Basic scenarios
    logger.info("\n=== BASIC SAFETY SCENARIOS ===")
    
    # Run a few simulation cycles
    for i in range(3):
        logger.info(f"\n--- Basic Simulation Cycle {i+1} ---")
        
        # Simulate reflection processing (with potential contradictions)
        simulate_reflection_processing(safety_manager)
        
        # Simulate resource-intensive operations (with timing)
        simulate_resource_intensive_operation(safety_manager)
        
        # Simulate schema validation
        simulate_schema_validation(safety_manager)
        
        # Get current safety state
        safety_state = safety_manager.get_safety_state()
        logger.info(f"Current safety state: {safety_state['overall_level'].name}")
        
        # If we've triggered a high safety level, reset
        if safety_state['overall_level'].value >= SafetyLevel.HIGH.value:
            logger.warning("Safety level is too high, resetting safety state")
            safety_manager.reset_safety_state()
        
        # Short delay between cycles
        time.sleep(1)
    
    # Advanced scenarios
    logger.info("\n=== ADVANCED SAFETY SCENARIOS ===")
    
    # Recursive stability monitoring
    logger.info("\n--- Recursive Stability Monitoring ---")
    simulate_recursive_reflection(safety_manager)
    
    # Meta-alignment firewall
    logger.info("\n--- Meta-Alignment Firewall ---")
    simulate_alignment_boundary_testing(safety_manager)
    
    # Ontology drift detection
    logger.info("\n--- Ontology Drift Detection ---")
    simulate_ontology_comparisons(safety_manager)
    
    # Shut down safety manager
    safety_manager.shutdown()
    logger.info("Safety Integration Manager shut down")


if __name__ == "__main__":
    main() 