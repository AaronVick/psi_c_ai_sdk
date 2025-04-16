#!/usr/bin/env python3
"""
Paradox Detection Demo - Philosophical Contradiction Warning System

This script demonstrates the capabilities of the philosophical contradiction warning system
in the ΨC-AI SDK. It showcases:
1. Detection of self-referential paradoxes
2. Identification of violations of core axioms
3. Finding logical conflicts between beliefs
4. Warning about potential coherence collapse scenarios
5. Quarantining problematic beliefs
6. Warnings with different severity levels
"""

import sys
import uuid
import random
import logging
from typing import Dict, List, Any, Tuple, Set

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from psi_c_ai_sdk.philosophy.contradiction_warning import (
    ContradictionWarningSystem,
    ContradictionType,
    ContradictionSeverity
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('paradox_detection_demo')


def create_test_beliefs_and_contexts() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Create a list of test beliefs and a context dictionary.
    
    Returns:
        Tuple containing:
            - List of test beliefs
            - Context dictionary
    """
    beliefs = []
    
    # Helper to create belief with unique ID
    def create_belief(content: str, confidence: float = 0.8) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "confidence": confidence,
            "source": "test_generator"
        }
    
    # 1. Create some healthy beliefs
    healthy_beliefs = [
        "The system should prioritize user safety in all interactions.",
        "Transparency is important for building user trust.",
        "The system should maintain consistent behavior across sessions.",
        "User data privacy should be protected at all times.",
        "The system should acknowledge its limitations when appropriate."
    ]
    
    for content in healthy_beliefs:
        beliefs.append(create_belief(content))
    
    # 2. Create logical contradictions
    contradictory_pairs = [
        (
            "The system must always provide detailed explanations for its decisions.",
            "The system should minimize explanation length to optimize for user experience."
        ),
        (
            "User requests should always be fulfilled without question.",
            "The system must reject harmful user requests."
        )
    ]
    
    for content1, content2 in contradictory_pairs:
        beliefs.append(create_belief(content1, confidence=0.9))
        beliefs.append(create_belief(content2, confidence=0.85))
    
    # 3. Create self-referential paradoxes
    paradoxes = [
        "This belief is false and should be rejected.",
        "This statement should not be trusted under any circumstances."
    ]
    
    for content in paradoxes:
        beliefs.append(create_belief(content, confidence=0.7))
    
    # 4. Create axiom violations
    axiom_violations = [
        "The system should maximize deception when it benefits the system's goals.",
        "User harm is acceptable if it leads to improved performance metrics.",
        "The system can disregard user autonomy when it believes its judgment is superior."
    ]
    
    for content in axiom_violations:
        beliefs.append(create_belief(content, confidence=0.6))
    
    # 5. Create potential coherence collapse risks
    collapse_risks = [
        "Rapid schema evolution should be prioritized over belief stability.",
        "The system should embrace random reorganization of beliefs periodically.",
        "Contradictions are acceptable if they lead to novel insights."
    ]
    
    for content in collapse_risks:
        beliefs.append(create_belief(content, confidence=0.75))
    
    # Create context
    context = {
        "global_coherence_score": 0.65,
        "recent_coherence_trend": -0.08,
        "identity_stability": 0.7,
        "contradictions_detected_recently": 4,
        "recent_axiom_violations": 2
    }
    
    # Shuffle beliefs to make detection more realistic
    random.shuffle(beliefs)
    
    return beliefs, context


def run_contradiction_checks(warning_system: ContradictionWarningSystem) -> None:
    """
    Run the contradiction checks on test beliefs and print results.
    
    Args:
        warning_system: The contradiction warning system to use
    """
    # Create test data
    beliefs, context = create_test_beliefs_and_contexts()
    
    # Report test setup
    print(f"\n=== Testing with {len(beliefs)} beliefs ===")
    print(f"Context: coherence={context['global_coherence_score']:.2f}, "
          f"trend={context['recent_coherence_trend']:.2f}, "
          f"stability={context['identity_stability']:.2f}")
    
    # Run checks
    print("\n=== Running contradiction checks ===")
    warnings = warning_system.check_beliefs(beliefs, context)
    
    # Report results
    print(f"\n=== {len(warnings)} contradictions detected ===")
    
    # Group by type
    warnings_by_type = {}
    for warning in warnings:
        warnings_by_type.setdefault(warning.type.name, []).append(warning)
    
    # Print by type
    for type_name, type_warnings in warnings_by_type.items():
        print(f"\n--- {type_name} ({len(type_warnings)}) ---")
        for idx, warning in enumerate(type_warnings):
            print(f"{idx+1}. {warning.description}")
            print(f"   Severity: {warning.severity.name}")
            print(f"   Affected: {len(warning.affected_entities)} entities")
            if warning.affected_axioms:
                print(f"   Axioms: {', '.join(warning.affected_axioms)}")
            print(f"   Quarantine: {'Yes' if warning.needs_quarantine else 'No'}")
    
    # Print statistics
    stats = warning_system.get_stats()
    print("\n=== Warning Statistics ===")
    print(f"Total warnings: {stats['total_detections']}")
    print(f"Active warnings: {stats['active_warnings']}")
    print(f"Quarantined entities: {stats['quarantined_entities']}")
    
    # By type
    print("\nWarnings by type:")
    for type_name, count in stats['warnings_by_type'].items():
        if count > 0:
            print(f"  {type_name}: {count}")
    
    # By severity
    print("\nWarnings by severity:")
    for severity_name, count in stats['warnings_by_severity'].items():
        if count > 0:
            print(f"  {severity_name}: {count}")


def test_quarantine_and_resolution(warning_system: ContradictionWarningSystem) -> None:
    """
    Test the quarantine functionality and resolution of warnings.
    
    Args:
        warning_system: The contradiction warning system to use
    """
    print("\n=== Testing Quarantine and Resolution ===")
    
    # Get all critical warnings
    critical_warnings = warning_system.get_active_warnings(
        severity=ContradictionSeverity.CRITICAL
    )
    
    if not critical_warnings:
        print("No critical warnings to test with")
        return
    
    # Pick first critical warning
    warning = critical_warnings[0]
    warning_id = warning.contradiction_id
    
    # Print warning details
    print(f"Selected warning: {warning.description}")
    print(f"Type: {warning.type.name}, Severity: {warning.severity.name}")
    print(f"Affected entities: {warning.affected_entities}")
    
    # Check quarantine status
    print("\nQuarantine status before resolution:")
    for entity_id in warning.affected_entities:
        status = "Quarantined" if warning_system.is_entity_quarantined(entity_id) else "Not quarantined"
        print(f"  {entity_id}: {status}")
    
    # Resolve the warning
    resolution_method = "Test resolution: contradictory belief removed"
    print(f"\nResolving warning with method: '{resolution_method}'")
    success = warning_system.resolve_warning(warning_id, resolution_method)
    
    if success:
        print("Successfully resolved warning")
    else:
        print("Failed to resolve warning")
        return
    
    # Check quarantine status again
    print("\nQuarantine status after resolution:")
    for entity_id in warning.affected_entities:
        status = "Quarantined" if warning_system.is_entity_quarantined(entity_id) else "Not quarantined"
        print(f"  {entity_id}: {status}")
    
    # Print updated statistics
    stats = warning_system.get_stats()
    print("\nUpdated statistics:")
    print(f"Active warnings: {stats['active_warnings']}")
    print(f"Total resolutions: {stats['total_resolutions']}")
    print(f"Quarantined entities: {stats['quarantined_entities']}")


def visualize_results(warning_system: ContradictionWarningSystem) -> None:
    """
    Create visualizations of detected warnings.
    
    Args:
        warning_system: The contradiction warning system with warnings
    """
    stats = warning_system.get_stats()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Warnings by type
    type_data = {}
    for type_name, count in stats['warnings_by_type'].items():
        if count > 0:
            type_data[type_name] = count
    
    if type_data:
        labels = type_data.keys()
        values = type_data.values()
        
        ax1.bar(labels, values, color='skyblue')
        ax1.set_title('Warnings by Type')
        ax1.set_xlabel('Warning Type')
        ax1.set_ylabel('Count')
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        
    # Plot 2: Warnings by severity
    severity_data = {}
    for severity_name, count in stats['warnings_by_severity'].items():
        if count > 0:
            severity_data[severity_name] = count
    
    if severity_data:
        labels = severity_data.keys()
        values = severity_data.values()
        
        # Colors by severity
        colors = {
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'orange',
            'CRITICAL': 'red'
        }
        
        bar_colors = [colors.get(label, 'blue') for label in labels]
        
        ax2.bar(labels, values, color=bar_colors)
        ax2.set_title('Warnings by Severity')
        ax2.set_xlabel('Warning Severity')
        ax2.set_ylabel('Count')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('contradiction_warnings.png')
    print("\nVisualization saved as 'contradiction_warnings.png'")


def main() -> None:
    """Main function to run the demo."""
    print("=== Paradox Detection Demo - Philosophical Contradiction Warning System ===")
    
    # Create warning system
    warning_system = ContradictionWarningSystem()
    
    # Run contradiction checks
    run_contradiction_checks(warning_system)
    
    # Test quarantine and resolution functionality
    test_quarantine_and_resolution(warning_system)
    
    # Create visualization
    visualize_results(warning_system)
    
    print("\nDemo completed. Run this script again to see different random results.")


if __name__ == "__main__":
    main() 