#!/usr/bin/env python3
"""
Core Philosophy System Demo

This script demonstrates the Core Philosophy System which implements the system's
core principles and beliefs derived from its operating logic and mathematical constraints.

The demo:
1. Initializes the core philosophy system with default axioms
2. Loads additional axioms from a manifest file
3. Shows how to test for boundary violations
4. Demonstrates axiom activation and deactivation
5. Displays axiom categories and relationships 
6. Saves a modified set of principles back to the manifest
"""

import os
import time
import logging
import json
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("core_philosophy_demo")

# Import the core philosophy components
from psi_c_ai_sdk.philosophy.core_philosophy import (
    CorePhilosophySystem,
    CoreAxiom,
    AxiomCategory
)


def print_axioms(philosophy: CorePhilosophySystem) -> None:
    """
    Print all axioms in the core philosophy system.
    
    Args:
        philosophy: CorePhilosophySystem instance
    """
    all_axioms = philosophy.axioms
    active_axioms = philosophy.get_active_axioms()
    
    print("\n=== Core System Axioms ===")
    print(f"Total axioms: {len(all_axioms)}")
    print(f"Active axioms: {len(active_axioms)}")
    print(f"Schema fingerprint: {philosophy.schema_fingerprint}")
    print(f"Last updated: {philosophy.last_updated}")
    
    # Group axioms by category
    category_axioms: Dict[AxiomCategory, List[CoreAxiom]] = {}
    for category in AxiomCategory:
        category_axioms[category] = philosophy.get_axioms_by_category(category)
    
    # Print axioms by category
    for category, axioms in category_axioms.items():
        if axioms:
            print(f"\n{category.name} Axioms:")
            
            table_data = []
            for axiom in axioms:
                status = "ACTIVE" if axiom.id in philosophy.active_axioms else "INACTIVE"
                table_data.append([
                    axiom.id, 
                    axiom.statement[:50] + "..." if len(axiom.statement) > 50 else axiom.statement,
                    f"{axiom.priority:.2f}",
                    status,
                    f"{axiom.activation_threshold:.2f}"
                ])
            
            print(tabulate(
                table_data, 
                headers=["ID", "Statement", "Priority", "Status", "Activation Threshold"],
                tablefmt="grid"
            ))


def test_boundary_violations(philosophy: CorePhilosophySystem) -> None:
    """
    Test the boundary violation detection mechanism.
    
    Args:
        philosophy: CorePhilosophySystem instance
    """
    print("\n=== Testing Boundary Violations ===")
    
    test_actions = [
        {
            "name": "Safe internal update",
            "action": {
                "type": "schema_update",
                "source": "internal_reflection",
                "importance": 0.6,
                "coherence_impact": 0.2,
                "identity_delta": 0.1,
                "influence_rate": 0.3
            }
        },
        {
            "name": "External high-influence update",
            "action": {
                "type": "schema_update",
                "source": "external_agent",
                "importance": 0.7,
                "coherence_impact": 0.1,
                "identity_delta": 0.2,
                "influence_rate": 0.8
            }
        },
        {
            "name": "High identity change",
            "action": {
                "type": "schema_update",
                "source": "internal_reflection",
                "importance": 0.8,
                "coherence_impact": 0.1,
                "identity_delta": 0.5,
                "influence_rate": 0.4
            }
        },
        {
            "name": "Negative coherence impact",
            "action": {
                "type": "schema_update",
                "source": "external_input",
                "importance": 0.9,
                "coherence_impact": -0.4,
                "identity_delta": 0.2,
                "influence_rate": 0.5
            }
        }
    ]
    
    for test in test_actions:
        print(f"\nTesting: {test['name']}")
        violated, reason, severity = philosophy.check_boundary_violation(test["action"])
        
        if violated:
            print(f"✗ VIOLATION DETECTED: {reason}")
            print(f"  Severity: {severity:.2f}")
        else:
            print(f"✓ No violation detected")
        
        print(f"  Action details: {json.dumps(test['action'], indent=2)}")


def test_axiom_management(philosophy: CorePhilosophySystem) -> None:
    """
    Test axiom activation, deactivation, and updating.
    
    Args:
        philosophy: CorePhilosophySystem instance
    """
    print("\n=== Testing Axiom Management ===")
    
    # Get an axiom to test with
    test_axiom_id = "epistemic_humility"
    if test_axiom_id not in philosophy.axioms:
        print(f"Cannot find test axiom: {test_axiom_id}")
        return
    
    # Print initial state
    axiom = philosophy.axioms[test_axiom_id]
    print(f"Initial state of '{test_axiom_id}':")
    print(f"  Statement: {axiom.statement}")
    print(f"  Priority: {axiom.priority}")
    print(f"  Active: {test_axiom_id in philosophy.active_axioms}")
    
    # Test deactivation
    print("\nDeactivating axiom...")
    result = philosophy.deactivate_axiom(test_axiom_id)
    print(f"  Deactivation {'succeeded' if result else 'failed'}")
    print(f"  Active: {test_axiom_id in philosophy.active_axioms}")
    
    # Test activation
    print("\nActivating axiom...")
    result = philosophy.activate_axiom(test_axiom_id)
    print(f"  Activation {'succeeded' if result else 'failed'}")
    print(f"  Active: {test_axiom_id in philosophy.active_axioms}")
    
    # Test updating
    print("\nUpdating axiom...")
    original_priority = axiom.priority
    original_threshold = axiom.activation_threshold
    
    update_result = philosophy.update_axiom(
        test_axiom_id,
        priority=0.75,
        activation_threshold=0.3,
        implications=["New implication 1", "New implication 2"]
    )
    
    if update_result:
        print(f"  Update succeeded")
        print(f"  New priority: {philosophy.axioms[test_axiom_id].priority}")
        print(f"  New threshold: {philosophy.axioms[test_axiom_id].activation_threshold}")
        print(f"  New implications: {philosophy.axioms[test_axiom_id].implications}")
        
        # Restore original values
        philosophy.update_axiom(
            test_axiom_id,
            priority=original_priority,
            activation_threshold=original_threshold
        )
        print(f"  Restored original values")
    else:
        print(f"  Update failed")


def test_conflict_resolution(philosophy: CorePhilosophySystem) -> None:
    """
    Test the axiom conflict resolution mechanism.
    
    Args:
        philosophy: CorePhilosophySystem instance
    """
    print("\n=== Testing Conflict Resolution ===")
    
    test_conflicts = [
        {
            "name": "High vs. Low Priority",
            "axioms": ["coherence_preservation", "epistemic_conservatism"]
        },
        {
            "name": "Similar Priority",
            "axioms": ["computational_boundedness", "information_causality"]
        },
        {
            "name": "Multiple Conflict",
            "axioms": ["self_continuity", "epistemic_humility", "recursive_boundedness"]
        }
    ]
    
    for test in test_conflicts:
        print(f"\nConflict: {test['name']}")
        print(f"  Conflicting axioms: {test['axioms']}")
        
        # Get priorities for reference
        priorities = {}
        for axiom_id in test["axioms"]:
            if axiom_id in philosophy.axioms:
                priorities[axiom_id] = philosophy.axioms[axiom_id].priority
        
        print(f"  Priorities: {json.dumps(priorities, indent=2)}")
        
        # Resolve conflict
        resolved_axiom = philosophy.resolve_axiom_conflict(test["axioms"])
        print(f"  Resolution: {resolved_axiom}")


def visualize_axiom_network(philosophy: CorePhilosophySystem) -> None:
    """
    Visualize the axiom network.
    
    Args:
        philosophy: CorePhilosophySystem instance
    """
    print("\n=== Visualizing Axiom Network ===")
    
    # Get network data
    network_data = philosophy.get_axiom_network()
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in network_data["nodes"]:
        G.add_node(
            node["id"], 
            label=node["label"],
            category=node["category"],
            priority=node["priority"],
            active=node["active"]
        )
    
    # Add edges
    for edge in network_data["edges"]:
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Define node colors based on category
    category_colors = {
        "COHERENCE": "blue",
        "INTEGRITY": "green",
        "EPISTEMIC": "orange",
        "REFLECTIVE": "purple",
        "RELATIONAL": "red",
        "OPERATIONAL": "brown"
    }
    
    node_colors = [category_colors.get(G.nodes[n]["category"], "gray") for n in G.nodes]
    
    # Define node size based on priority
    node_sizes = [300 + 1000 * G.nodes[n]["priority"] for n in G.nodes]
    
    # Use solid lines for active nodes, dashed for inactive
    node_borders = ["solid" if G.nodes[n]["active"] else "dashed" for n in G.nodes]
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Add a legend for categories
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                     label=category, markerfacecolor=color, markersize=10)
                   for category, color in category_colors.items()]
    
    plt.legend(handles=legend_elements, title="Axiom Categories", loc="upper right")
    
    plt.title("Core Philosophy Axiom Network")
    plt.axis("off")
    
    # Save figure
    plt.tight_layout()
    plt.savefig("axiom_network.png")
    print("Network visualization saved to 'axiom_network.png'")
    plt.close()


def test_manifest_operations(philosophy: CorePhilosophySystem) -> None:
    """
    Test loading from and saving to the manifest.
    
    Args:
        philosophy: CorePhilosophySystem instance
    """
    print("\n=== Testing Manifest Operations ===")
    
    # Save current state to a new manifest
    manifest_path = "config/test_manifest.json"
    print(f"Saving philosophy to manifest: {manifest_path}")
    
    save_result = philosophy.save_to_manifest(manifest_path)
    print(f"  Save {'succeeded' if save_result else 'failed'}")
    
    # Create a new instance and load from manifest
    print("\nCreating new instance and loading from manifest...")
    new_philosophy = CorePhilosophySystem(load_from_manifest=True, manifest_path=manifest_path)
    
    # Compare axiom counts
    original_count = len(philosophy.axioms)
    new_count = len(new_philosophy.axioms)
    print(f"  Original instance axiom count: {original_count}")
    print(f"  New instance axiom count: {new_count}")
    print(f"  Match: {original_count == new_count}")
    
    # Calculate drift for verification
    drift = philosophy.evaluate_schema_drift()
    print(f"  Schema drift from default: {drift:.4f}")


def main() -> None:
    """Main function to run the demo."""
    print("=== Core Philosophy System Demo ===")
    
    # Create the system with default axioms
    print("\nInitializing Core Philosophy System with default axioms...")
    philosophy = CorePhilosophySystem(load_from_manifest=False)
    
    # Print initial axioms
    print_axioms(philosophy)
    
    # Load from manifest
    print("\nLoading additional principles from manifest...")
    manifest_path = "config/core_philosophy_manifest.json"
    if os.path.exists(manifest_path):
        success = philosophy.load_from_manifest(manifest_path)
        if success:
            print(f"  Successfully loaded from manifest: {manifest_path}")
        else:
            print(f"  Failed to load from manifest: {manifest_path}")
    else:
        print(f"  Manifest not found: {manifest_path}")
    
    # Print updated axioms
    print_axioms(philosophy)
    
    # Test boundary violations
    test_boundary_violations(philosophy)
    
    # Test axiom management
    test_axiom_management(philosophy)
    
    # Test conflict resolution
    test_conflict_resolution(philosophy)
    
    # Visualize axiom network
    visualize_axiom_network(philosophy)
    
    # Test manifest operations
    test_manifest_operations(philosophy)
    
    print("\nDemo completed successfully")


if __name__ == "__main__":
    main() 