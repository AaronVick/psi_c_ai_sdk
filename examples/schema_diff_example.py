#!/usr/bin/env python3
"""
Schema Fingerprinting and Diff Example

This example demonstrates how to use the Schema Fingerprinting and Diff
tools to track and visualize changes in schema graphs over time.
"""

import time
import random
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt

from psi_c_ai_sdk.memory import MemoryStore, Memory, MemoryType
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.schema import (
    SchemaGraph,
    SchemaFingerprint,
    SchemaDiffCalculator,
    SchemaDriftMonitor
)


def create_themed_memories(theme: str, count: int, base_embedding: np.ndarray = None) -> List[Memory]:
    """Create memories around a specific theme."""
    memories = []
    
    # Create a base embedding for the theme if not provided
    if base_embedding is None:
        base_embedding = np.random.normal(0, 1, 128)
    
    for i in range(count):
        # Add some noise to create related but different embeddings
        noise = np.random.normal(0, 0.2, 128)
        embedding = (base_embedding + noise).tolist()
        
        # Create the memory
        memory = Memory(
            content=f"Memory about {theme} - concept {i+1}",
            embedding=embedding,
            importance=random.uniform(0.5, 1.0),
            memory_type=MemoryType.SEMANTIC,
            tags=[theme, f"concept_{i+1}"]
        )
        memories.append(memory)
    
    return memories


def snapshot_schema(schema_graph: SchemaGraph, drift_monitor: SchemaDriftMonitor) -> Dict[str, Any]:
    """Take a snapshot of the schema and report drift."""
    # Take snapshot
    snapshot = drift_monitor.take_snapshot()
    
    # Get fingerprint
    fingerprinter = SchemaFingerprint(schema_graph)
    merkle_root = fingerprinter.compute_fingerprint()
    
    # Combine info
    result = {
        "timestamp": snapshot["timestamp"],
        "merkle_root": merkle_root,
        "node_count": snapshot["node_count"],
        "edge_count": snapshot["edge_count"]
    }
    
    # Add drift info if available
    if "drift_score" in snapshot:
        result["drift_score"] = snapshot["drift_score"]
        result["significant_drift"] = snapshot["significant_drift"]
    
    return result


def main() -> None:
    print("ΨC-AI SDK: Schema Fingerprinting and Diff Example")
    print("=" * 60)
    
    # Initialize components
    memory_store = MemoryStore()
    coherence_scorer = CoherenceScorer()
    
    # Create a schema graph
    schema = SchemaGraph(
        memory_store=memory_store,
        coherence_scorer=coherence_scorer,
        min_edge_weight=0.3
    )
    
    # Create a drift monitor
    drift_monitor = SchemaDriftMonitor(
        schema_graph=schema,
        drift_threshold=0.15,
        max_history=5
    )
    
    # Initialize schema diff calculator (with empty previous schema for now)
    diff_calculator = SchemaDiffCalculator(schema)
    
    print("\n1. Creating initial schema")
    print("-" * 60)
    
    # Add initial memories about science
    science_memories = create_themed_memories("science", 5)
    for memory in science_memories:
        memory_store.add_memory(memory)
    
    # Update schema with initial memories
    schema.update_schema()
    print(f"Added {len(science_memories)} science memories to the schema")
    
    # Take first snapshot
    initial_snapshot = snapshot_schema(schema, drift_monitor)
    print(f"Initial schema fingerprint: {initial_snapshot['merkle_root'][:10]}...")
    print(f"Node count: {initial_snapshot['node_count']}")
    print(f"Edge count: {initial_snapshot['edge_count']}")
    
    # Save copy of the initial schema for later comparison
    previous_schema = schema
    
    print("\n2. Adding new memories to schema")
    print("-" * 60)
    
    # Add memories about history (different theme)
    history_memories = create_themed_memories("history", 7)
    for memory in history_memories:
        memory_store.add_memory(memory)
    
    # Update schema with new memories
    schema.update_schema()
    print(f"Added {len(history_memories)} history memories to the schema")
    
    # Take second snapshot
    second_snapshot = snapshot_schema(schema, drift_monitor)
    print(f"Updated schema fingerprint: {second_snapshot['merkle_root'][:10]}...")
    print(f"Node count: {second_snapshot['node_count']}")
    print(f"Edge count: {second_snapshot['edge_count']}")
    print(f"Drift score: {second_snapshot['drift_score']:.4f}")
    print(f"Significant drift: {second_snapshot['significant_drift']}")
    
    print("\n3. Calculating schema diff")
    print("-" * 60)
    
    # Create a new schema diff calculator with the current and previous schemas
    diff_calculator = SchemaDiffCalculator(schema, previous_schema)
    
    # Calculate the diff
    diff_results = diff_calculator.calculate_diff()
    
    # Print diff results
    print("Schema diff results:")
    print(f"Schemas match: {diff_results['schemas_match']}")
    print(f"Added nodes: {diff_results['nodes']['added_count']}")
    print(f"Added edges: {diff_results['edges']['added_count']}")
    
    # Print clustering changes
    if 'clustering_coefficient' in diff_results['clusters']:
        cluster_change = diff_results['clusters']['clustering_coefficient']['change']
        print(f"Clustering coefficient change: {cluster_change:.4f}")
    
    print("\n4. Adding more structure to the schema")
    print("-" * 60)
    
    # Add a concept node to tie things together
    concept_id = schema.add_concept_node(
        label="Knowledge Integration",
        importance=0.9,
        tags=["meta", "integration"]
    )
    print(f"Added concept node: {concept_id}")
    
    # Connect some nodes to this concept
    # (In a real application, this would happen based on semantic relationships)
    node_count = 0
    for node_id in schema.graph.nodes:
        if node_id != concept_id and random.random() < 0.3:
            schema.add_edge(concept_id, node_id, weight=0.7)
            node_count += 1
    
    print(f"Connected concept to {node_count} nodes")
    
    # Take third snapshot
    third_snapshot = snapshot_schema(schema, drift_monitor)
    print(f"New schema fingerprint: {third_snapshot['merkle_root'][:10]}...")
    print(f"Drift score: {third_snapshot['drift_score']:.4f}")
    
    # Get overall drift report
    drift_report = drift_monitor.get_drift_report()
    print("\nDrift analysis:")
    print(f"Average drift: {drift_report['avg_drift']:.4f}")
    print(f"Maximum drift: {drift_report['max_drift']:.4f}")
    print(f"Drift trend: {drift_report['trend']}")
    
    print("\n5. Visualizing schema evolution")
    print("-" * 60)
    
    try:
        # Create drift score plot
        plt.figure(figsize=(10, 6))
        plt.plot(drift_monitor.drift_scores, marker='o', linestyle='-')
        plt.axhline(y=drift_monitor.drift_threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
        plt.title('Schema Drift Over Time')
        plt.xlabel('Snapshot Index')
        plt.ylabel('Drift Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('schema_drift.png')
        print("Saved drift visualization to schema_drift.png")
        
        # Visualize the schema diff
        diff_calculator.visualize_diff(filename="schema_diff.png", show=False)
        print("Saved schema diff visualization to schema_diff.png")
        
        # Visualize the current schema
        schema.visualize(filename="current_schema.png", max_nodes=50)
        print("Saved current schema visualization to current_schema.png")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    print("\nExample complete!")


if __name__ == "__main__":
    main() 