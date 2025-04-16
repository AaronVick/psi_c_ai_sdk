#!/usr/bin/env python3
"""
Schema Graph Builder Example

This example demonstrates how to use the Schema Graph Builder
to create a graph-based representation of memories and their relationships.
"""

import time
import random
import numpy as np
from typing import List

from psi_c_ai_sdk.memory import MemoryStore, Memory, MemoryType
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.schema import SchemaGraph


def generate_random_embedding(dim: int = 128) -> List[float]:
    """Generate a random embedding vector."""
    embedding = np.random.normal(0, 1, dim).tolist()
    return embedding


def create_themed_memories(theme: str, count: int = 5) -> List[Memory]:
    """Create a group of thematically related memories."""
    memories = []
    
    # Create a base embedding for the theme with some noise
    base_embedding = np.random.normal(0, 1, 128)
    
    for i in range(count):
        # Add some noise to create related but different embeddings
        noise = np.random.normal(0, 0.2, 128)
        embedding = (base_embedding + noise).tolist()
        
        # Create the memory
        memory = Memory(
            content=f"Memory about {theme} - aspect {i+1}",
            embedding=embedding,
            importance=random.uniform(0.5, 1.0),
            memory_type=MemoryType.SEMANTIC,
            tags=[theme, f"aspect_{i+1}"]
        )
        memories.append(memory)
    
    return memories


def main() -> None:
    print("ΨC-AI SDK: Schema Graph Builder Example")
    print("=" * 50)
    
    # Initialize the memory store
    memory_store = MemoryStore()
    
    # Initialize coherence scorer
    coherence_scorer = CoherenceScorer()
    
    # Initialize schema graph
    schema = SchemaGraph(
        memory_store=memory_store,
        coherence_scorer=coherence_scorer,
        min_edge_weight=0.3  # Minimum coherence score to create an edge
    )
    
    # Create themed memory groups
    themes = ["science", "history", "philosophy", "art", "technology"]
    all_memories = []
    
    print("\nCreating themed memory groups...")
    for theme in themes:
        memories = create_themed_memories(theme, count=5)
        all_memories.extend(memories)
        print(f"Created {len(memories)} memories about {theme}")
    
    # Add all memories to the memory store
    for memory in all_memories:
        memory_store.add_memory(memory)
    
    # Update the schema graph with all memories
    print("\nBuilding schema graph...")
    stats = schema.update_schema(max_memories=100)
    
    print(f"Added {stats['nodes_added']} nodes and {stats['edges_added']} edges")
    
    # Print graph statistics
    graph_stats = schema.get_stats()
    print("\nSchema Graph Statistics:")
    print(f"Nodes: {graph_stats['node_count']}")
    print(f"Edges: {graph_stats['edge_count']}")
    print(f"Node types: {graph_stats['node_types']}")
    print(f"Average node degree: {graph_stats['avg_degree']:.2f}")
    print(f"Clustering coefficient: {graph_stats['clustering_coefficient']:.2f}")
    print(f"Connected components: {graph_stats['connected_components']}")
    
    # Add a concept node (not tied to a specific memory)
    concept_id = schema.add_concept_node(
        label="Interdisciplinary Knowledge",
        importance=0.8,
        tags=["meta", "integration"]
    )
    print(f"\nAdded concept node: {concept_id}")
    
    # Find relationships between themes
    print("\nExploring relationships between themes:")
    for theme in themes:
        # Find a memory with this theme
        theme_memories = [m for m in all_memories if theme in m.tags]
        if not theme_memories:
            continue
            
        sample_memory = theme_memories[0]
        node_id = f"memory_{sample_memory.uuid}"
        
        # Get a subgraph around this memory
        subgraph = schema.get_subgraph(node_id, max_distance=1)
        
        print(f"\n{theme.capitalize()} connections:")
        print(f"- Connected to {len(subgraph) - 1} other nodes")
        
        # Print node connections
        for connected_node in subgraph:
            if connected_node != node_id:
                node_label = subgraph.nodes[connected_node].get("label", "Unknown")
                print(f"  - {node_label}")
    
    # Visualize the schema graph
    try:
        print("\nCreating schema visualization...")
        schema.visualize(filename="schema_graph.png", max_nodes=50)
        print("Visualization saved to schema_graph.png")
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("\nExample complete!")


if __name__ == "__main__":
    main() 