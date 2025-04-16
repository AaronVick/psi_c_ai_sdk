#!/usr/bin/env python3
"""
Schema Mutation System Demo

This example demonstrates how to use the Schema Mutation System to evolve and 
adapt the schema graph when coherence drops or contradictions are detected.
"""

import time
import uuid
import random
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from psi_c_ai_sdk.memory import MemoryStore, Memory, MemoryType
from psi_c_ai_sdk.embedding import EmbeddingEngine
from psi_c_ai_sdk.coherence import CoherenceScorer, BasicCoherenceScorer
from psi_c_ai_sdk.schema import SchemaGraph, SchemaMutationSystem, MutationType

# Sample data for creating memories
SAMPLE_MEMORIES = [
    # Topic 1: Animals
    {"content": "Dogs are loyal pets that offer companionship and protection.", "tags": ["animals", "pets", "dogs"]},
    {"content": "Cats are independent pets that are known for their grooming habits.", "tags": ["animals", "pets", "cats"]},
    {"content": "Wolves are wild canines that live in packs and hunt together.", "tags": ["animals", "wild", "wolves"]},
    {"content": "Lions are large cats that live in prides on the African savanna.", "tags": ["animals", "wild", "lions"]},
    
    # Topic 2: Technology
    {"content": "Artificial intelligence is revolutionizing various industries.", "tags": ["technology", "AI"]},
    {"content": "Machine learning models require large datasets for training.", "tags": ["technology", "AI", "ML"]},
    {"content": "Neural networks are inspired by the human brain's structure.", "tags": ["technology", "AI", "neural networks"]},
    {"content": "Quantum computing uses quantum bits instead of classical bits.", "tags": ["technology", "quantum", "computing"]},
    
    # Topic 3: Food
    {"content": "Pizza is a popular dish consisting of a flat round base with toppings.", "tags": ["food", "italian", "pizza"]},
    {"content": "Sushi is a Japanese dish made with vinegared rice and various fillings.", "tags": ["food", "japanese", "sushi"]},
    {"content": "Tacos are a traditional Mexican dish with a folded tortilla and fillings.", "tags": ["food", "mexican", "tacos"]},
    {"content": "Pasta comes in various shapes and is a staple of Italian cuisine.", "tags": ["food", "italian", "pasta"]},
    
    # Topic 4: Mixed (potential contradictions)
    {"content": "Some people claim that dogs are better pets than cats.", "tags": ["opinion", "pets", "comparison"]},
    {"content": "Many argue that cats make better pets than dogs for apartment living.", "tags": ["opinion", "pets", "comparison"]},
    {"content": "AI will eventually surpass human intelligence in all domains.", "tags": ["opinion", "technology", "AI"]},
    {"content": "AI will always be limited compared to human creativity and intuition.", "tags": ["opinion", "technology", "AI"]}
]


def create_memory_store() -> MemoryStore:
    """Create and populate a memory store with sample memories."""
    memory_store = MemoryStore()
    embedding_engine = EmbeddingEngine()
    
    for i, memory_data in enumerate(SAMPLE_MEMORIES):
        # Create memory
        memory = Memory(
            uuid=uuid.uuid4(),
            content=memory_data["content"],
            memory_type=MemoryType.FACT,
            importance=random.uniform(0.5, 1.0),
            tags=memory_data["tags"],
            source="example"
        )
        
        # Generate embedding
        memory.embedding = embedding_engine.get_embedding(memory.content)
        
        # Add to store
        memory_store.add_memory(memory)
        print(f"Added memory {i+1}: {memory.content[:40]}...")
        
    return memory_store


def visualize_schema(schema_graph: SchemaGraph, title: str, filename: str):
    """Visualize the schema graph and save to file."""
    schema_graph.visualize(filename=filename, max_nodes=100)
    print(f"Schema visualization saved as {filename}")


def main():
    """Run the schema mutation demo."""
    print("=== Schema Mutation System Demo ===")
    
    # 1. Create memory store with sample memories
    print("\n1. Creating memory store with sample memories...")
    memory_store = create_memory_store()
    
    # 2. Initialize coherence scorer
    print("\n2. Initializing coherence scorer...")
    coherence_scorer = BasicCoherenceScorer()
    
    # 3. Create schema graph
    print("\n3. Creating schema graph...")
    schema_graph = SchemaGraph(
        memory_store=memory_store,
        coherence_scorer=coherence_scorer,
        min_edge_weight=0.3
    )
    
    # 4. Initialize schema with memories
    print("\n4. Initializing schema graph with memories...")
    schema_graph.update_schema(max_memories=100)
    
    # Display initial schema statistics
    num_nodes = len(schema_graph.graph.nodes)
    num_edges = len(schema_graph.graph.edges)
    print(f"   Initial schema: {num_nodes} nodes, {num_edges} edges")
    
    # Visualize initial schema
    visualize_schema(schema_graph, "Initial Schema", "schema_initial.png")
    
    # 5. Create schema mutation system
    print("\n5. Creating schema mutation system...")
    mutation_system = SchemaMutationSystem(
        schema_graph=schema_graph,
        coherence_scorer=coherence_scorer,
        memory_store=memory_store,
        mutation_threshold=0.7,  # High threshold to ensure mutations happen
        initial_temperature=1.0,
        cooling_rate=0.1
    )
    
    # 6. Perform different types of mutations
    print("\n6. Performing mutations...")
    
    # a. Add concept nodes
    print("\n   a. Adding concept nodes...")
    mutation_event = mutation_system.force_mutation(MutationType.ADD_CONCEPT)
    if mutation_event:
        print(f"      Added concept: affected {len(mutation_event.affected_nodes)} nodes")
        
    # Visualize after adding concepts
    visualize_schema(schema_graph, "After Adding Concepts", "schema_after_concepts.png")
    
    # b. Merge similar nodes
    print("\n   b. Merging similar nodes...")
    mutation_event = mutation_system.force_mutation(MutationType.MERGE)
    if mutation_event:
        print(f"      Merged nodes: affected {len(mutation_event.affected_nodes)} nodes")
        
    # Visualize after merging
    visualize_schema(schema_graph, "After Merging", "schema_after_merge.png")
    
    # c. Prune weak connections
    print("\n   c. Pruning weak connections...")
    mutation_event = mutation_system.force_mutation(MutationType.PRUNE)
    if mutation_event:
        print(f"      Pruned connections: affected {len(mutation_event.affected_edges)} edges")
        
    # Visualize after pruning
    visualize_schema(schema_graph, "After Pruning", "schema_after_prune.png")
    
    # d. Restructure subgraph
    print("\n   d. Restructuring subgraph...")
    mutation_event = mutation_system.force_mutation(MutationType.RESTRUCTURE)
    if mutation_event:
        print(f"      Restructured: affected {len(mutation_event.affected_nodes)} nodes")
        
    # Visualize final schema
    visualize_schema(schema_graph, "Final Schema", "schema_final.png")
    
    # 7. Display mutation statistics
    print("\n7. Mutation statistics:")
    stats = mutation_system.get_mutation_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 8. Display mutation history
    print("\n8. Mutation history:")
    history = mutation_system.get_mutation_history(limit=5)
    for event in history:
        print(f"   {event['mutation_type']} at {event['timestamp']}: " + 
              f"coherence {event['coherence_before']:.3f} → {event['coherence_after']:.3f}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main() 