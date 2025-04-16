#!/usr/bin/env python3
"""
Entropy Management System Demo

This script demonstrates how to use the entropy management system to:
1. Calculate various entropy measures on memory systems
2. Monitor memory coherence and stability
3. Make decisions based on entropy thresholds

The demo creates a set of memories with varying levels of coherence and
shows how entropy increases as coherence decreases.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from psi_c_ai_sdk.memory import MemoryStore, Memory
from psi_c_ai_sdk.embedding import EmbeddingEngine
from psi_c_ai_sdk.entropy import (
    EntropyCalculator,
    EmbeddingEntropyMeasure,
    SemanticCoherenceEntropyMeasure,
    TemporalEntropyMeasure
)


def create_test_memories(num_memories=50, coherence_level='high'):
    """
    Create test memories with varying levels of coherence.
    
    Args:
        num_memories: Number of memories to create
        coherence_level: 'high', 'medium', or 'low' coherence
    
    Returns:
        A list of Memory objects
    """
    # Create an embedding engine
    embedding_engine = EmbeddingEngine()
    
    # Define topics based on coherence level
    if coherence_level == 'high':
        # High coherence - all memories related to a single topic
        topics = ["AI research is making significant progress.", 
                  "Neural networks continue to advance in capability.",
                  "Deep learning models are becoming more sophisticated.",
                  "Machine learning algorithms are improving rapidly.",
                  "Artificial intelligence is being applied in new domains."]
    elif coherence_level == 'medium':
        # Medium coherence - memories span a few related topics
        topics = ["AI research is advancing quickly.",
                  "Climate change is affecting global weather patterns.",
                  "Renewable energy sources are becoming more efficient.",
                  "Machine learning applications are expanding.",
                  "Sustainable development requires new technologies."]
    else:
        # Low coherence - memories span many unrelated topics
        topics = ["AI research continues to advance.",
                  "The history of ancient Egypt is fascinating.",
                  "Basketball teams competed in the championship.",
                  "New recipes for vegetarian cuisine are trending.",
                  "Space exploration reaches new frontiers.",
                  "Classical music concerts attract diverse audiences.",
                  "Economic policies affect market stability.",
                  "Marine biology studies reveal new species."]
    
    # Create memories
    memories = []
    now = datetime.now()
    
    for i in range(num_memories):
        # Select a topic (more random selection for lower coherence)
        if coherence_level == 'low':
            topic = random.choice(topics)
        else:
            # For high and medium coherence, select topics with some continuity
            topic_idx = min(i // (num_memories // len(topics)), len(topics) - 1)
            topic = topics[topic_idx]
        
        # Create a memory with the topic
        memory = Memory(
            content=f"{topic} Detail #{i}: {random.choice(['Study', 'Research', 'Analysis', 'Investigation'])} shows {random.choice(['promising', 'interesting', 'unexpected', 'significant'])} results.",
            source="entropy_demo",
            importance=random.uniform(0.5, 1.0),
        )
        
        # Generate embedding
        memory.embedding = embedding_engine.generate_embedding(memory.content)
        
        # Set timestamp (more clustered for high coherence, more spread for low coherence)
        if coherence_level == 'high':
            time_offset = timedelta(minutes=i)
        elif coherence_level == 'medium':
            time_offset = timedelta(hours=i * 2)
        else:
            time_offset = timedelta(days=i)
        
        memory.timestamp = now - time_offset
        memories.append(memory)
    
    return memories


def plot_entropy_comparison(memory_sets, labels):
    """
    Plot entropy metrics for different memory sets
    
    Args:
        memory_sets: List of memory lists to compare
        labels: Labels for each memory set
    """
    # Initialize entropy calculators
    embedding_entropy = EmbeddingEntropyMeasure()
    semantic_entropy = SemanticCoherenceEntropyMeasure()
    temporal_entropy = TemporalEntropyMeasure()
    combined_entropy = EntropyCalculator()
    
    # Calculate entropy for each memory set
    embedding_values = []
    semantic_values = []
    temporal_values = []
    combined_values = []
    
    for memories in memory_sets:
        embedding_values.append(embedding_entropy.calculate_entropy(memories))
        semantic_values.append(semantic_entropy.calculate_entropy(memories))
        temporal_values.append(temporal_entropy.calculate_entropy(memories))
        combined_values.append(combined_entropy.calculate_total_entropy(memories))
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.2
    
    ax.bar(x - width*1.5, embedding_values, width, label='Embedding Entropy')
    ax.bar(x - width/2, semantic_values, width, label='Semantic Entropy')
    ax.bar(x + width/2, temporal_values, width, label='Temporal Entropy')
    ax.bar(x + width*1.5, combined_values, width, label='Combined Entropy')
    
    ax.set_ylabel('Entropy Score (0-1)')
    ax.set_title('Entropy Measures for Different Memory Coherence Levels')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('entropy_comparison.png')
    print(f"Plot saved as 'entropy_comparison.png'")
    plt.show()


def demonstrate_entropy_monitoring():
    """
    Demonstrate how entropy changes as we add less coherent memories
    """
    # Create a memory store
    memory_store = MemoryStore()
    
    # Create and initialize the entropy calculator
    entropy_calculator = EntropyCalculator()
    
    # Start with highly coherent memories
    coherent_memories = create_test_memories(20, 'high')
    
    # Add these to the memory store
    for memory in coherent_memories:
        memory_store.add_memory(memory)
    
    # Track entropy over time
    entropy_history = []
    entropy_history.append(entropy_calculator.calculate_total_entropy(memory_store.get_all_memories()))
    
    print(f"Initial entropy with 20 coherent memories: {entropy_history[0]:.4f}")
    
    # Now add some medium coherence memories
    medium_memories = create_test_memories(15, 'medium')
    for memory in medium_memories:
        memory_store.add_memory(memory)
        current_entropy = entropy_calculator.calculate_total_entropy(memory_store.get_all_memories())
        entropy_history.append(current_entropy)
    
    print(f"Entropy after adding 15 medium-coherence memories: {entropy_history[-1]:.4f}")
    
    # Finally add some low coherence memories
    low_memories = create_test_memories(15, 'low')
    for memory in low_memories:
        memory_store.add_memory(memory)
        current_entropy = entropy_calculator.calculate_total_entropy(memory_store.get_all_memories())
        entropy_history.append(current_entropy)
    
    print(f"Final entropy after adding 15 low-coherence memories: {entropy_history[-1]:.4f}")
    
    # Plot the entropy progression
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(entropy_history)), entropy_history)
    plt.axvline(x=0, color='g', linestyle='--', label='Start')
    plt.axvline(x=20, color='y', linestyle='--', label='Medium coherence added')
    plt.axvline(x=35, color='r', linestyle='--', label='Low coherence added')
    plt.xlabel('Memories Added')
    plt.ylabel('Total Entropy')
    plt.title('Entropy Progression as Memories with Lower Coherence are Added')
    plt.legend()
    plt.grid(True)
    plt.savefig('entropy_progression.png')
    print(f"Plot saved as 'entropy_progression.png'")
    plt.show()


def main():
    # Compare different coherence levels
    print("Generating memory sets with different coherence levels...")
    high_coherence = create_test_memories(40, 'high')
    medium_coherence = create_test_memories(40, 'medium')
    low_coherence = create_test_memories(40, 'low')
    
    memory_sets = [high_coherence, medium_coherence, low_coherence]
    labels = ['High Coherence', 'Medium Coherence', 'Low Coherence']
    
    print("Plotting entropy comparison...")
    plot_entropy_comparison(memory_sets, labels)
    
    print("\nDemonstrating entropy monitoring over time...")
    demonstrate_entropy_monitoring()
    
    print("\nEntropy Management System Demo completed.")


if __name__ == "__main__":
    main() 