"""
Basic Example - ΨC-AI SDK

This example demonstrates the core functionality of the ΨC-AI SDK, including:
- Memory Storage
- Embedding Generation
- Coherence Calculation
- Contradiction Detection
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import from psi_c_ai_sdk
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psi_c_ai_sdk.memory.memory import MemoryStore
from psi_c_ai_sdk.embedding.embedding import EmbeddingEngine
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer
from psi_c_ai_sdk.contradiction.contradiction import ContradictionDetector

# Could also use this import style after installation:
# from psi_c_ai_sdk import MemoryStore, EmbeddingEngine, CoherenceScorer, ContradictionDetector


def main():
    # Initialize components
    print("Initializing ΨC-AI SDK components...")
    memory_store = MemoryStore(decay_constant=0.05)
    embedding_engine = EmbeddingEngine(use_cache=True)
    coherence_scorer = CoherenceScorer(embedding_engine, tag_weight=0.2)
    contradiction_detector = ContradictionDetector(embedding_engine, similarity_threshold=0.6)
    
    # Add some memories
    print("\nAdding memories...")
    
    # Memory set 1: Coherent memories about weather
    weather_memories = [
        "The sky is blue on a clear day.",
        "Rain falls from clouds in the sky.",
        "Sunlight makes the day warm.",
        "Clouds are made of water vapor."
    ]
    
    # Memory set 2: Coherent memories about animals
    animal_memories = [
        "Dogs are mammals that can be trained as pets.",
        "Cats are independent animals that often live with humans.",
        "Birds have feathers and can fly through the air.",
        "Fish live in water and breathe through gills."
    ]
    
    # Memory set 3: Contradictory memories
    contradictory_memories = [
        "Coffee contains caffeine.",
        "Coffee does not contain caffeine.",
        "The Earth orbits around the Sun.",
        "The Sun orbits around the Earth."
    ]
    
    # Add all memories with tags
    all_memory_ids = []
    
    for content in weather_memories:
        memory_id = memory_store.add(content, tags=["weather"], importance=1.0)
        all_memory_ids.append(memory_id)
    
    for content in animal_memories:
        memory_id = memory_store.add(content, tags=["animals"], importance=1.0)
        all_memory_ids.append(memory_id)
    
    for content in contradictory_memories:
        memory_id = memory_store.add(content, tags=["facts"], importance=1.0)
        all_memory_ids.append(memory_id)
    
    print(f"Added {len(all_memory_ids)} memories.")
    
    # Retrieve all memories as objects
    all_memories = [memory_store.get_memory(mid) for mid in all_memory_ids]
    
    # Calculate coherence
    print("\nCalculating coherence matrix...")
    coherence_matrix = coherence_scorer.calculate_coherence_matrix(all_memories)
    
    # Calculate global coherence
    global_coherence = coherence_scorer.calculate_global_coherence(all_memories)
    print(f"Global coherence: {global_coherence:.4f}")
    
    # Identify highest and lowest coherence pairs
    try:
        lowest_pair = coherence_scorer.get_lowest_coherence_pair(all_memories)
        print(f"\nLowest coherence pair ({lowest_pair[2]:.4f}):")
        print(f"  1. '{lowest_pair[0].content}'")
        print(f"  2. '{lowest_pair[1].content}'")
        
        highest_pair = coherence_scorer.get_highest_coherence_pair(all_memories)
        print(f"\nHighest coherence pair ({highest_pair[2]:.4f}):")
        print(f"  1. '{highest_pair[0].content}'")
        print(f"  2. '{highest_pair[1].content}'")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Find contradictions
    print("\nDetecting contradictions...")
    contradictions = contradiction_detector.find_contradictions(all_memories)
    
    if contradictions:
        print(f"Found {len(contradictions)} contradictions:")
        for i, (mem1, mem2, confidence, explanation) in enumerate(contradictions, 1):
            print(f"\n{i}. Contradiction (confidence: {confidence:.4f}):")
            print(f"   '{mem1.content}'")
            print(f"   '{mem2.content}'")
            print(f"   Explanation: {explanation}")
    else:
        print("No contradictions found.")
    
    # Export the memories to a file
    export_file = "memories_export.json"
    memory_store.export(export_file)
    print(f"\nExported memories to {export_file}")
    
    # Visualize coherence matrix (if matplotlib is available)
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(coherence_matrix, cmap='viridis')
        plt.colorbar(label='Coherence Score')
        plt.title('Memory Coherence Matrix')
        plt.savefig('coherence_matrix.png')
        print("Saved coherence matrix visualization to coherence_matrix.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 