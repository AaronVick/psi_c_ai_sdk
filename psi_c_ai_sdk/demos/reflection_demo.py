#!/usr/bin/env python3
"""
Reflection Engine Demo

This script demonstrates the core reflection capabilities of the ΨC-AI SDK.
It shows how reflection cycles are triggered based on coherence thresholds,
entropy rates, and contradictions in the memory system.
"""

import time
import random
import numpy as np
from typing import List

from psi_c_ai_sdk.memory.memory import MemoryStore, Memory
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer
from psi_c_ai_sdk.reflection.reflection import (
    ReflectionEngine, ReflectionScheduler, ReflectionUtility, ReflectionTrigger
)


def generate_random_embedding(dim: int = 128) -> List[float]:
    """Generate a random embedding vector."""
    embedding = np.random.normal(0, 1, dim).tolist()
    return embedding


def create_sample_memories(count: int = 20) -> List[Memory]:
    """Create a set of sample memories for demonstration."""
    memories = []
    topics = ["science", "history", "art", "technology", "nature"]
    
    for i in range(count):
        # Select a random topic
        topic = random.choice(topics)
        
        # Create memory with tag matching the topic
        memory = Memory(
            content=f"Memory {i+1} about {topic}",
            embedding=generate_random_embedding(),
            importance=random.uniform(0.3, 1.0),
            metadata={"topic": topic}
        )
        memory.add_tag(topic)
        memories.append(memory)
    
    return memories


def create_coherent_memory_group(base_topic: str, count: int = 5) -> List[Memory]:
    """Create a group of memories that are coherent with each other."""
    memories = []
    base_embedding = np.random.normal(0, 1, 128)
    
    for i in range(count):
        # Create a slight variation of the base embedding
        noise = np.random.normal(0, 0.1, 128)
        embedding = (base_embedding + noise).tolist()
        
        memory = Memory(
            content=f"Coherent memory {i+1} about {base_topic}",
            embedding=embedding,
            importance=random.uniform(0.6, 1.0),
            metadata={"topic": base_topic, "group": "coherent"}
        )
        memory.add_tag(base_topic)
        memory.add_tag("coherent")
        memories.append(memory)
    
    return memories


def create_contradictory_memories() -> List[Memory]:
    """Create a pair of contradictory memories."""
    base_embedding = np.random.normal(0, 1, 128)
    opposite_embedding = (-1 * base_embedding).tolist()
    base_embedding = base_embedding.tolist()
    
    memory1 = Memory(
        content="The earth is flat",
        embedding=base_embedding,
        importance=0.8,
        metadata={"topic": "science", "group": "contradiction"}
    )
    memory1.add_tag("science")
    memory1.add_tag("contradictory")
    
    memory2 = Memory(
        content="The earth is spherical",
        embedding=opposite_embedding,
        importance=0.9,
        metadata={"topic": "science", "group": "contradiction"}
    )
    memory2.add_tag("science")
    memory2.add_tag("contradictory")
    
    return [memory1, memory2]


def on_reflection_complete(reflection_state):
    """Callback function when a reflection cycle completes."""
    print("\n" + "="*50)
    print(f"REFLECTION COMPLETE: {reflection_state.trace}")
    print(f"Trigger: {reflection_state.metadata.get('trigger', 'unknown')}")
    print(f"Coherence: {reflection_state.metadata.get('initial_coherence', 0):.2f} → {reflection_state.metadata.get('final_coherence', 0):.2f}")
    print(f"Entropy: {reflection_state.metadata.get('initial_entropy', 0):.2f} → {reflection_state.metadata.get('final_entropy', 0):.2f}")
    print(f"Contradictions found: {len(reflection_state.contradictions)}")
    print(f"Memories processed: {reflection_state.metadata.get('memory_count', 0)}")
    print("="*50 + "\n")


def main():
    print("ΨC-AI SDK: Reflection Engine Demo")
    print("=" * 40)
    
    # Initialize the memory store
    memory_store = MemoryStore()
    
    # Initialize the coherence scorer
    coherence_scorer = CoherenceScorer()
    
    # Create a reflection scheduler with custom parameters for the demo
    # Use shorter intervals for demonstration purposes
    reflection_scheduler = ReflectionScheduler(
        coherence_threshold=0.6,  # Trigger reflection when coherence drops below 0.6
        entropy_threshold=0.2,    # Trigger when entropy rate exceeds 0.2
        min_interval=10.0,        # Min 10 seconds between reflections
        max_interval=60.0         # Max 60 seconds between reflections
    )
    
    # Create the reflection engine
    reflection_engine = ReflectionEngine(
        memory_store=memory_store,
        coherence_scorer=coherence_scorer,
        reflection_scheduler=reflection_scheduler,
        utility_threshold=0.3     # Lower threshold for demo purposes
    )
    
    # Register the callback
    reflection_engine.on_reflection_complete = on_reflection_complete
    
    # === DEMO SCENARIO 1: Base Memories ===
    print("\nScenario 1: Adding initial random memories...")
    initial_memories = create_sample_memories(15)
    for memory in initial_memories:
        memory_store.add_memory(memory)
        print(f"Added memory: {memory.content} (importance: {memory.importance:.2f})")
    
    # Update reflection engine to process initial state
    print("\nChecking if reflection is needed with initial memories...")
    reflected = reflection_engine.update()
    print(f"Reflection performed: {reflected}")
    
    # === DEMO SCENARIO 2: Coherent Group ===
    print("\nScenario 2: Adding a coherent group of memories about physics...")
    coherent_group = create_coherent_memory_group("physics", 5)
    for memory in coherent_group:
        memory_store.add_memory(memory)
        print(f"Added coherent memory: {memory.content}")
    
    # Update again after adding coherent memories
    print("\nChecking reflection after adding coherent memories...")
    reflected = reflection_engine.update()
    print(f"Reflection performed: {reflected}")
    
    # === DEMO SCENARIO 3: Contradictions ===
    print("\nScenario 3: Adding contradictory memories...")
    contradictory_memories = create_contradictory_memories()
    for memory in contradictory_memories:
        memory_store.add_memory(memory)
        print(f"Added contradictory memory: {memory.content}")
    
    # This should trigger a reflection due to low coherence
    print("\nChecking reflection after adding contradictions...")
    reflected = reflection_engine.update()
    print(f"Reflection performed: {reflected}")
    
    # === DEMO SCENARIO 4: Manual Reflection ===
    print("\nScenario 4: Forcing a manual reflection cycle...")
    reflection_state = reflection_engine.force_reflection()
    print(f"Manual reflection completed with {len(reflection_state.contradictions)} contradictions")
    
    # === DEMO SCENARIO 5: Monitor Over Time ===
    print("\nScenario 5: Monitoring reflection cycles over time...")
    print("Running for 60 seconds, adding new memories periodically...")
    
    start_time = time.time()
    interval = 5  # Add new memories every 5 seconds
    next_addition = start_time + interval
    
    # Run for 60 seconds
    while time.time() < start_time + 60:
        current_time = time.time()
        
        # Add new memories at intervals
        if current_time >= next_addition:
            topic = random.choice(["science", "history", "philosophy"])
            
            # Randomly decide to add a normal or contradictory memory
            if random.random() < 0.3:  # 30% chance of contradiction
                # Add a contradictory memory
                base_memory = random.choice(memory_store.get_all_memories())
                
                # Create an opposite embedding
                if base_memory.embedding:
                    opposite_embedding = [-x for x in base_memory.embedding]
                    
                    # Create memory with opposing content
                    content = f"Contradicting {base_memory.content}"
                    memory = Memory(
                        content=content,
                        embedding=opposite_embedding,
                        importance=random.uniform(0.5, 1.0),
                        metadata={"topic": topic, "contradicts": base_memory.uuid}
                    )
                    memory_store.add_memory(memory)
                    print(f"Added contradicting memory: {content}")
            else:
                # Add a normal memory
                memory = Memory(
                    content=f"New memory about {topic} at {int(current_time)}",
                    embedding=generate_random_embedding(),
                    importance=random.uniform(0.3, 1.0),
                    metadata={"topic": topic, "timestamp": current_time}
                )
                memory_store.add_memory(memory)
                print(f"Added normal memory: {memory.content}")
            
            next_addition = current_time + interval
        
        # Update the reflection engine
        reflected = reflection_engine.update()
        if not reflected:
            print(".", end="", flush=True)
            
        # Small delay to prevent CPU overuse
        time.sleep(0.5)
    
    print("\n\nDemo completed! Final statistics:")
    stats = reflection_engine.get_stats()
    print(f"Total reflection cycles: {stats['reflection_count']}")
    print(f"Average coherence improvement: {stats['avg_coherence_change']:.4f}")
    print(f"Total contradictions detected: {stats['contradiction_count']}")
    
    # Show recent reflection history
    print("\nRecent reflection history:")
    history = reflection_engine.get_reflection_history(limit=3)
    for i, state in enumerate(history):
        print(f"{i+1}. {state.trace} - Coherence: {state.coherence_score:.2f}")


if __name__ == "__main__":
    main() 