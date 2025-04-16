#!/usr/bin/env python3
"""
Example demonstrating the Reflection Engine in the ΨC-AI SDK.

This example shows how to initialize and use the reflection engine
to improve memory coherence and consolidation in a cognitive system.
"""
import time
import random
from psi_c_ai_sdk.memory import MemoryStore, Memory, MemoryType
from psi_c_ai_sdk.reflection import ReflectionEngine, ReflectionScheduler, ReflectionTrigger

# Initialize a memory store with some sample memories
memory_store = MemoryStore()

# Add some example memories with contradictions
memory_store.add(Memory(
    content="The sky is blue because of Rayleigh scattering",
    type=MemoryType.SEMANTIC,
    importance=0.8,
    metadata={"topic": "meteorology", "confidence": 0.9}
))

memory_store.add(Memory(
    content="The sky appears blue due to the reflection of the ocean",
    type=MemoryType.SEMANTIC,
    importance=0.7,
    metadata={"topic": "meteorology", "confidence": 0.6}
))

memory_store.add(Memory(
    content="I observed a red sunset yesterday",
    type=MemoryType.EPISODIC,
    importance=0.5,
    metadata={"time": "2023-09-15T18:30:00", "location": "home"}
))

memory_store.add(Memory(
    content="Sunsets are red because of atmospheric scattering of light",
    type=MemoryType.SEMANTIC,
    importance=0.6,
    metadata={"topic": "meteorology", "confidence": 0.8}
))

# Initialize the reflection components
scheduler = ReflectionScheduler(
    coherence_threshold=0.7,
    entropy_threshold=0.3,
    time_threshold=3600,  # 1 hour
    memory_count_threshold=10
)

reflection_engine = ReflectionEngine(
    memory_store=memory_store,
    scheduler=scheduler,
    coherence_weight=0.6,
    consolidation_threshold=0.5
)

# Simulate some time passing and new observations
print("Initial memory state:")
for memory in memory_store.memories:
    print(f" - {memory.content} (Importance: {memory.importance:.2f})")

print("\nSimulating cognitive activity...\n")
for i in range(3):
    # Simulate the passage of time and addition of new memories
    time.sleep(1)
    
    # Add a random memory
    topics = ["weather", "science", "astronomy", "geography"]
    memory_store.add(Memory(
        content=f"New observation about {random.choice(topics)}",
        type=MemoryType.EPISODIC,
        importance=random.random(),
        metadata={"time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    ))
    
    # Manually trigger reflection (in a real system this would happen automatically)
    print(f"Iteration {i+1}: Checking if reflection is needed...")
    if reflection_engine.should_reflect():
        print("Reflection cycle triggered!")
        reflection_state = reflection_engine.reflect()
        print(f"Reflection complete. Coherence improved: {reflection_state.coherence_after - reflection_state.coherence_before:.2f}")
        print(f"Contradictions resolved: {len(reflection_state.resolved_contradictions)}")
        print(f"Memories consolidated: {len(reflection_state.consolidated_memories)}")
    else:
        print("No reflection needed at this time.")
    
    print()

print("Final memory state:")
for memory in memory_store.memories:
    print(f" - {memory.content} (Importance: {memory.importance:.2f})")

# Force a reflection with a specific trigger
print("\nForcing a manual reflection cycle...")
reflection_state = reflection_engine.reflect(trigger=ReflectionTrigger.MANUAL)
print(f"Manual reflection complete. Actions taken: {reflection_state.actions_taken}") 