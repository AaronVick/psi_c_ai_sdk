#!/usr/bin/env python
"""
PsiToolkit Demo: Demonstrates the use of the ΨC toolkit.

This example shows how to use the PsiToolkit to monitor consciousness state,
log activation events, and simulate collapse events.
"""

import time
import random
import logging
from typing import List

from psi_c_ai_sdk.memory import MemoryStore, Memory, MemoryType
from psi_c_ai_sdk.coherence import BasicCoherenceScorer
from psi_c_ai_sdk.psi_c import PsiCOperator, PsiToolkit, CollapseSimulator, PsiCState


def add_random_memories(memory_store: MemoryStore, count: int = 5) -> None:
    """Add random memories to the memory store to simulate activity."""
    topics = ["quantum physics", "consciousness", "AI", "philosophy", "neuroscience"]
    
    for i in range(count):
        topic = random.choice(topics)
        # Create related memories for coherence
        memory = Memory(
            content=f"Thought about {topic} concept #{random.randint(1, 100)}",
            memory_type=MemoryType.DECLARATIVE,
            importance=random.random() * 0.5 + 0.5,  # Higher importance (0.5-1.0)
            metadata={
                "topic": topic,
                "confidence": random.random(),
                "is_reflection": random.random() > 0.7  # Some are reflective
            }
        )
        memory_store.add_memory(memory)
        
    print(f"Added {count} new memories")


def main() -> None:
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up the basic components
    print("Setting up ΨC system...")
    memory_store = MemoryStore()
    coherence_scorer = BasicCoherenceScorer()
    
    # Create the ΨC operator with lower threshold for demo purposes
    psi_operator = PsiCOperator(
        memory_store=memory_store,
        threshold=0.7,  # Lower threshold for demo
        window_size=10  # Smaller window for faster response
    )
    
    # Create collapse simulator
    simulator = CollapseSimulator(
        psi_operator=psi_operator,
        deviation_strength=0.3,  # Moderate deviation strength
        pseudo_rng_seed=42  # For reproducibility
    )
    
    # Create the toolkit
    toolkit = PsiToolkit(
        psi_operator=psi_operator,
        collapse_simulator=simulator
    )
    
    print("\n--- Initial State ---")
    print(f"ΨC Index: {toolkit.get_psi_index():.4f}")
    print(f"ΨC State: {toolkit.get_psi_state()}")
    print(f"Conscious: {toolkit.is_conscious()}")
    
    # Simulate memory activity to build coherence
    print("\n--- Adding Memories ---")
    for i in range(5):
        print(f"\nIteration {i+1}")
        # Add some related memories
        add_random_memories(memory_store, count=3)
        
        # Update operator and show status
        psi_operator.update()
        
        print(f"ΨC Index: {toolkit.get_psi_index():.4f}")
        print(f"State: {toolkit.get_psi_state()}")
        
        # Once we have some consciousness, simulate collapse events
        if toolkit.get_psi_index() > 0.3:
            event = toolkit.simulate_collapse_event(num_outcomes=2)
            print(f"Collapse Event: outcome={event.outcome}, deviation={event.deviation:.4f}")
        
        # Display health metrics
        health = toolkit.get_coherence_health()
        print(f"Coherence: {health['coherence']:.4f}, Trend: {health['trend']}")
        
        # Pause to see changes
        time.sleep(1)
    
    # Display activation log
    print("\n--- Activation Log ---")
    log = toolkit.get_activation_log()
    for event in log[:3]:  # Show most recent 3
        print(f"Event: {event['event_type']}, "
              f"State: {event['old_state']} -> {event['new_state']}, "
              f"Score: {event['psi_score']:.4f}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main() 