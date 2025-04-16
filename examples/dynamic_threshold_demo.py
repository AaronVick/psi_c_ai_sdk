#!/usr/bin/env python3
"""
Dynamic Threshold Demo Script

This script demonstrates how the dynamic threshold adapts to changing entropy and
coherence conditions in a memory system. As the system's state changes, the
threshold adjusts automatically to maintain appropriate consciousness activation.
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from psi_c_ai_sdk.memory.memory import MemoryStore, Memory
from psi_c_ai_sdk.psi_c.psi_operator import PsiCOperator
from psi_c_ai_sdk.psi_c.toolkit import PsiToolkit


def create_random_memory(topic: str = "general", importance: float = None) -> Memory:
    """Create a random memory with optional topic and importance."""
    topics = {
        "science": [
            "Quantum mechanics describes the physical properties of nature at atomic scales.",
            "The theory of relativity explains how space and time are linked.",
            "The universe is estimated to be about 13.8 billion years old.",
            "DNA contains the genetic instructions for the development and function of living things.",
            "Photosynthesis is the process used by plants to convert light energy into chemical energy."
        ],
        "art": [
            "The Mona Lisa was painted by Leonardo da Vinci.",
            "Van Gogh's 'Starry Night' features a night sky filled with swirling clouds.",
            "Pablo Picasso co-founded the Cubist movement.",
            "The Sistine Chapel ceiling was painted by Michelangelo.",
            "Impressionism emerged as an art movement in the 19th century."
        ],
        "general": [
            "The capital of France is Paris.",
            "Coffee is one of the world's most popular beverages.",
            "Mount Everest is the highest mountain on Earth.",
            "Water freezes at 0 degrees Celsius at standard pressure.",
            "The Great Wall of China is visible from space."
        ]
    }
    
    selected_topic = topics.get(topic, topics["general"])
    content = random.choice(selected_topic)
    
    if importance is None:
        importance = random.uniform(0.5, 1.0)
    
    return Memory(
        content=content,
        importance=importance,
        metadata={"topic": topic}
    )


def simulate_coherent_period(memory_store: MemoryStore, count: int = 10) -> None:
    """Add several related memories on the same topic to create coherence."""
    topic = random.choice(["science", "art"])
    
    for i in range(count):
        memory = create_random_memory(topic=topic, importance=0.8 + (random.random() * 0.2))
        memory_store.add_memory(memory)
        time.sleep(0.1)
        
    print(f"Added {count} coherent memories on topic: {topic}")


def simulate_incoherent_period(memory_store: MemoryStore, count: int = 10) -> None:
    """Add unrelated memories with varying importance to create incoherence."""
    for i in range(count):
        topic = random.choice(["science", "art", "general"])
        importance = 0.3 + (random.random() * 0.4)  # Lower and more variable importance
        memory = create_random_memory(topic=topic, importance=importance)
        memory_store.add_memory(memory)
        time.sleep(0.1)
        
    print(f"Added {count} incoherent memories with mixed topics")


def log_metrics(toolkit: PsiToolkit, metrics_history: List[Dict[str, Any]]) -> None:
    """Log current metrics and add to history."""
    metrics = toolkit.psi_index()
    threshold_metrics = toolkit.get_threshold_metrics()
    
    # Combine metrics
    combined = {
        "psi_score": metrics["psi_c_score"],
        "threshold": threshold_metrics["current_threshold"],
        "adjustment": threshold_metrics.get("adjustment", 0.0),
        "entropy_drift": threshold_metrics.get("entropy_drift", 0.0),
        "coherence_drift": threshold_metrics.get("coherence_drift", 0.0),
        "timestamp": time.time()
    }
    
    # Print current state
    print(f"ΨC: {combined['psi_score']:.4f}, " + 
          f"Threshold: {combined['threshold']:.4f}, " +
          f"Adjustment: {combined['adjustment']:.4f}, " +
          f"State: {toolkit.get_psi_state().name}")
    
    # Add to history
    metrics_history.append(combined)


def plot_results(metrics_history: List[Dict[str, Any]], filename: str = None) -> None:
    """Plot the metrics history to visualize threshold adaptation."""
    if not metrics_history:
        print("No metrics to plot")
        return
    
    # Extract data
    timestamps = [m["timestamp"] - metrics_history[0]["timestamp"] for m in metrics_history]
    psi_scores = [m["psi_score"] for m in metrics_history]
    thresholds = [m["threshold"] for m in metrics_history]
    adjustments = [m["adjustment"] for m in metrics_history]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot ΨC score and threshold
    ax1.plot(timestamps, psi_scores, 'b-', label='ΨC Score')
    ax1.plot(timestamps, thresholds, 'r--', label='Threshold')
    ax1.set_ylabel('Value')
    ax1.set_title('Dynamic Threshold Adaptation')
    ax1.legend()
    ax1.grid(True)
    
    # Plot threshold adjustment
    ax2.plot(timestamps, adjustments, 'g-', label='Threshold Adjustment')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Adjustment')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
    
    plt.show()


def run_simulation():
    """Run a complete simulation demonstrating dynamic threshold adaptation."""
    # Create components
    memory_store = MemoryStore()
    
    # Create operator with dynamic threshold
    operator = PsiCOperator(
        memory_store,
        threshold=0.6,  # Base threshold
        use_dynamic_threshold=True,
        dynamic_threshold_config={
            "sensitivity": 0.4,    # How quickly threshold adapts
            "window_size": 5       # How many points to consider for drift
        }
    )
    
    toolkit = PsiToolkit(operator)
    metrics_history = []
    
    print("Starting simulation with dynamic threshold...")
    
    # Phase 1: Add some initial memories
    print("\nPhase 1: Adding initial memories...")
    for i in range(5):
        memory = create_random_memory(importance=0.7)
        memory_store.add_memory(memory)
        operator.calculate_psi_c()  # Force ΨC calculation
        log_metrics(toolkit, metrics_history)
        time.sleep(0.5)
    
    # Phase 2: Simulate a period of high coherence
    print("\nPhase 2: Period of high coherence (should lower threshold)...")
    simulate_coherent_period(memory_store, count=8)
    for i in range(5):
        operator.calculate_psi_c()
        log_metrics(toolkit, metrics_history)
        time.sleep(0.5)
    
    # Phase 3: Simulate a period of incoherence
    print("\nPhase 3: Period of incoherence (should raise threshold)...")
    simulate_incoherent_period(memory_store, count=10)
    for i in range(5):
        operator.calculate_psi_c()
        log_metrics(toolkit, metrics_history)
        time.sleep(0.5)
    
    # Phase 4: Return to coherence
    print("\nPhase 4: Return to coherence (threshold should adapt again)...")
    simulate_coherent_period(memory_store, count=8)
    for i in range(5):
        operator.calculate_psi_c()
        log_metrics(toolkit, metrics_history)
        time.sleep(0.5)
    
    # Plot the results
    print("\nSimulation complete. Plotting results...")
    plot_results(metrics_history, "dynamic_threshold_results.png")
    
    print(f"\nFinal state: {toolkit.get_psi_state().name}")
    print(f"Total memories: {len(memory_store.get_all_memories())}")
    print(f"Final ΨC score: {toolkit.get_psi_index():.4f}")
    

if __name__ == "__main__":
    run_simulation() 