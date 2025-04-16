#!/usr/bin/env python3
"""
Bounded Cognitive Runtime Demo

This example demonstrates the ComplexityController which implements a bounded
cognitive runtime that monitors and throttles ΨC operations to prevent
feedback loops, runaway recursion, and unbounded activation of high-tier features.
"""

import time
import random
import uuid
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple

from psi_c_ai_sdk.memory import MemoryStore, Memory, MemoryType
from psi_c_ai_sdk.schema import SchemaGraph
from psi_c_ai_sdk.coherence import BasicCoherenceScorer
from psi_c_ai_sdk.embedding import EmbeddingEngine
from psi_c_ai_sdk.reflection import ReflectionEngine, ReflectionScheduler
from psi_c_ai_sdk.runtime import ComplexityController, ComplexityTier, FeatureActivation


class DemoOperation:
    """Demo operation to show how complexity-controlled features work."""
    
    def __init__(self, name: str, feature: FeatureActivation, metric_range: Tuple[float, float]):
        """
        Initialize the demo operation.
        
        Args:
            name: Name of the operation
            feature: FeatureActivation type this operation represents
            metric_range: Range of metric values for this operation (min, max)
        """
        self.name = name
        self.feature = feature
        self.metric_range = metric_range
        self.attempts = 0
        self.successes = 0
    
    def get_random_metric(self) -> float:
        """Get a random metric value within this operation's range."""
        return random.uniform(self.metric_range[0], self.metric_range[1])


def create_random_memory(importance: float = None, coherence_value: float = None) -> Memory:
    """Create a random memory for testing."""
    topics = ["animals", "technology", "science", "art", "history", "food"]
    subtopics = {
        "animals": ["dogs", "cats", "birds", "fish", "mammals"],
        "technology": ["computers", "phones", "AI", "gadgets", "software"],
        "science": ["physics", "biology", "chemistry", "astronomy", "math"],
        "art": ["painting", "sculpture", "music", "literature", "photography"],
        "history": ["ancient", "medieval", "modern", "world wars", "civilization"],
        "food": ["recipes", "cuisine", "ingredients", "cooking", "restaurants"]
    }
    
    topic = random.choice(topics)
    subtopic = random.choice(subtopics[topic])
    
    if importance is None:
        importance = random.uniform(0.3, 1.0)
        
    # Generate coherent or incoherent content based on coherence_value
    if coherence_value is not None and coherence_value < 0.5:
        # Create more incoherent content
        content = f"Random fact about {random.choice(topics)} mixed with {random.choice(topics)}"
        tags = [random.choice(topics), random.choice(subtopics[random.choice(topics)])]
    else:
        # Create more coherent content
        content = f"Interesting fact about {subtopic} in the field of {topic}"
        tags = [topic, subtopic]
    
    return Memory(
        uuid=uuid.uuid4(),
        content=content,
        memory_type=MemoryType.FACT,
        importance=importance,
        tags=tags,
        source="demo"
    )


def simulate_operations(
    controller: ComplexityController,
    operations: List[DemoOperation],
    iterations: int = 10
) -> List[Dict[str, Any]]:
    """
    Simulate a series of operations with the complexity controller.
    
    Args:
        controller: The complexity controller to use
        operations: The operations to simulate
        iterations: Number of iterations to run
        
    Returns:
        List of complexity stats for each iteration
    """
    stats_history = []
    
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        print(f"Current complexity tier: {controller.get_current_tier().name}")
        
        # Try each operation
        for op in operations:
            metric = op.get_random_metric()
            op.attempts += 1
            
            print(f"  Attempting {op.name} (feature: {op.feature.name}, metric: {metric:.2f})...")
            if controller.activate_feature(op.feature, metric):
                op.successes += 1
                print(f"  ✅ {op.name} succeeded")
                
                # For memory storage, actually create and add a memory
                if op.feature == FeatureActivation.MEMORY_STORAGE:
                    memory = create_random_memory()
                    controller.memory_store.add_memory(memory)
                    print(f"    Added memory: {memory.content[:30]}...")
            else:
                print(f"  ❌ {op.name} blocked by complexity controller")
        
        # Get and store stats
        stats = controller.get_complexity_stats()
        stats_history.append(stats)
        
        # Display current stats
        print(f"Complexity score: {stats['complexity_score']:.2f}")
        print(f"Tier: {stats['current_tier']}")
        print(f"Feedback loop detected: {stats['feedback_loop_detected']}")
        print(f"Available energy: {stats['available_energy']:.2f}")
        
        # Decay recent activations between iterations
        controller.decay_recent_activations(0.7)
        
        # Small delay between iterations
        time.sleep(0.5)
    
    return stats_history


def plot_results(stats_history: List[Dict[str, Any]], operations: List[DemoOperation]):
    """Plot the results of the simulation."""
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Complexity score over time
    plt.subplot(2, 2, 1)
    complexity_scores = [stats["complexity_score"] for stats in stats_history]
    plt.plot(complexity_scores)
    plt.title("Complexity Score Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Complexity Score")
    
    # Add tier threshold lines
    for tier, threshold in sorted(stats_history[0]["activation_parameters"].items()):
        if tier in ["TIER_1", "TIER_2", "TIER_3"]:
            plt.axhline(y=threshold, linestyle="--", alpha=0.5, label=f"{tier} threshold")
    plt.legend()
    
    # Plot 2: Tier changes
    plt.subplot(2, 2, 2)
    tier_values = [list(ComplexityTier.__members__.keys()).index(stats["current_tier"]) 
                  for stats in stats_history]
    plt.step(range(len(tier_values)), tier_values, where="post")
    plt.yticks(range(len(ComplexityTier.__members__)), 
               [tier.name for tier in ComplexityTier])
    plt.title("Complexity Tier Over Time")
    plt.xlabel("Iteration")
    
    # Plot 3: Operation success rates
    plt.subplot(2, 2, 3)
    op_names = [op.name for op in operations]
    success_rates = [op.successes / max(1, op.attempts) * 100 for op in operations]
    
    # Color bars by tier
    colors = ["green", "blue", "orange", "red"]
    bar_colors = [colors[min(3, op.feature.value // 2)] for op in operations]
    
    plt.bar(op_names, success_rates, color=bar_colors)
    plt.title("Operation Success Rates")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 100)
    
    # Add tier labels
    for i, op in enumerate(operations):
        plt.text(i, 5, f"Tier {op.feature.value//2}", 
                 ha="center", color="white", fontweight="bold")
    
    # Plot 4: Available energy
    plt.subplot(2, 2, 4)
    available_energy = [stats["available_energy"] for stats in stats_history]
    plt.plot(available_energy)
    plt.title("Available Energy Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    
    plt.tight_layout()
    plt.savefig("complexity_controller_results.png")
    print("\nPlot saved as 'complexity_controller_results.png'")


def main():
    """Run the bounded cognitive runtime demo."""
    print("=== Bounded Cognitive Runtime Demo ===")
    
    # 1. Create core components
    print("\n1. Creating components...")
    memory_store = MemoryStore()
    embedding_engine = EmbeddingEngine()
    coherence_scorer = BasicCoherenceScorer()
    schema_graph = SchemaGraph(memory_store, coherence_scorer)
    
    reflection_scheduler = ReflectionScheduler(
        coherence_threshold=0.6,
        min_interval=10.0  # Shorter interval for demo purposes
    )
    
    reflection_engine = ReflectionEngine(
        memory_store=memory_store,
        scheduler=reflection_scheduler
    )
    
    # 2. Initialize complexity controller
    print("\n2. Initializing complexity controller...")
    controller = ComplexityController(
        memory_store=memory_store,
        schema_graph=schema_graph,
        reflection_engine=reflection_engine,
        # Use custom thresholds for faster demo
        tier_thresholds={
            ComplexityTier.TIER_0: 0.0,     # Always active
            ComplexityTier.TIER_1: 10.0,    # Lower threshold for demo
            ComplexityTier.TIER_2: 30.0,    # Lower threshold for demo
            ComplexityTier.TIER_3: 50.0,    # Lower threshold for demo
        },
        alpha_weights=(0.5, 1.0, 2.0),  # Weight contradictions more heavily
        min_interval=5.0,  # Shorter interval for demo
        history_window=5   # Smaller window for faster adaptation
    )
    
    # 3. Define operations for the demo
    print("\n3. Setting up operations...")
    operations = [
        # Tier 0 operations
        DemoOperation("Store memory", FeatureActivation.MEMORY_STORAGE, (0.3, 0.9)),
        DemoOperation("Track coherence", FeatureActivation.COHERENCE_TRACKING, (0.5, 0.9)),
        
        # Tier 1 operations
        DemoOperation("Reflection cycle", FeatureActivation.REFLECTION, (0.4, 0.8)),
        DemoOperation("Value decay", FeatureActivation.VALUE_DECAY, (0.3, 0.7)),
        
        # Tier 2 operations
        DemoOperation("Schema mutation", FeatureActivation.SCHEMA_MUTATION, (0.6, 0.9)),
        
        # Tier 3 operations
        DemoOperation("Identity modeling", FeatureActivation.IDENTITY_MODELING, (0.7, 1.0)),
        DemoOperation("Legacy creation", FeatureActivation.LEGACY_CREATION, (0.8, 1.0))
    ]
    
    # 4. Run the simulation
    print("\n4. Starting simulation...")
    stats_history = simulate_operations(
        controller=controller,
        operations=operations,
        iterations=20  # Run for 20 iterations
    )
    
    # 5. Display results
    print("\n5. Simulation complete. Generating charts...")
    plot_results(stats_history, operations)
    
    # 6. Display final operation statistics
    print("\n6. Operation statistics:")
    for op in operations:
        success_rate = (op.successes / max(1, op.attempts)) * 100
        print(f"  {op.name}: {op.successes}/{op.attempts} ({success_rate:.1f}% success rate)")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main() 