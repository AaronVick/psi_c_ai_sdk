#!/usr/bin/env python3
"""
Reflection Credit System Demo

This script demonstrates the use of the ReflectionCreditSystem to manage cognitive
resources for reflection operations in the ΨC-AI SDK.
"""

import time
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging

from psi_c_ai_sdk.memory import MemoryStore, Memory, MemoryType
from psi_c_ai_sdk.coherence import BasicCoherenceScorer
from psi_c_ai_sdk.embedding import EmbeddingEngine
from psi_c_ai_sdk.reflection import (
    ReflectionEngine,
    ReflectionScheduler,
    ReflectionTrigger,
    ReflectionCreditSystem,
    ReflectionOutcome,
    calculate_cognitive_debt
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("reflection_credit_demo")


def create_test_memory(content: str, importance: float, tags: List[str]) -> Memory:
    """
    Create a test memory with the given content and properties.
    
    Args:
        content: Memory content
        importance: Memory importance (0.0-1.0)
        tags: List of tags for the memory
        
    Returns:
        Memory object
    """
    return Memory(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=importance,
        tags=tags,
        source="reflection_credit_demo"
    )


def setup_system():
    """
    Set up the memory store, credit system, and reflection engine.
    
    Returns:
        Tuple of (memory_store, reflection_engine, credit_system)
    """
    # Initialize components
    memory_store = MemoryStore()
    embedding_engine = EmbeddingEngine()
    coherence_scorer = BasicCoherenceScorer()
    
    # Create reflection scheduler with shorter intervals for demo
    scheduler = ReflectionScheduler(
        coherence_threshold=0.6,  # Trigger reflection below this coherence
        entropy_threshold=0.3,    # Trigger reflection above this entropy
        min_interval=5.0,         # 5 seconds between reflections (for demo)
        max_interval=15.0,        # Maximum 15 seconds between reflections
        max_memory_count=50,      # Trigger reflection after 50 memories
        memory_threshold=10,      # At least 10 memories between reflections
        time_threshold=30.0       # Allow reflection after 30 seconds
    )
    
    # Create credit system with demo parameters
    credit_system = ReflectionCreditSystem(
        initial_credit=100.0,           # Start with 100 credits
        max_credit=150.0,               # Maximum possible credit
        min_credit=-50.0,               # Allow up to 50 credits of debt
        base_reflection_cost=20.0,      # Each reflection costs 20 credits
        credit_regeneration_rate=5.0,   # Regenerate 5 credits per minute
        max_debt_ratio=0.3,             # Don't allow debt > 30% of max credit
        cooldown_base_period=10.0,      # 10-second base cooldown (for demo)
        effectiveness_history_window=5  # Consider last 5 reflections for stats
    )
    
    # Create reflection engine
    reflection_engine = ReflectionEngine(
        memory_store=memory_store,
        coherence_scorer=coherence_scorer,
        scheduler=scheduler,
        max_reflections_history=10,
        on_reflection_complete=lambda state: logger.info(
            f"Reflection complete: {state.id}, success: {state.successful}, "
            f"coherence improvement: {state.coherence_improvement:.4f}"
        ),
        credit_system=credit_system
    )
    
    return memory_store, reflection_engine, credit_system


def run_demo_scenario(memory_store, reflection_engine, credit_system):
    """
    Run a demonstration scenario that shows credit system in action.
    
    Args:
        memory_store: Memory store
        reflection_engine: Reflection engine
        credit_system: Credit system
    """
    logger.info("Starting reflection credit system demo")
    logger.info(f"Initial credit: {credit_system.get_available_credit():.1f}")
    
    # Create some memories of varying coherence
    memories = [
        # Coherent memories about AI
        ("AI systems use neural networks to process data.", 0.8, ["ai", "coherent"]),
        ("Machine learning models require training data to learn patterns.", 0.8, ["ai", "coherent"]),
        ("Deep learning is a subset of machine learning using multiple neural network layers.", 0.9, ["ai", "coherent"]),
        
        # Some incoherent memories
        ("Some computer things do stuff with numbers and code.", 0.4, ["tech", "incoherent"]),
        ("Programs are built with languages that computers understand.", 0.6, ["tech", "incoherent"]),
        
        # Contradictory memories
        ("Neural networks can only solve simple mathematical problems.", 0.5, ["ai", "contradictory"]),
        ("Machine learning requires no data to function properly.", 0.3, ["ai", "contradictory"]),
        
        # More coherent memories on a different topic
        ("The Earth orbits the Sun in an elliptical path.", 0.8, ["science", "coherent"]),
        ("Gravity is a fundamental force that attracts objects with mass.", 0.9, ["science", "coherent"]),
        ("The solar system contains eight planets orbiting the Sun.", 0.8, ["science", "coherent"]),
        
        # More contradictions
        ("The Earth is the center of the universe and everything orbits it.", 0.4, ["science", "contradictory"]),
        ("Gravity is just a theory with no evidence supporting it.", 0.3, ["science", "contradictory"])
    ]
    
    # Track metrics
    metrics = []
    
    # Add memories one by one with a small delay
    for i, (content, importance, tags) in enumerate(memories):
        # Create and add memory
        memory = create_test_memory(content, importance, tags)
        memory_store.add_memory(memory)
        
        logger.info(f"Added memory {i+1}/{len(memories)}: {content[:40]}...")
        
        # Check if reflection is needed
        should_reflect, trigger = reflection_engine.check_reflection_needed()
        
        # Collect metrics before potential reflection
        credit = credit_system.get_available_credit()
        coherence = reflection_engine.coherence_scorer.calculate_global_coherence(memory_store)
        
        metrics.append({
            "memory_count": i+1,
            "credit": credit,
            "coherence": coherence,
            "reflected": False
        })
        
        # Perform reflection if needed or force at specific points
        if should_reflect or (i+1) % 5 == 0:  # Force reflection every 5 memories for demo
            if should_reflect:
                logger.info(f"Reflection triggered: {trigger}")
            else:
                logger.info("Forcing scheduled reflection for demo")
                
            # Check if we can reflect according to credit system
            can_reflect, reason = credit_system.can_reflect()
            
            if can_reflect:
                reflection_id = reflection_engine.start_reflection_cycle(
                    trigger=ReflectionTrigger.SCHEDULED if not should_reflect else trigger
                )
                
                # Update metrics for this step to show reflection happened
                metrics[-1]["reflected"] = True
                metrics[-1]["reflection_id"] = reflection_id
                
                # Collect post-reflection metrics
                credit_after = credit_system.get_available_credit()
                coherence_after = reflection_engine.coherence_scorer.calculate_global_coherence(memory_store)
                
                metrics[-1]["credit_after"] = credit_after
                metrics[-1]["coherence_after"] = coherence_after
                metrics[-1]["credit_change"] = credit_after - credit
                metrics[-1]["coherence_change"] = coherence_after - coherence
                
                logger.info(f"Credit after reflection: {credit_after:.1f} (change: {credit_after - credit:.1f})")
            else:
                logger.info(f"Reflection denied by credit system: {reason}")
                
        # Small delay between adding memories
        time.sleep(1)
        
    # Show final credit system stats
    logger.info("\nFinal Credit System Stats:")
    stats = credit_system.get_credit_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
        
    # Calculate cognitive debt
    debt = calculate_cognitive_debt(credit_system, reflection_engine.reflection_history)
    logger.info("\nCognitive Debt Metrics:")
    for key, value in debt.items():
        logger.info(f"  {key}: {value}")
        
    # Return metrics for visualization
    return metrics


def visualize_results(metrics):
    """
    Visualize the results of the demo scenario.
    
    Args:
        metrics: List of metric dictionaries from the demo run
    """
    plt.figure(figsize=(12, 8))
    
    # Plot credit and coherence over time
    memory_counts = [m["memory_count"] for m in metrics]
    credits = [m["credit"] for m in metrics]
    coherence = [m["coherence"] for m in metrics]
    
    reflection_points = [i for i, m in enumerate(metrics) if m.get("reflected", False)]
    
    # Set up subplot for credit
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(memory_counts, credits, 'b-', label='Credit')
    ax1.set_ylabel('Credit')
    ax1.set_title('Reflection Credit System Demo Results')
    ax1.grid(True)
    
    # Mark reflection points on credit graph
    for point in reflection_points:
        ax1.axvline(x=memory_counts[point], color='r', linestyle='--', alpha=0.5)
        
    # Add credit change annotations
    for i in reflection_points:
        if "credit_change" in metrics[i]:
            change = metrics[i]["credit_change"]
            ax1.annotate(f"{change:.1f}", 
                        (memory_counts[i], credits[i]),
                        xytext=(5, 10), 
                        textcoords='offset points',
                        color='red')
    
    # Set up subplot for coherence
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(memory_counts, coherence, 'g-', label='Coherence')
    ax2.set_xlabel('Memory Count')
    ax2.set_ylabel('Coherence')
    ax2.grid(True)
    
    # Mark reflection points on coherence graph
    for point in reflection_points:
        ax2.axvline(x=memory_counts[point], color='r', linestyle='--', alpha=0.5)
        
    # Add coherence change annotations
    for i in reflection_points:
        if "coherence_change" in metrics[i]:
            change = metrics[i]["coherence_change"]
            ax2.annotate(f"{change:.2f}", 
                        (memory_counts[i], coherence[i]),
                        xytext=(5, 10), 
                        textcoords='offset points',
                        color='green' if change > 0 else 'red')
    
    plt.tight_layout()
    plt.savefig("reflection_credit_demo_results.png")
    logger.info("Results visualization saved to 'reflection_credit_demo_results.png'")
    plt.close()


def main():
    """Main function to run the demo"""
    # Set up the system
    memory_store, reflection_engine, credit_system = setup_system()
    
    # Run demo scenario
    metrics = run_demo_scenario(memory_store, reflection_engine, credit_system)
    
    # Visualize results
    visualize_results(metrics)
    
    logger.info("Demo complete!")


if __name__ == "__main__":
    main() 