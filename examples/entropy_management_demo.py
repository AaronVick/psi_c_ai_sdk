#!/usr/bin/env python
"""
Entropy Management System Demo

This demo showcases the entropy measurement, monitoring, and pruning functionality
in the ΨC-AI SDK. It demonstrates:

1. How to measure entropy in embeddings and memory systems
2. How to monitor entropy levels over time
3. How to use entropy-based pruning to maintain system coherence
4. How the entropy management system can influence termination decisions
"""

import time
import random
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.entropy.entropy import (
    EntropyCalculator, 
    EmbeddingEntropyMeasure,
    SemanticCoherenceEntropyMeasure
)
from psi_c_ai_sdk.entropy.monitor import (
    EntropyMonitor, 
    EntropyThresholdSettings, 
    EntropyAlertLevel
)
from psi_c_ai_sdk.entropy.pruning import EntropyBasedPruner
from psi_c_ai_sdk.coherence.scorer import BasicCoherenceScorer
from psi_c_ai_sdk.embedding.engine import EmbeddingEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample data for creating memories
SAMPLE_MEMORIES = [
    # Coherent group about animals
    "Dogs are loyal pets that many people keep as companions.",
    "Cats are independent animals known for their grooming habits.",
    "Elephants are the largest land mammals with impressive memory capabilities.",
    "Lions are apex predators often called the kings of the jungle.",
    
    # Coherent group about technology
    "Artificial intelligence is revolutionizing various industries.",
    "Machine learning algorithms can identify patterns in large datasets.",
    "Neural networks are designed to mimic the human brain's structure.",
    "Deep learning has enabled breakthroughs in image and speech recognition.",
    
    # Incoherent/higher entropy memories
    "The square root of 841 is exactly 29 which is a prime number.",
    "Bananas taste better when listening to classical music on Thursdays.",
    "Purple elephants dance silently when nobody is watching their shadows.",
    "The concept of time is both linear and circular depending on perspective."
]

def generate_random_embedding(dim: int = 384) -> List[float]:
    """Generate a random embedding vector of the specified dimension."""
    return list(np.random.normal(0, 1, dim).astype(float))

def create_memory_store() -> MemoryStore:
    """Create and populate a memory store with sample memories."""
    memory_store = MemoryStore()
    embedding_engine = EmbeddingEngine()
    
    # Add memories with embeddings
    for i, content in enumerate(SAMPLE_MEMORIES):
        # For this demo, we'll create embeddings directly
        # In a real application, you would use the embedding engine
        try:
            embedding = embedding_engine.embed_text(content)
        except:
            # Fallback to random embeddings if the engine fails
            embedding = generate_random_embedding()
            
        # Create memory with varying importance
        memory = Memory(
            content=content,
            importance=random.uniform(0.1, 0.9),
            metadata={"source": "demo", "index": i}
        )
        memory.embedding = embedding
        memory_store.add_memory(memory)
        
    return memory_store

def add_high_entropy_memories(memory_store: MemoryStore, count: int = 5) -> None:
    """Add some high-entropy memories to the store."""
    embedding_engine = EmbeddingEngine()
    
    for i in range(count):
        # Generate random incoherent text
        words = [
            random.choice([
                "quantum", "banana", "philosophy", "paradox", "elephant",
                "cryptocurrency", "dimension", "purple", "theory", "mountain"
            ]) for _ in range(8)
        ]
        content = " ".join(words)
        
        try:
            embedding = embedding_engine.embed_text(content)
        except:
            # Fallback to random embeddings
            embedding = generate_random_embedding()
            # Make it higher entropy by making it more extreme
            embedding = [x * 1.5 for x in embedding]
            
        memory = Memory(
            content=content,
            importance=random.uniform(0.1, 0.3),  # Lower importance
            metadata={"source": "high_entropy_demo", "index": i}
        )
        memory.embedding = embedding
        memory_store.add_memory(memory)
        
def plot_entropy_over_time(entropy_history: List[Dict[str, Any]]) -> None:
    """Plot entropy levels and alert thresholds over time."""
    timestamps = [entry["timestamp"] for entry in entropy_history]
    entropy_values = [entry["entropy"] for entry in entropy_history]
    alert_levels = [entry["alert_level"] for entry in entropy_history]
    
    # Convert timestamps to datetime objects
    datetime_objects = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    # Set up colors for each alert level
    colors = {
        "normal": "green",
        "warning": "yellow",
        "critical": "orange",
        "emergency": "red"
    }
    
    point_colors = [colors[level] for level in alert_levels]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(datetime_objects, entropy_values, c=point_colors)
    plt.plot(datetime_objects, entropy_values, 'b-', alpha=0.3)
    
    # Add threshold lines
    if entropy_history:
        plt.axhline(y=0.7, color='yellow', linestyle='--', label='Warning Threshold')
        plt.axhline(y=0.8, color='orange', linestyle='--', label='Critical Threshold')
        plt.axhline(y=0.9, color='red', linestyle='--', label='Emergency Threshold')
    
    plt.title('Memory System Entropy Over Time')
    plt.xlabel('Time')
    plt.ylabel('Entropy')
    plt.legend()
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    plt.tight_layout()
    plt.savefig('entropy_over_time.png')
    logger.info(f"Entropy plot saved as 'entropy_over_time.png'")

def on_alert_change(alert_level: EntropyAlertLevel, metrics: Dict[str, Any]) -> None:
    """Callback function triggered when the entropy alert level changes."""
    logger.warning(f"ALERT LEVEL CHANGED TO: {alert_level}")
    logger.warning(f"Current entropy: {metrics['entropy']:.4f}")
    
    if alert_level == EntropyAlertLevel.WARNING:
        logger.warning("Entropy is increasing. Consider monitoring more closely.")
    elif alert_level == EntropyAlertLevel.CRITICAL:
        logger.warning("Entropy has reached critical levels. System stability may be compromised.")
    elif alert_level == EntropyAlertLevel.EMERGENCY:
        logger.warning("EMERGENCY: Entropy levels are dangerous. System may require intervention.")

def on_termination_decision(metrics: Dict[str, Any]) -> bool:
    """
    Callback function triggered when a termination decision is made.
    Return True to proceed with termination, False to override.
    """
    logger.critical(f"TERMINATION DECISION REQUIRED")
    logger.critical(f"Reason: {metrics['termination_reason']}")
    logger.critical(f"Current entropy: {metrics['entropy']:.4f}")
    
    # In this demo, we'll override the termination to keep the demo running
    logger.critical("TERMINATION OVERRIDDEN FOR DEMO PURPOSES")
    return False  # Override termination

def main():
    """Run the entropy management demo."""
    logger.info("Starting Entropy Management Demo")
    
    # Create and populate memory store
    memory_store = create_memory_store()
    logger.info(f"Created memory store with {len(memory_store.get_all_memories())} initial memories")
    
    # Set up coherence scorer for semantic coherence entropy
    coherence_scorer = BasicCoherenceScorer()
    
    # 1. Measuring entropy
    logger.info("\n=== PART 1: MEASURING ENTROPY ===")
    
    # Initialize entropy calculator
    entropy_calculator = EntropyCalculator()
    
    # Measure embedding entropy
    embedding_entropy = entropy_calculator.calculate_memory_store_entropy(memory_store)
    logger.info(f"Initial memory store entropy: {embedding_entropy:.4f}")
    
    # Examine individual memories
    for memory in memory_store.get_all_memories():
        memory_entropy = entropy_calculator.calculate_memory_entropy(memory)
        logger.info(f"Memory '{memory.content[:30]}...': Entropy = {memory_entropy:.4f}")
    
    # 2. Monitoring entropy
    logger.info("\n=== PART 2: ENTROPY MONITORING ===")
    
    # Configure entropy monitor
    settings = EntropyThresholdSettings(
        warning_threshold=0.6,    # Lower thresholds for demo purposes
        critical_threshold=0.7,
        emergency_threshold=0.8,
        termination_threshold=0.9,
        auto_terminate=True,
        termination_window_seconds=10.0,  # Shortened for demo
        check_interval_seconds=2.0        # Shortened for demo
    )
    
    monitor = EntropyMonitor(
        memory_store=memory_store,
        settings=settings,
        on_alert_change=on_alert_change,
        on_termination_decision=on_termination_decision
    )
    
    # Start with manual checks
    logger.info("Initial entropy check:")
    metrics = monitor.check_entropy()
    logger.info(f"Current entropy: {metrics['entropy']:.4f}, Alert level: {metrics['alert_level']}")
    
    # Add high entropy memories to increase system entropy
    logger.info("\nAdding high entropy memories...")
    add_high_entropy_memories(memory_store, count=8)
    
    # Check entropy again
    metrics = monitor.check_entropy()
    logger.info(f"After adding high entropy memories: {metrics['entropy']:.4f}, Alert level: {metrics['alert_level']}")
    
    # Start continuous monitoring
    logger.info("\nStarting continuous entropy monitoring...")
    monitor.start_monitoring()
    
    # Wait for some monitoring cycles
    for i in range(5):
        time.sleep(3)
        status = monitor.get_monitoring_status()
        logger.info(f"Monitoring cycle {i+1}: Entropy = {status['current_entropy']:.4f}, Alert level: {status['alert_level']}")
    
    # 3. Entropy-based pruning
    logger.info("\n=== PART 3: ENTROPY-BASED PRUNING ===")
    
    # Initialize pruner
    pruner = EntropyBasedPruner(
        entropy_threshold=0.6,     # Lower for demo purposes
        importance_boost=0.3,
        recency_protection=2,
        max_pruning_ratio=0.2
    )
    
    # Identify prunable memories
    logger.info("Identifying high-entropy memories for potential pruning...")
    candidates = pruner.identify_prunable_memories(memory_store)
    
    logger.info(f"Found {len(candidates)} pruning candidates:")
    for memory, entropy, score in candidates[:3]:  # Show top 3
        logger.info(f"- '{memory.content[:30]}...': Entropy = {entropy:.4f}, Score = {score:.4f}")
    
    # Perform dry run
    logger.info("\nPerforming pruning dry run...")
    dry_run_stats = pruner.prune_high_entropy_memories(memory_store, dry_run=True)
    logger.info(f"Dry run would prune {dry_run_stats['pruned_count']} memories with average entropy {dry_run_stats['average_entropy']:.4f}")
    
    # Perform actual pruning
    logger.info("\nPerforming actual pruning...")
    pruning_stats = pruner.prune_high_entropy_memories(memory_store, dry_run=False)
    logger.info(f"Pruned {pruning_stats['pruned_count']} memories with average entropy {pruning_stats['average_entropy']:.4f}")
    
    # Check entropy after pruning
    metrics = monitor.check_entropy()
    logger.info(f"After pruning: Entropy = {metrics['entropy']:.4f}, Alert level: {metrics['alert_level']}")
    
    # Stop monitoring
    logger.info("\nStopping entropy monitoring...")
    monitor.stop_monitoring()
    
    # Plot entropy over time
    plot_entropy_over_time(monitor.entropy_history)
    
    logger.info("\nEntropy Management Demo completed!")

if __name__ == "__main__":
    main() 