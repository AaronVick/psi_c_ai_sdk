#!/usr/bin/env python3
"""
Entropy Response System Demo

This script demonstrates how the entropy response system works to automatically
manage and reduce entropy in a ΨC-AI system. It creates a memory store with
high-entropy memories and shows how the response system detects and responds
to elevated entropy conditions.
"""

import time
import logging
import random
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.embedding.engine import EmbeddingEngine
from psi_c_ai_sdk.entropy.entropy import EntropyCalculator
from psi_c_ai_sdk.entropy.monitor import EntropyMonitor, EntropyAlert, EntropySubscriber
from psi_c_ai_sdk.entropy.response import (
    EntropyResponseStrategy, 
    EntropyResponseConfig,
    EntropyResponse,
    create_entropy_response
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("entropy_response_demo")


def generate_random_embedding(dim: int = 384) -> List[float]:
    """Generate a random embedding vector."""
    return list(np.random.normal(0, 1, dim))


def create_memory_with_entropy(content: str, entropy_level: float) -> Memory:
    """
    Create a memory with controlled entropy level.
    
    Args:
        content: Memory content
        entropy_level: Target entropy level (0.0-1.0)
        
    Returns:
        Memory object with the specified entropy level
    """
    # Create a base memory
    memory = Memory(content=content)
    
    # Generate a random embedding
    base_embedding = np.array(generate_random_embedding())
    
    # Normalize the embedding
    base_embedding = base_embedding / np.linalg.norm(base_embedding)
    
    # Add noise based on entropy level
    # Higher entropy = more extreme values
    if entropy_level > 0:
        noise = np.random.normal(0, entropy_level * 3, len(base_embedding))
        noisy_embedding = base_embedding + noise
        
        # Normalize again
        noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)
        memory.embedding = list(noisy_embedding)
    else:
        memory.embedding = list(base_embedding)
    
    # Set creation and access times
    now = datetime.now()
    memory.created_at = now - timedelta(days=random.randint(0, 30))
    memory.last_accessed = now - timedelta(minutes=random.randint(0, 600))
    
    # Set importance based on inverse of entropy
    # Lower entropy = higher importance
    memory.importance = max(0.1, min(0.9, 1.0 - entropy_level))
    
    return memory


def create_test_memory_store() -> MemoryStore:
    """
    Create a test memory store with memories of varying entropy levels.
    
    Returns:
        Populated memory store
    """
    memory_store = MemoryStore()
    
    # Generate memories with controlled entropy
    # Low entropy memories (0.0-0.3)
    low_entropy_contents = [
        "The sky is blue on a clear day.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth orbits around the Sun.",
        "Humans need oxygen to breathe.",
        "Plants convert sunlight into energy through photosynthesis."
    ]
    
    # Medium entropy memories (0.4-0.6)
    medium_entropy_contents = [
        "Sometimes I think the universe has a pattern, but other times it seems random.",
        "The meaning of life might be subjective or objective, depending on perspective.",
        "I'm not sure if free will truly exists or if all actions are predetermined.",
        "Consciousness could be an emergent property or something more fundamental.",
        "The boundary between art and non-art is often blurry and context-dependent."
    ]
    
    # High entropy memories (0.7-0.9)
    high_entropy_contents = [
        "The sky is green and water flows upward when time reverses its course.",
        "I simultaneously exist and don't exist in quantum superposition of states.",
        "Yesterday is tomorrow and tomorrow was yesterday when viewed from never.",
        "The self is both the observer and the observed, creating infinite recursion.",
        "My memories contradict each other, making me question which are real."
    ]
    
    # Add low entropy memories
    for content in low_entropy_contents:
        entropy = random.uniform(0.1, 0.3)
        memory = create_memory_with_entropy(content, entropy)
        memory_store.add_memory(memory)
    
    # Add medium entropy memories
    for content in medium_entropy_contents:
        entropy = random.uniform(0.4, 0.6)
        memory = create_memory_with_entropy(content, entropy)
        memory_store.add_memory(memory)
    
    # Add high entropy memories
    for content in high_entropy_contents:
        entropy = random.uniform(0.7, 0.9)
        memory = create_memory_with_entropy(content, entropy)
        memory_store.add_memory(memory)
    
    return memory_store


class DemoReflectionHandler:
    """Simulates a reflection system for the demo."""
    
    def __init__(self):
        """Initialize the reflection handler."""
        self.reflections = []
    
    def trigger_reflection(self, topic: str, details: Dict[str, Any]) -> None:
        """
        Handle a reflection trigger.
        
        Args:
            topic: Reflection topic
            details: Additional details for the reflection
        """
        logger.info(f"REFLECTION TRIGGERED: {topic}")
        logger.info(f"Reflection details: {details}")
        
        # Record the reflection
        self.reflections.append({
            "timestamp": datetime.now(),
            "topic": topic,
            "details": details
        })
        
        # In a real system, this would initiate a reflection cycle
        # For the demo, we'll just log it


class DemoLoggingSubscriber(EntropySubscriber):
    """Entropy subscriber that logs all events for the demo."""
    
    def on_entropy_alert(self, alert_level: EntropyAlert, entropy_value: float,
                        details: Dict[str, Any]) -> None:
        """Log entropy alerts."""
        logger.info(f"ALERT: {alert_level.name} entropy level: {entropy_value:.4f}")
        if alert_level != EntropyAlert.NORMAL:
            logger.info(f"Alert details: {details}")
    
    def on_termination_decision(self, entropy_value: float, details: Dict[str, Any]) -> bool:
        """Handle termination decisions."""
        logger.critical(f"TERMINATION DECISION REQUIRED: entropy = {entropy_value:.4f}")
        logger.info(f"Termination details: {details}")
        
        # In a real system, this might implement complex decision logic
        # For the demo, we'll always override termination
        return False


def simulate_increasing_entropy(memory_store: MemoryStore,
                               steps: int = 5,
                               delay: float = 2.0) -> None:
    """
    Simulate increasing entropy by adding progressively higher entropy memories.
    
    Args:
        memory_store: Memory store to add memories to
        steps: Number of simulation steps
        delay: Seconds to wait between steps
    """
    logger.info("Starting entropy increase simulation...")
    
    for step in range(steps):
        # Calculate target entropy for this step
        target_entropy = 0.5 + (step / steps) * 0.4
        
        # Create 2 memories with the target entropy
        for i in range(2):
            content = f"Simulation step {step+1}, memory {i+1}: entropy level {target_entropy:.2f}"
            memory = create_memory_with_entropy(content, target_entropy)
            memory_store.add_memory(memory)
        
        # Calculate current system entropy
        calculator = EntropyCalculator()
        current_entropy = calculator.calculate_memory_store_entropy(memory_store)
        
        logger.info(f"Simulation step {step+1}: Added memories with entropy {target_entropy:.2f}, system entropy: {current_entropy:.4f}")
        
        # Wait before next step
        if step < steps - 1:
            time.sleep(delay)


def run_demo():
    """Run the entropy response system demo."""
    logger.info("Starting Entropy Response System Demo")
    
    # Create a memory store with test memories
    logger.info("Creating memory store with test memories...")
    memory_store = create_test_memory_store()
    
    # Calculate initial entropy
    calculator = EntropyCalculator()
    initial_entropy = calculator.calculate_memory_store_entropy(memory_store)
    logger.info(f"Initial system entropy: {initial_entropy:.4f}")
    
    # Create entropy monitor with custom thresholds for demo
    # Using lower thresholds to trigger alerts more easily
    logger.info("Setting up entropy monitor...")
    monitor = EntropyMonitor(
        memory_store=memory_store,
        elevated_threshold=0.4,    # Normally 0.6
        high_threshold=0.6,        # Normally 0.75
        critical_threshold=0.75,   # Normally 0.85
        termination_threshold=0.85, # Normally 0.95
        check_interval=1.0         # Check every second for demo
    )
    
    # Add a logging subscriber
    logger.info("Adding logging subscriber...")
    logging_subscriber = DemoLoggingSubscriber()
    monitor.add_subscriber(logging_subscriber)
    
    # Create reflection handler
    logger.info("Setting up reflection handler...")
    reflection_handler = DemoReflectionHandler()
    
    # Create entropy response system with custom configuration
    logger.info("Setting up entropy response system...")
    response_config = EntropyResponseConfig(
        elevated_strategies=[
            EntropyResponseStrategy.MEMORY_PRUNING,
            EntropyResponseStrategy.TARGETED_REFLECTION
        ],
        high_strategies=[
            EntropyResponseStrategy.MEMORY_PRUNING,
            EntropyResponseStrategy.MEMORY_ISOLATION,
            EntropyResponseStrategy.TARGETED_REFLECTION
        ],
        critical_strategies=[
            EntropyResponseStrategy.MEMORY_PRUNING,
            EntropyResponseStrategy.MEMORY_ISOLATION,
            EntropyResponseStrategy.ENTROPY_DIFFUSION
        ],
        cooldown_period=3.0,  # Short cooldown for demo
        max_memory_prune_ratio=0.2,  # Higher ratio for demo
        min_entropy_reduction_target=0.05  # Lower target for demo
    )
    
    response = create_entropy_response(
        memory_store=memory_store,
        entropy_monitor=monitor,
        reflection_trigger_callback=reflection_handler.trigger_reflection,
        config=response_config
    )
    
    # Start monitoring
    logger.info("Starting entropy monitoring...")
    monitor.start()
    
    # Run the demo: simulate increasing entropy
    logger.info("Starting entropy increase simulation...")
    time.sleep(2)  # Let the monitor initialize
    
    # Simulate increasing entropy
    simulate_increasing_entropy(memory_store, steps=5, delay=3.0)
    
    # Show final status
    time.sleep(5)  # Wait for final responses
    
    # Get response history
    response_history = response.get_response_history()
    if response_history:
        logger.info(f"Response history: {len(response_history)} entries")
        for i, entry in enumerate(response_history):
            logger.info(f"Response {i+1}:")
            logger.info(f"  Alert level: {entry['alert_level']}")
            logger.info(f"  Entropy change: {entry['initial_entropy']:.4f} -> {entry['final_entropy']:.4f}")
            logger.info(f"  Strategies applied: {[s['strategy'] for s in entry['strategies_applied']]}")
    else:
        logger.info("No response history recorded")
    
    # Show reflection history
    if reflection_handler.reflections:
        logger.info(f"Reflection history: {len(reflection_handler.reflections)} entries")
        for i, reflection in enumerate(reflection_handler.reflections):
            logger.info(f"Reflection {i+1}: {reflection['topic']}")
    else:
        logger.info("No reflections triggered")
    
    # Stop monitoring
    logger.info("Stopping entropy monitoring...")
    monitor.stop()
    
    # Calculate final entropy
    final_entropy = calculator.calculate_memory_store_entropy(memory_store)
    logger.info(f"Final system entropy: {final_entropy:.4f}")
    logger.info(f"Entropy change: {initial_entropy:.4f} -> {final_entropy:.4f}")
    
    logger.info("Entropy Response System Demo complete")


if __name__ == "__main__":
    run_demo() 