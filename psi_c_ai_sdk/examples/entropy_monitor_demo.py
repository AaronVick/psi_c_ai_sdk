#!/usr/bin/env python3
"""
Entropy Monitoring System Demo

This script demonstrates the functionality of the Entropy Monitoring System in the ΨC-AI SDK.
It shows how the system:
1. Measures different types of entropy (embedding, semantic, temporal)
2. Monitors entropy levels and triggers alerts when thresholds are exceeded
3. Implements entropy-based memory pruning based on configurable strategies
4. Visualizes entropy trends and the impact of pruning operations

The demo simulates memory accumulation over time and shows how entropy affects system performance.
"""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.memory.embedding import EmbeddingModel
from psi_c_ai_sdk.entropy.entropy import (
    EntropyCalculator, 
    EmbeddingEntropyMeasure, 
    SemanticCoherenceEntropyMeasure,
    TemporalCoherenceEntropyMeasure
)
from psi_c_ai_sdk.entropy.monitor import (
    EntropyMonitor,
    EntropyAlert,
    EntropyAlertHandler,
    EntropyThreshold
)
from psi_c_ai_sdk.entropy.pruning import EntropyPruner, PruningStrategy
from psi_c_ai_sdk.util.logging import setup_logger

# Set up logging
logger = setup_logger("entropy_monitor_demo", level=logging.INFO)

class DemoAlertHandler(EntropyAlertHandler):
    """Custom alert handler for the demo that logs alerts and takes actions."""
    
    def __init__(self, memory_store: MemoryStore, entropy_pruner: Optional['EntropyPruner'] = None):
        self.memory_store = memory_store
        self.entropy_pruner = entropy_pruner
        self.alerts_history = []
        
    def handle_alert(self, alert: EntropyAlert) -> None:
        """Handle an entropy alert by logging it and taking appropriate action."""
        self.alerts_history.append(alert)
        
        logger.warning(f"ENTROPY ALERT: {alert.alert_type} entropy ({alert.entropy_type}) "
                      f"reached {alert.current_value:.4f}, threshold: {alert.threshold_value:.4f}")
        
        # Take action based on alert severity
        if alert.alert_type == "WARNING":
            logger.info("Taking action: Running diagnostic analysis")
            self._run_diagnostic()
            
        elif alert.alert_type == "CRITICAL":
            logger.warning("Taking action: Initiating entropy reduction via pruning")
            if self.entropy_pruner:
                self._apply_pruning(alert)
                
        elif alert.alert_type == "EXTREME":
            logger.error("Taking action: Emergency pruning and system slowdown")
            if self.entropy_pruner:
                self._apply_emergency_pruning(alert)
    
    def _run_diagnostic(self) -> None:
        """Run a diagnostic to analyze entropy sources."""
        memory_count = len(self.memory_store.get_all())
        logger.info(f"Diagnostic: Memory store contains {memory_count} memories")
        
        # For demo purposes, just log some information
        recent_count = len([m for m in self.memory_store.get_all() 
                           if (datetime.now() - m.creation_time).total_seconds() < 3600])
        logger.info(f"Diagnostic: {recent_count} memories added in the last hour")
        
    def _apply_pruning(self, alert: EntropyAlert) -> None:
        """Apply pruning based on the alert."""
        if self.entropy_pruner:
            pre_count = len(self.memory_store.get_all())
            
            # Select pruning strategy based on entropy type
            if alert.entropy_type == "embedding":
                strategy = PruningStrategy.OUTLIER_REMOVAL
            elif alert.entropy_type == "semantic":
                strategy = PruningStrategy.REDUNDANCY_REMOVAL
            else:
                strategy = PruningStrategy.TEMPORAL_DECAY
                
            # Apply pruning
            removed_count = self.entropy_pruner.prune(strategy=strategy, target_reduction=0.15)
            post_count = len(self.memory_store.get_all())
            
            logger.info(f"Pruning complete: Removed {removed_count} memories "
                        f"({pre_count} → {post_count})")
    
    def _apply_emergency_pruning(self, alert: EntropyAlert) -> None:
        """Apply emergency pruning for extreme entropy levels."""
        if self.entropy_pruner:
            pre_count = len(self.memory_store.get_all())
            
            # Apply aggressive pruning
            removed_count = self.entropy_pruner.prune(
                strategy=PruningStrategy.AGGRESSIVE, 
                target_reduction=0.30
            )
            post_count = len(self.memory_store.get_all())
            
            logger.error(f"Emergency pruning complete: Removed {removed_count} memories "
                         f"({pre_count} → {post_count})")

def generate_random_memories(count: int, time_span_days: int = 30) -> List[Memory]:
    """
    Generate random memories for testing entropy.
    
    Args:
        count: Number of memories to generate
        time_span_days: Time span in days to distribute the memories
        
    Returns:
        List of generated Memory objects
    """
    # Sample content templates for generating variations
    content_templates = [
        "The user mentioned that they {feeling} {activity}.",
        "During the conversation, the user expressed {emotion} about {topic}.",
        "The user's preference for {item} is {level}.",
        "The system detected {emotion} when discussing {topic}.",
        "According to data from {source}, the user {action} {frequency}.",
        "The user's interaction with {feature} showed {level} engagement.",
        "The analysis indicated that the user has {level} interest in {topic}.",
        "The user stated that they {agreement} with the {statement} about {topic}.",
        "The recorded preference shows {level} importance for {feature}.",
        "The user has {frequency} accessed the {feature} in the past {time_period}."
    ]
    
    # Sample data for template placeholders
    feelings = ["enjoys", "dislikes", "loves", "hates", "is neutral about"]
    activities = ["reading books", "watching movies", "playing games", "exercising", 
                 "coding", "cooking", "traveling", "shopping", "social media"]
    emotions = ["happiness", "frustration", "excitement", "confusion", "satisfaction", 
               "disappointment", "surprise", "curiosity", "boredom", "anxiety"]
    topics = ["AI technology", "privacy settings", "user interface", "performance", 
             "new features", "subscription plans", "customer support", "app design",
             "content recommendations", "social features"]
    items = ["dark mode", "light mode", "notifications", "email updates", "auto-play", 
            "sound effects", "animations", "data usage", "high contrast", "font size"]
    levels = ["high", "moderate", "low", "minimal", "extreme", "variable", "consistent",
             "increasing", "decreasing", "stable"]
    sources = ["usage data", "survey responses", "direct feedback", "behavior analysis", 
              "preference settings", "support tickets", "in-app reactions"]
    actions = ["uses the app", "engages with content", "changes settings", "reads help articles",
              "contacts support", "explores new features", "shares content"]
    frequencies = ["frequently", "occasionally", "rarely", "never", "always", "sometimes",
                  "consistently", "sporadically", "regularly", "seldom"]
    features = ["search function", "recommendation system", "settings menu", "profile page",
               "notification center", "help section", "feedback form", "content filters"]
    agreement = ["strongly agrees", "somewhat agrees", "is neutral", "somewhat disagrees", 
                "strongly disagrees", "is undecided", "has mixed feelings"]
    statements = ["policy", "proposal", "suggestion", "recommendation", "statement", "assertion"]
    time_periods = ["week", "month", "day", "year", "quarter", "session"]
    
    # Generate random memories
    memories = []
    end_time = datetime.now()
    start_time = end_time - timedelta(days=time_span_days)
    time_range = (end_time - start_time).total_seconds()
    
    for i in range(count):
        # Select a random template and fill it
        template = random.choice(content_templates)
        content = template
        
        if "{feeling}" in content and "{activity}" in content:
            content = content.replace("{feeling}", random.choice(feelings))
            content = content.replace("{activity}", random.choice(activities))
        
        if "{emotion}" in content and "{topic}" in content:
            content = content.replace("{emotion}", random.choice(emotions))
            content = content.replace("{topic}", random.choice(topics))
            
        if "{item}" in content and "{level}" in content:
            content = content.replace("{item}", random.choice(items))
            content = content.replace("{level}", random.choice(levels))
            
        if "{source}" in content:
            content = content.replace("{source}", random.choice(sources))
            
        if "{action}" in content and "{frequency}" in content:
            content = content.replace("{action}", random.choice(actions))
            content = content.replace("{frequency}", random.choice(frequencies))
            
        if "{feature}" in content:
            content = content.replace("{feature}", random.choice(features))
            
        if "{agreement}" in content and "{statement}" in content:
            content = content.replace("{agreement}", random.choice(agreement))
            content = content.replace("{statement}", random.choice(statements))
            
        if "{time_period}" in content:
            content = content.replace("{time_period}", random.choice(time_periods))
        
        # Replace any remaining placeholders
        for placeholder in ["{feeling}", "{activity}", "{emotion}", "{topic}", "{item}", 
                           "{level}", "{source}", "{action}", "{frequency}", "{feature}", 
                           "{agreement}", "{statement}", "{time_period}"]:
            if placeholder in content:
                content = content.replace(placeholder, random.choice(
                    feelings + activities + emotions + topics + items + levels + 
                    sources + actions + frequencies + features + agreement + 
                    statements + time_periods
                ))
        
        # Create random timestamp within the time range
        random_time_offset = random.random() * time_range
        timestamp = start_time + timedelta(seconds=random_time_offset)
        
        # Create memory with random importance
        memory = Memory(
            content=content,
            source=random.choice(["user_input", "system_observation", "data_analysis", 
                                 "external_api", "preference_setting"]),
            creation_time=timestamp,
            importance=random.uniform(0.3, 0.9),
            metadata={
                "confidence": random.uniform(0.5, 1.0),
                "category": random.choice(["preference", "behavior", "feedback", 
                                          "interaction", "analysis"])
            }
        )
        memories.append(memory)
    
    # Sort by creation time
    memories.sort(key=lambda m: m.creation_time)
    return memories

def plot_entropy_trends(entropy_history: List[Dict[str, Any]]) -> None:
    """Plot entropy trends over time based on history data."""
    if not entropy_history:
        logger.warning("No entropy history to plot")
        return
    
    # Extract data for plotting
    timestamps = [entry["timestamp"] for entry in entropy_history]
    total_values = [entry["total"] for entry in entropy_history]
    embedding_values = [entry.get("embedding", 0) for entry in entropy_history]
    semantic_values = [entry.get("semantic", 0) for entry in entropy_history]
    temporal_values = [entry.get("temporal", 0) for entry in entropy_history]
    
    # Convert timestamps to relative seconds for x-axis
    start_time = timestamps[0]
    seconds = [(t - start_time).total_seconds() for t in timestamps]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(seconds, embedding_values, 'r-', label='Embedding Entropy')
    plt.plot(seconds, semantic_values, 'g-', label='Semantic Entropy')
    plt.plot(seconds, temporal_values, 'b-', label='Temporal Entropy')
    plt.plot(seconds, total_values, 'k--', label='Total Entropy', linewidth=2)
    
    # Mark pruning events
    memory_counts = [entry.get("memory_count", 0) for entry in entropy_history]
    for i in range(1, len(memory_counts)):
        if memory_counts[i] < memory_counts[i-1]:
            plt.axvline(x=seconds[i], color='purple', linestyle='--', alpha=0.5)
            plt.text(seconds[i], max(total_values)*0.9, 'Pruning', 
                    rotation=90, verticalalignment='top')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Entropy Value')
    plt.title('Entropy Trends Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add thresholds
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
    plt.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
    plt.axhline(y=0.9, color='darkred', linestyle='--', alpha=0.7, label='Extreme Threshold')
    
    plt.tight_layout()
    plt.savefig('entropy_trends.png')
    logger.info("Entropy trend plot saved to 'entropy_trends.png'")
    plt.close()

def run_demo() -> None:
    """Run the entropy monitoring system demonstration."""
    logger.info("Starting Entropy Monitoring System Demo")
    
    # Initialize core components
    embedding_model = EmbeddingModel()
    memory_store = MemoryStore(embedding_model)
    
    # Set up entropy measures
    embedding_entropy = EmbeddingEntropyMeasure()
    semantic_entropy = SemanticCoherenceEntropyMeasure()
    temporal_entropy = TemporalCoherenceEntropyMeasure()
    
    # Create entropy calculator with weights
    entropy_calculator = EntropyCalculator(
        measures={
            "embedding": embedding_entropy,
            "semantic": semantic_entropy,
            "temporal": temporal_entropy
        },
        weights={
            "embedding": 0.4,
            "semantic": 0.4,
            "temporal": 0.2
        }
    )
    
    # Create entropy pruner
    entropy_pruner = EntropyPruner(
        memory_store=memory_store,
        entropy_calculator=entropy_calculator
    )
    
    # Create custom alert handler
    alert_handler = DemoAlertHandler(
        memory_store=memory_store,
        entropy_pruner=entropy_pruner
    )
    
    # Configure entropy thresholds
    thresholds = [
        EntropyThreshold(entropy_type="embedding", value=0.6, alert_type="WARNING"),
        EntropyThreshold(entropy_type="embedding", value=0.75, alert_type="CRITICAL"),
        EntropyThreshold(entropy_type="embedding", value=0.9, alert_type="EXTREME"),
        
        EntropyThreshold(entropy_type="semantic", value=0.6, alert_type="WARNING"),
        EntropyThreshold(entropy_type="semantic", value=0.75, alert_type="CRITICAL"),
        EntropyThreshold(entropy_type="semantic", value=0.9, alert_type="EXTREME"),
        
        EntropyThreshold(entropy_type="temporal", value=0.6, alert_type="WARNING"),
        EntropyThreshold(entropy_type="temporal", value=0.75, alert_type="CRITICAL"),
        EntropyThreshold(entropy_type="temporal", value=0.9, alert_type="EXTREME"),
        
        EntropyThreshold(entropy_type="total", value=0.6, alert_type="WARNING"),
        EntropyThreshold(entropy_type="total", value=0.75, alert_type="CRITICAL"),
        EntropyThreshold(entropy_type="total", value=0.9, alert_type="EXTREME"),
    ]
    
    # Create entropy monitor
    entropy_monitor = EntropyMonitor(
        memory_store=memory_store,
        entropy_calculator=entropy_calculator,
        thresholds=thresholds,
        alert_handler=alert_handler,
        check_interval=5  # in seconds
    )
    
    # Start with some initial memories
    logger.info("Generating initial memories...")
    initial_memories = generate_random_memories(50, time_span_days=30)
    for memory in initial_memories:
        memory_store.add(memory)
    logger.info(f"Added {len(initial_memories)} initial memories")
    
    # Start entropy monitor
    logger.info("Starting entropy monitor...")
    entropy_monitor.start()
    
    # Keep track of entropy history for plotting
    entropy_history = []
    
    try:
        # Simulation loop - add memories over time and measure entropy
        logger.info("Running simulation - adding memories and measuring entropy...")
        
        # Create batches of memories to add gradually
        memory_batches = [
            generate_random_memories(20, time_span_days=5),  # normal load
            generate_random_memories(50, time_span_days=5),  # higher load
            generate_random_memories(100, time_span_days=5),  # heavy load
            generate_random_memories(20, time_span_days=5),  # back to normal
        ]
        
        for batch_index, memory_batch in enumerate(memory_batches):
            logger.info(f"Adding batch {batch_index + 1}/{len(memory_batches)} "
                       f"with {len(memory_batch)} memories...")
            
            # Add memories from the batch with delay between each
            for i, memory in enumerate(memory_batch):
                memory_store.add(memory)
                
                # Calculate current entropy levels
                entropy_values = entropy_calculator.calculate_memory_store_entropy(memory_store)
                total_entropy = entropy_calculator.calculate_total_entropy(entropy_values)
                
                # Add to history
                entropy_history.append({
                    "timestamp": datetime.now(),
                    "memory_count": len(memory_store.get_all()),
                    "embedding": entropy_values.get("embedding", 0),
                    "semantic": entropy_values.get("semantic", 0),
                    "temporal": entropy_values.get("temporal", 0),
                    "total": total_entropy
                })
                
                logger.info(f"Memory {i+1}/{len(memory_batch)} added. "
                           f"Current entropy: {total_entropy:.4f}")
                
                # Check for alerts manually (entropy monitor should do this automatically)
                entropy_monitor.check_thresholds(entropy_values, total_entropy)
                
                # Sleep to simulate time passing
                time.sleep(0.2)
            
            # Wait a bit longer between batches
            time.sleep(2)
        
        # Run one final check
        entropy_values = entropy_calculator.calculate_memory_store_entropy(memory_store)
        total_entropy = entropy_calculator.calculate_total_entropy(entropy_values)
        entropy_monitor.check_thresholds(entropy_values, total_entropy)
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        # Stop entropy monitor
        entropy_monitor.stop()
        
        # Generate entropy trend plot
        plot_entropy_trends(entropy_history)
        
        # Print summary
        alert_count = len(alert_handler.alerts_history)
        logger.info(f"\nDemo completed with {alert_count} entropy alerts generated")
        
        alert_types = {}
        for alert in alert_handler.alerts_history:
            alert_type = f"{alert.alert_type}:{alert.entropy_type}"
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        
        logger.info("Alert summary:")
        for alert_type, count in alert_types.items():
            logger.info(f"  {alert_type}: {count}")
        
        # Final memory count
        final_count = len(memory_store.get_all())
        logger.info(f"Final memory count: {final_count}")
        logger.info(f"Initial memory count: {len(initial_memories)}")
        logger.info(f"Total memories added: {sum(len(batch) for batch in memory_batches)}")
        logger.info(f"Memories pruned: {len(initial_memories) + sum(len(batch) for batch in memory_batches) - final_count}")
        
        logger.info("\nEntropy Monitoring System Demo completed successfully")

if __name__ == "__main__":
    run_demo() 