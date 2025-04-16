#!/usr/bin/env python3
"""
Entropy Monitoring Demo - ΨC-AI SDK

This example demonstrates how to use the entropy monitoring system to track
and respond to entropy levels in a memory system.

The demo:
1. Creates a memory store with random memories
2. Sets up an entropy monitor with customized thresholds
3. Implements subscribers to respond to entropy events
4. Intentionally introduces high-entropy content to trigger alerts
5. Visualizes entropy trends over time
"""

import time
import random
import threading
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

from psi_c_ai_sdk.memory.memory import MemoryStore, Memory
from psi_c_ai_sdk.embedding.embedding import EmbeddingEngine, BasicEmbeddingEngine
from psi_c_ai_sdk.entropy.entropy import EntropyCalculator
from psi_c_ai_sdk.entropy.monitor import EntropyMonitor, EntropyAlert, EntropySubscriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("entropy_demo")


class EntropyDashboard(EntropySubscriber):
    """A simple dashboard to visualize entropy levels and alerts."""
    
    def __init__(self):
        self.alert_history = []
        self.termination_requests = 0
        self.termination_overrides = 0
        self.alert_counts = {
            EntropyAlert.NORMAL: 0,
            EntropyAlert.ELEVATED: 0,
            EntropyAlert.HIGH: 0,
            EntropyAlert.CRITICAL: 0
        }
        # For plotting
        self.times = []
        self.entropy_values = []
        self.alert_times = []
        self.alert_values = []
        self.alert_colors = []
        
        # Create the plot now
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle("Memory System Entropy Monitor")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Entropy Value")
        self.ax.set_ylim(0, 1)
        self.ax.axhline(y=0.6, color='yellow', linestyle='--', alpha=0.5, label="Elevated")
        self.ax.axhline(y=0.75, color='orange', linestyle='--', alpha=0.5, label="High")
        self.ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label="Critical")
        self.ax.axhline(y=0.95, color='darkred', linestyle='--', alpha=0.5, label="Termination")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.line, = self.ax.plot([], [], 'b-', label="Entropy")
        self.scatter = self.ax.scatter([], [], s=50, alpha=0.7)
        
        self.last_update = datetime.now()
    
    def on_entropy_alert(self, alert_level: EntropyAlert, entropy_value: float, 
                          details: Dict[str, Any]) -> None:
        """Handle entropy alerts by updating the dashboard."""
        logger.info(f"Dashboard received alert: {alert_level.name} ({entropy_value:.4f})")
        
        # Record the alert
        self.alert_history.append({
            "time": datetime.now(),
            "level": alert_level,
            "value": entropy_value,
            "details": details
        })
        self.alert_counts[alert_level] += 1
        
        # Add to the visualization data
        self.alert_times.append(datetime.now())
        self.alert_values.append(entropy_value)
        
        # Set color based on alert level
        if alert_level == EntropyAlert.ELEVATED:
            self.alert_colors.append('yellow')
        elif alert_level == EntropyAlert.HIGH:
            self.alert_colors.append('orange')
        elif alert_level == EntropyAlert.CRITICAL:
            self.alert_colors.append('red')
        else:
            self.alert_colors.append('green')
    
    def on_termination_decision(self, entropy_value: float, 
                                 details: Dict[str, Any]) -> bool:
        """Handle termination decisions."""
        self.termination_requests += 1
        logger.warning(f"Termination decision requested: entropy={entropy_value:.4f}")
        
        # In this demo, we'll always override termination to keep the demo running
        self.termination_overrides += 1
        return False  # Override termination
    
    def add_entropy_data(self, time: datetime, value: float) -> None:
        """Add a new entropy data point to the dashboard."""
        self.times.append(time)
        self.entropy_values.append(value)
    
    def update_display(self) -> None:
        """Update the visualization display."""
        # Only update every second to avoid too many redraws
        if (datetime.now() - self.last_update).total_seconds() < 1.0:
            return
            
        self.last_update = datetime.now()
        
        # Convert datetime to matplotlib format
        times_float = [mdates.date2num(t) for t in self.times[-100:]]
        
        # Update the line
        self.line.set_xdata(times_float)
        self.line.set_ydata(self.entropy_values[-100:])
        
        # Update the scatter points for alerts
        if self.alert_times:
            alert_times_float = [mdates.date2num(t) for t in self.alert_times]
            self.scatter.set_offsets(np.column_stack([alert_times_float, self.alert_values]))
            self.scatter.set_facecolor(self.alert_colors)
        
        # Adjust x-axis limits
        if times_float:
            self.ax.set_xlim(min(times_float), max(times_float))
            
        # Set date formatter
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current dashboard statistics."""
        return {
            "total_alerts": len(self.alert_history),
            "alert_counts": {k.name: v for k, v in self.alert_counts.items()},
            "termination_requests": self.termination_requests,
            "termination_overrides": self.termination_overrides,
            "data_points": len(self.times)
        }


class EntropyResponder(EntropySubscriber):
    """
    A demonstration class that takes action in response to entropy alerts.
    
    In a real system, this might implement entropy reduction strategies like:
    - Pruning low-importance memories
    - Consolidating similar memories
    - Restructuring concept graphs
    - Limiting input/perception
    """
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.actions_taken = []
    
    def on_entropy_alert(self, alert_level: EntropyAlert, entropy_value: float, 
                          details: Dict[str, Any]) -> None:
        """Respond to entropy alerts with appropriate actions."""
        if alert_level == EntropyAlert.NORMAL:
            # No action needed
            return
            
        # Log that we received an alert
        logger.info(f"EntropyResponder handling alert: {alert_level.name}")
        
        # Take action based on alert level
        if alert_level == EntropyAlert.ELEVATED:
            # Gentle response - remove oldest memories
            self._prune_oldest_memories(3)
            
        elif alert_level == EntropyAlert.HIGH:
            # Stronger response - remove low importance memories
            self._prune_low_importance_memories(5)
            
        elif alert_level == EntropyAlert.CRITICAL:
            # Emergency response - aggressive pruning
            self._prune_low_importance_memories(10)
            self._consolidate_similar_memories()
    
    def on_termination_decision(self, entropy_value: float, 
                                 details: Dict[str, Any]) -> bool:
        """Handle termination decisions with last-resort actions."""
        logger.warning("Taking emergency action to reduce entropy")
        
        # Last resort entropy reduction
        self._emergency_entropy_reduction()
        
        # Record the action
        self.actions_taken.append({
            "time": datetime.now(),
            "action": "emergency_reduction",
            "entropy": entropy_value
        })
        
        # We'll still override termination for the demo
        return False
    
    def _prune_oldest_memories(self, count: int) -> None:
        """Remove oldest memories to reduce entropy."""
        memories = self.memory_store.get_all_memories()
        if not memories:
            return
            
        # Sort by creation time
        sorted_memories = sorted(memories, key=lambda m: m.creation_time)
        
        # Remove oldest memories
        for memory in sorted_memories[:count]:
            self.memory_store.remove_memory(memory.id)
        
        # Record the action
        self.actions_taken.append({
            "time": datetime.now(),
            "action": "prune_oldest",
            "count": count
        })
        
        logger.info(f"Pruned {count} oldest memories")
    
    def _prune_low_importance_memories(self, count: int) -> None:
        """Remove least important memories to reduce entropy."""
        memories = self.memory_store.get_all_memories()
        if not memories:
            return
            
        # Sort by importance (in a real system, you'd have a proper importance score)
        sorted_memories = sorted(memories, 
                                key=lambda m: m.metadata.get("importance", 0))
        
        # Remove least important memories
        for memory in sorted_memories[:count]:
            self.memory_store.remove_memory(memory.id)
        
        # Record the action
        self.actions_taken.append({
            "time": datetime.now(),
            "action": "prune_low_importance",
            "count": count
        })
        
        logger.info(f"Pruned {count} low-importance memories")
    
    def _consolidate_similar_memories(self) -> None:
        """Consolidate similar memories to reduce redundancy."""
        # This would use embedding similarity to find and merge similar memories
        # For the demo, we'll just simulate this
        
        # Record the action
        self.actions_taken.append({
            "time": datetime.now(),
            "action": "consolidate_similar"
        })
        
        logger.info("Consolidated similar memories")
    
    def _emergency_entropy_reduction(self) -> None:
        """Emergency entropy reduction - aggressive pruning."""
        # In a real system, this would involve more sophisticated strategies
        
        # For the demo, we'll remove 25% of all memories
        memories = self.memory_store.get_all_memories()
        count = len(memories) // 4
        
        if count > 0:
            # Randomly select memories to remove
            to_remove = random.sample(memories, count)
            for memory in to_remove:
                self.memory_store.remove_memory(memory.id)
        
        # Record the action
        self.actions_taken.append({
            "time": datetime.now(),
            "action": "emergency_reduction",
            "count": count
        })
        
        logger.warning(f"Emergency entropy reduction: removed {count} memories")


def create_memory_store() -> MemoryStore:
    """Create and populate a memory store with sample memories."""
    # Create embedding engine
    embedding_engine = BasicEmbeddingEngine(dim=32)
    
    # Create memory store
    memory_store = MemoryStore(embedding_engine=embedding_engine)
    
    # Add some initial memories with different topics
    topics = ["science", "art", "philosophy", "technology", "nature"]
    
    for i in range(20):
        topic = random.choice(topics)
        content = f"Memory about {topic} #{i}: {random.randint(1000, 9999)}"
        
        # Create memory with metadata
        memory = Memory(
            content=content,
            metadata={
                "topic": topic,
                "importance": random.random(),
                "source": "initial_setup",
                "topic_consistency": random.random()
            }
        )
        
        memory_store.add_memory(memory)
    
    logger.info(f"Created memory store with {len(memory_store.get_all_memories())} memories")
    return memory_store


def add_high_entropy_memories(memory_store: MemoryStore) -> None:
    """Add memories that increase entropy in the system."""
    # Create some conflicting memories
    conflicting_pairs = [
        ("The sky is blue", "The sky is green"),
        ("Water freezes at 0°C", "Water freezes at 32°F"),
        ("2+2=4", "2+2=5"),
        ("The Earth is round", "The Earth is flat"),
        ("Entropy always increases", "Entropy can be reversed")
    ]
    
    # Add one of each pair
    for truth, contradiction in conflicting_pairs:
        use_contradiction = random.random() > 0.5
        content = contradiction if use_contradiction else truth
        
        memory = Memory(
            content=content,
            metadata={
                "topic": "contradiction",
                "importance": random.random(),
                "source": "entropy_generator",
                "is_contradiction": use_contradiction
            }
        )
        
        memory_store.add_memory(memory)
    
    # Add some random entropy-increasing memories
    topics = ["quantum", "paradox", "confusion", "uncertainty", "chaos"]
    
    for i in range(10):
        topic = random.choice(topics)
        # Create more chaotic, less coherent content
        words = ["uncertain", "paradoxical", "quantum", "chaotic", 
                "inconsistent", "random", "fluctuating", "noisy"]
        content = f"High entropy {topic}: " + " ".join(random.sample(words, 4))
        
        memory = Memory(
            content=content,
            metadata={
                "topic": topic,
                "importance": random.random(),
                "source": "entropy_generator",
                "entropy_factor": random.random() * 0.5 + 0.5  # Higher values
            }
        )
        
        memory_store.add_memory(memory)
    
    logger.info("Added high-entropy memories to the store")


def memory_generator_thread(memory_store: MemoryStore, stop_event: threading.Event) -> None:
    """Thread that periodically adds new memories to increase entropy."""
    high_entropy_mode = False
    counter = 0
    
    while not stop_event.is_set():
        # Add a new memory
        counter += 1
        
        # Toggle high entropy mode occasionally
        if counter % 10 == 0:
            high_entropy_mode = not high_entropy_mode
            logger.info(f"Memory generator: {'HIGH' if high_entropy_mode else 'NORMAL'} entropy mode")
        
        if high_entropy_mode:
            # Create high-entropy memories
            topics = ["paradox", "contradiction", "uncertainty"]
            topic = random.choice(topics)
            
            # Create somewhat contradictory or uncertain content
            content = f"Generated high-entropy {topic} #{counter}: uncertain and paradoxical statement #{random.randint(1000, 9999)}"
            
            # Add inconsistency metadata
            importance = random.random() * 0.7 + 0.3  # Higher importance
            memory = Memory(
                content=content,
                metadata={
                    "topic": topic,
                    "importance": importance,
                    "source": "generator",
                    "consistency": random.random() * 0.3,  # Low consistency
                    "certainty": random.random() * 0.4     # Low certainty
                }
            )
        else:
            # Create normal memories
            topics = ["science", "art", "technology"]
            topic = random.choice(topics)
            
            # Create normal content
            content = f"Generated memory {topic} #{counter}: {random.randint(1000, 9999)}"
            
            importance = random.random() * 0.5  # Normal importance
            memory = Memory(
                content=content,
                metadata={
                    "topic": topic,
                    "importance": importance,
                    "source": "generator",
                    "consistency": random.random() * 0.5 + 0.5,  # Higher consistency
                    "certainty": random.random() * 0.5 + 0.5     # Higher certainty
                }
            )
        
        memory_store.add_memory(memory)
        
        # Add multiple memories in high entropy mode
        if high_entropy_mode and random.random() > 0.7:
            add_high_entropy_memories(memory_store)
        
        # Sleep for a random interval
        time.sleep(random.random() * 2 + 1)


def main():
    """Run the entropy monitoring demonstration."""
    import matplotlib.dates as mdates
    
    logger.info("Starting entropy monitoring demonstration")
    
    # Create a memory store with initial memories
    memory_store = create_memory_store()
    
    # Create entropy calculator and monitor
    entropy_calculator = EntropyCalculator()
    
    # Create monitor with custom thresholds
    monitor = EntropyMonitor(
        memory_store=memory_store,
        entropy_calculator=entropy_calculator,
        elevated_threshold=0.6,
        high_threshold=0.75,
        critical_threshold=0.85,
        termination_threshold=0.95,
        check_interval=1.0,  # Check every second for the demo
        window_size=10
    )
    
    # Create dashboard
    dashboard = EntropyDashboard()
    monitor.add_subscriber(dashboard)
    
    # Create responder
    responder = EntropyResponder(memory_store)
    monitor.add_subscriber(responder)
    
    # Start the monitor
    monitor.start()
    
    # Start memory generator thread
    stop_event = threading.Event()
    generator_thread = threading.Thread(
        target=memory_generator_thread,
        args=(memory_store, stop_event),
        daemon=True
    )
    generator_thread.start()
    
    # Add some initial high-entropy memories
    add_high_entropy_memories(memory_store)
    
    # Main demo loop
    start_time = time.time()
    duration = 60  # Run for 1 minute
    
    try:
        while time.time() - start_time < duration:
            # Get current entropy
            entropy = monitor.get_current_entropy()
            now = datetime.now()
            
            # Update dashboard
            dashboard.add_entropy_data(now, entropy)
            dashboard.update_display()
            
            # Print status every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                logger.info(f"Current entropy: {entropy:.4f}, " 
                            f"Alert level: {monitor.current_alert.name}, "
                            f"Memories: {len(memory_store.get_all_memories())}")
                
                # Print dashboard stats
                stats = dashboard.get_stats()
                logger.info(f"Dashboard stats: {stats}")
                
                # Print detailed entropy metrics
                metrics = monitor.get_detailed_metrics()
                logger.info(f"Entropy metrics: {metrics}")
            
            # Sleep briefly
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    finally:
        # Clean up
        logger.info("Shutting down demo...")
        stop_event.set()
        monitor.stop()
        
        # Final plot update - non-interactive
        plt.ioff()
        # Update axis labels
        plt.title("Memory System Entropy Over Time")
        plt.tight_layout()
        plt.savefig("entropy_monitor_demo.png")
        plt.close()
        
        logger.info("Demo complete")
        logger.info(f"Final stats: {dashboard.get_stats()}")
        logger.info(f"Final entropy: {entropy:.4f}")
        logger.info(f"Final memory count: {len(memory_store.get_all_memories())}")


if __name__ == "__main__":
    main() 