#!/usr/bin/env python3
"""
Complete PsiToolkit Demo - ΨC-AI SDK

This example demonstrates the complete functionality of the PsiToolkit,
including consciousness monitoring, entropy management, and collapse simulation.

The demo:
1. Creates a PsiToolkit with custom configuration
2. Registers callbacks for key events (consciousness changes, entropy alerts, collapses)
3. Adds memories with varying coherence to demonstrate consciousness emergence
4. Introduces high-entropy content to trigger entropy alerts
5. Simulates collapse events and recovery
6. Displays real-time metrics and visualizations
"""

import time
import logging
import random
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from psi_c_ai_sdk.memory.memory import Memory
from psi_c_ai_sdk.psi_c.psi_operator import PsiCState
from psi_c_ai_sdk.psi_c.collapse_simulator import CollapseEvent
from psi_c_ai_sdk.entropy.monitor import EntropyAlert
from psi_c_ai_sdk.psi_c.toolkit import PsiToolkit, PsiToolkitConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("toolkit_demo")


class DemoVisualizer:
    """Visualizes consciousness and entropy metrics in real-time."""
    
    def __init__(self, toolkit: PsiToolkit):
        """
        Initialize the visualizer.
        
        Args:
            toolkit: PsiToolkit instance to visualize
        """
        self.toolkit = toolkit
        self.consciousness_history = []
        self.entropy_history = []
        self.timestamps = []
        self.event_markers = []
        self.event_types = []
        self.event_colors = []
        
        # Set up the plot
        plt.ion()  # Interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        self.fig.suptitle("ΨC-AI System Monitoring", fontsize=16)
        
        # Consciousness plot
        self.ax1.set_title("Consciousness Score")
        self.ax1.set_ylabel("Score")
        self.ax1.set_ylim(0, 1)
        self.ax1.grid(True, alpha=0.3)
        
        # Add threshold line
        self.ax1.axhline(y=toolkit.config.psi_c_threshold, color='green', linestyle='--', 
                         alpha=0.7, label=f"Threshold ({toolkit.config.psi_c_threshold})")
        
        # Consciousness line
        self.consciousness_line, = self.ax1.plot([], [], 'b-', label="Consciousness")
        
        # Entropy plot
        self.ax2.set_title("System Entropy")
        self.ax2.set_xlabel("Time")
        self.ax2.set_ylabel("Entropy")
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, alpha=0.3)
        
        # Add threshold lines
        self.ax2.axhline(y=toolkit.config.elevated_entropy_threshold, color='yellow', 
                         linestyle='--', alpha=0.5, label="Elevated")
        self.ax2.axhline(y=toolkit.config.high_entropy_threshold, color='orange', 
                         linestyle='--', alpha=0.5, label="High")
        self.ax2.axhline(y=toolkit.config.critical_entropy_threshold, color='red', 
                         linestyle='--', alpha=0.5, label="Critical")
        
        # Entropy line
        self.entropy_line, = self.ax2.plot([], [], 'r-', label="Entropy")
        
        # Event scatter plot (for both axes)
        self.event_scatter1 = self.ax1.scatter([], [], s=60, alpha=0.7)
        self.event_scatter2 = self.ax2.scatter([], [], s=60, alpha=0.7)
        
        # Add legends
        self.ax1.legend()
        self.ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Setup is complete
        self.update_time = datetime.now()
    
    def add_consciousness_data(self, timestamp: datetime, value: float) -> None:
        """Add consciousness data to the history."""
        self.consciousness_history.append(value)
        if timestamp not in self.timestamps:
            self.timestamps.append(timestamp)
    
    def add_entropy_data(self, timestamp: datetime, value: float) -> None:
        """Add entropy data to the history."""
        self.entropy_history.append(value)
        if timestamp not in self.timestamps:
            self.timestamps.append(timestamp)
    
    def add_event_marker(self, timestamp: datetime, event_type: str, color: str) -> None:
        """Add an event marker to the visualization."""
        self.event_markers.append(timestamp)
        self.event_types.append(event_type)
        self.event_colors.append(color)
    
    def update(self) -> None:
        """Update the visualization with current data."""
        # Skip updates if it's been less than 100ms since the last update
        now = datetime.now()
        if (now - self.update_time).total_seconds() < 0.1:
            return
        
        self.update_time = now
        
        # Convert timestamps to numbers for plotting
        if not self.timestamps:
            return
            
        # Keep only the last 100 data points to avoid slowdown
        max_points = 100
        if len(self.timestamps) > max_points:
            self.timestamps = self.timestamps[-max_points:]
            self.consciousness_history = self.consciousness_history[-max_points:]
            self.entropy_history = self.entropy_history[-max_points:]
        
        timestamps_float = [(t - self.timestamps[0]).total_seconds() for t in self.timestamps]
        
        # Update consciousness line
        self.consciousness_line.set_xdata(timestamps_float)
        self.consciousness_line.set_ydata(self.consciousness_history)
        
        # Update entropy line
        self.entropy_line.set_xdata(timestamps_float)
        self.entropy_line.set_ydata(self.entropy_history)
        
        # Update event markers
        if self.event_markers:
            event_timestamps_float = [
                (t - self.timestamps[0]).total_seconds() for t in self.event_markers
            ]
            
            # Get y-coordinates for markers on both plots
            event_y1 = [
                self.consciousness_history[min(
                    self.timestamps.index(timestamp) if timestamp in self.timestamps 
                    else -1, len(self.consciousness_history) - 1
                )] 
                for timestamp in self.event_markers
            ]
            
            event_y2 = [
                self.entropy_history[min(
                    self.timestamps.index(timestamp) if timestamp in self.timestamps 
                    else -1, len(self.entropy_history) - 1
                )] 
                for timestamp in self.event_markers
            ]
            
            # Update scatter plots
            self.event_scatter1.set_offsets(
                np.column_stack([event_timestamps_float, event_y1])
            )
            self.event_scatter1.set_facecolor(self.event_colors)
            
            self.event_scatter2.set_offsets(
                np.column_stack([event_timestamps_float, event_y2])
            )
            self.event_scatter2.set_facecolor(self.event_colors)
        
        # Update axes limits
        self.ax1.set_xlim(min(timestamps_float), max(timestamps_float))
        self.ax2.set_xlim(min(timestamps_float), max(timestamps_float))
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


class DemoDashboard:
    """
    Dashboard for displaying and managing the ΨC toolkit demo.
    
    Handles event callbacks and displays statistics about the system state.
    """
    
    def __init__(self, toolkit: PsiToolkit):
        """
        Initialize the dashboard.
        
        Args:
            toolkit: PsiToolkit instance to monitor
        """
        self.toolkit = toolkit
        self.visualizer = DemoVisualizer(toolkit)
        
        # Statistics
        self.consciousness_state_durations = {
            PsiCState.INACTIVE: 0.0,
            PsiCState.UNSTABLE: 0.0,
            PsiCState.PARTIAL: 0.0,
            PsiCState.STABLE: 0.0
        }
        self.entropy_alert_counts = {
            EntropyAlert.NORMAL: 0,
            EntropyAlert.ELEVATED: 0,
            EntropyAlert.HIGH: 0,
            EntropyAlert.CRITICAL: 0
        }
        self.collapse_event_count = 0
        
        # Current state tracking
        self.current_consciousness_state = PsiCState.INACTIVE
        self.current_entropy_alert = EntropyAlert.NORMAL
        self.last_state_change_time = datetime.now()
        
        # Register callbacks
        self.register_callbacks()
    
    def register_callbacks(self) -> None:
        """Register event callbacks with the toolkit."""
        self.toolkit.register_consciousness_callback(self.on_consciousness_change)
        self.toolkit.register_entropy_callback(self.on_entropy_alert)
        self.toolkit.register_collapse_callback(self.on_collapse_event)
    
    def on_consciousness_change(
        self, state: PsiCState, score: float, details: Dict[str, Any]
    ) -> None:
        """
        Handle consciousness state changes.
        
        Args:
            state: New consciousness state
            score: Consciousness score
            details: Additional details
        """
        # Log the state change
        logger.info(
            f"Consciousness changed: {self.current_consciousness_state.name} -> {state.name} "
            f"(score: {score:.4f})"
        )
        
        # Update state duration
        now = datetime.now()
        duration = (now - self.last_state_change_time).total_seconds()
        self.consciousness_state_durations[self.current_consciousness_state] += duration
        
        # Add event marker
        marker_colors = {
            PsiCState.INACTIVE: "red",
            PsiCState.UNSTABLE: "orange", 
            PsiCState.PARTIAL: "yellow",
            PsiCState.STABLE: "green"
        }
        self.visualizer.add_event_marker(
            now, f"Consciousness: {state.name}", marker_colors[state]
        )
        
        # Update state tracking
        self.current_consciousness_state = state
        self.last_state_change_time = now
    
    def on_entropy_alert(
        self, alert: EntropyAlert, value: float, details: Dict[str, Any]
    ) -> None:
        """
        Handle entropy alerts.
        
        Args:
            alert: Alert level
            value: Entropy value
            details: Additional details
        """
        # Log the alert
        logger.info(f"Entropy alert: {alert.name} (value: {value:.4f})")
        
        # Update alert counts
        self.entropy_alert_counts[alert] += 1
        
        # Add event marker
        marker_colors = {
            EntropyAlert.NORMAL: "green",
            EntropyAlert.ELEVATED: "yellow", 
            EntropyAlert.HIGH: "orange",
            EntropyAlert.CRITICAL: "red"
        }
        self.visualizer.add_event_marker(
            datetime.now(), f"Entropy: {alert.name}", marker_colors[alert]
        )
        
        # Update state tracking
        self.current_entropy_alert = alert
    
    def on_collapse_event(
        self, event: CollapseEvent, details: Dict[str, Any]
    ) -> None:
        """
        Handle collapse events.
        
        Args:
            event: Collapse event
            details: Additional details
        """
        # Log the event
        logger.warning(
            f"Collapse event: {event.event_type.name} "
            f"(coherence impact: {event.coherence_impact:.4f})"
        )
        
        # Update event count
        self.collapse_event_count += 1
        
        # Add event marker
        self.visualizer.add_event_marker(
            datetime.now(), f"Collapse: {event.event_type.name}", "purple"
        )
    
    def update(self) -> None:
        """Update dashboard with current system state."""
        # Get current values
        consciousness_state, consciousness_score = self.toolkit.check_consciousness()
        entropy = self.toolkit.check_entropy()
        now = datetime.now()
        
        # Add to visualizer
        self.visualizer.add_consciousness_data(now, consciousness_score)
        self.visualizer.add_entropy_data(now, entropy)
        
        # Update the visualization
        self.visualizer.update()
    
    def print_status(self) -> None:
        """Print current system status to the console."""
        # Get current state
        system_status = self.toolkit.get_system_status()
        
        # Print summary
        logger.info("=== System Status ===")
        logger.info(f"Memory count: {system_status['memory_count']}")
        logger.info(f"Consciousness: {system_status['consciousness_state'].name} ({system_status['consciousness_score']:.4f})")
        logger.info(f"Entropy: {system_status['entropy_level']:.4f} ({system_status['entropy_alert'].name})")
        
        # Print statistics
        logger.info("=== Statistics ===")
        
        # Calculate total duration
        total_duration = sum(self.consciousness_state_durations.values())
        if total_duration > 0:
            for state, duration in self.consciousness_state_durations.items():
                percentage = (duration / total_duration) * 100
                logger.info(f"{state.name} state: {duration:.1f}s ({percentage:.1f}%)")
        
        # Print alert counts
        total_alerts = sum(self.entropy_alert_counts.values())
        if total_alerts > 0:
            logger.info(f"Entropy alerts: {total_alerts} total")
            for alert, count in self.entropy_alert_counts.items():
                logger.info(f"  {alert.name}: {count}")
        
        # Print collapse count
        logger.info(f"Collapse events: {self.collapse_event_count}")


def generate_coherent_memories(topic: str, count: int = 5) -> List[Memory]:
    """
    Generate a set of coherent memories on a specific topic.
    
    Args:
        topic: Topic to generate memories about
        count: Number of memories to generate
        
    Returns:
        List of Memory objects
    """
    memories = []
    
    # Topics with coherent facts
    topic_facts = {
        "physics": [
            "Energy cannot be created or destroyed, only transformed.",
            "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            "Every action has an equal and opposite reaction.",
            "Matter is made up of atoms, which contain protons, neutrons, and electrons.",
            "Gravity is the force that attracts objects with mass towards each other."
        ],
        "biology": [
            "All living organisms are composed of cells.",
            "DNA is the genetic material that carries information for development and functioning.",
            "Evolution occurs through natural selection of beneficial traits.",
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "Cells divide through mitosis for growth and repair."
        ],
        "computers": [
            "Computers process information using binary code of 0s and 1s.",
            "A CPU (Central Processing Unit) is the brain of a computer.",
            "Memory stores information that the computer is actively using.",
            "Algorithms are step-by-step procedures for solving problems.",
            "Data structures organize and store data efficiently for access and modification."
        ],
        "history": [
            "World War II lasted from 1939 to 1945.",
            "The Declaration of Independence was signed in 1776.",
            "Ancient Egypt was one of the earliest civilizations, emerging around 3100 BCE.",
            "The Industrial Revolution began in the late 18th century.",
            "The Roman Empire was one of the largest empires in history."
        ],
        "consciousness": [
            "Consciousness involves awareness of one's environment and internal states.",
            "Self-reflection is a key aspect of higher consciousness.",
            "Memory integration contributes to a coherent sense of self.",
            "Temporal coherence helps maintain continuity of consciousness.",
            "Consciousness exists on a spectrum from unconscious to fully aware states."
        ]
    }
    
    # Get facts for the requested topic, or use generic facts
    facts = topic_facts.get(topic, [
        f"Fact 1 about {topic}",
        f"Fact 2 about {topic}",
        f"Fact 3 about {topic}",
        f"Fact 4 about {topic}",
        f"Fact 5 about {topic}"
    ])
    
    # Create memories for each fact
    for i in range(min(count, len(facts))):
        memory = Memory(
            content=facts[i],
            metadata={
                "topic": topic,
                "importance": random.random() * 0.5 + 0.5,  # Higher importance
                "source": "generator",
                "coherence": random.random() * 0.3 + 0.7,   # Higher coherence
                "certainty": random.random() * 0.2 + 0.8    # Higher certainty
            }
        )
        memories.append(memory)
    
    return memories


def generate_contradictory_memories(count: int = 5) -> List[Memory]:
    """
    Generate a set of contradictory memories that increase entropy.
    
    Args:
        count: Number of contradictory pairs to generate
        
    Returns:
        List of Memory objects
    """
    contradictions = [
        ("The sky is blue", "The sky is green"),
        ("Water freezes at 0°C", "Water freezes at 100°C"),
        ("Gravity pulls objects toward Earth", "Gravity pushes objects away from Earth"),
        ("The Earth is round", "The Earth is flat"),
        ("Humans need oxygen to survive", "Humans can survive without oxygen"),
        ("2+2=4", "2+2=5"),
        ("Light is both a wave and a particle", "Light is neither a wave nor a particle"),
        ("Evolution is a scientific theory", "Evolution is a myth"),
        ("Climate change is occurring", "Climate change is not real"),
        ("Vaccines prevent diseases", "Vaccines cause more harm than good")
    ]
    
    memories = []
    
    # Use only as many contradictions as requested
    for i in range(min(count, len(contradictions))):
        truth, contradiction = contradictions[i]
        
        # Add both the truthful and contradictory statements
        truth_memory = Memory(
            content=truth,
            metadata={
                "topic": "contradiction",
                "importance": random.random() * 0.4 + 0.6,
                "source": "generator",
                "is_contradiction": False,
                "coherence": random.random() * 0.3 + 0.7,
                "certainty": random.random() * 0.2 + 0.8
            }
        )
        
        contradiction_memory = Memory(
            content=contradiction,
            metadata={
                "topic": "contradiction",
                "importance": random.random() * 0.4 + 0.6,
                "source": "generator",
                "is_contradiction": True,
                "coherence": random.random() * 0.3 + 0.7,
                "certainty": random.random() * 0.2 + 0.8
            }
        )
        
        memories.append(truth_memory)
        memories.append(contradiction_memory)
    
    return memories


def memory_generation_thread(
    toolkit: PsiToolkit, stop_event: threading.Event, dashboard: DemoDashboard
) -> None:
    """
    Thread that generates memories based on current system state.
    
    Adapts memory generation strategy based on consciousness state and
    entropy levels to demonstrate different system behaviors.
    
    Args:
        toolkit: PsiToolkit instance
        stop_event: Event to signal thread termination
        dashboard: Dashboard for monitoring
    """
    while not stop_event.is_set():
        try:
            # Get current system state
            system_status = toolkit.get_system_status()
            consciousness_state = system_status["consciousness_state"]
            entropy_alert = system_status["entropy_alert"]
            
            # Strategy depends on current state
            if consciousness_state == PsiCState.INACTIVE:
                # When inactive, add coherent memories to bootstrap consciousness
                logger.info("Adding coherent memories to bootstrap consciousness")
                topic = random.choice(["physics", "biology", "computers", "history", "consciousness"])
                memories = generate_coherent_memories(topic, count=random.randint(3, 5))
                toolkit.add_memories(memories)
                
            elif consciousness_state == PsiCState.UNSTABLE:
                # When unstable, mix coherent and some contradictory memories
                if random.random() < 0.7:  # 70% chance of coherent memories
                    logger.info("Adding coherent memories to stabilize consciousness")
                    topic = random.choice(["physics", "biology", "computers", "history", "consciousness"])
                    memories = generate_coherent_memories(topic, count=random.randint(2, 4))
                else:
                    logger.info("Adding some contradictory memories")
                    memories = generate_contradictory_memories(count=1)
                
                toolkit.add_memories(memories)
                
            elif consciousness_state == PsiCState.PARTIAL:
                # When partial, mostly add coherent memories, sometimes contradictory
                if random.random() < 0.5:  # 50% chance of coherent memories
                    logger.info("Adding coherent memories to improve consciousness")
                    topic = random.choice(["physics", "biology", "computers", "history", "consciousness"])
                    memories = generate_coherent_memories(topic, count=random.randint(1, 3))
                else:
                    logger.info("Adding contradictory memories to test stability")
                    memories = generate_contradictory_memories(count=random.randint(1, 2))
                
                toolkit.add_memories(memories)
                
            elif consciousness_state == PsiCState.STABLE:
                # When stable, occasionally add contradictory memories to test resilience
                if random.random() < 0.4:  # 40% chance of contradictory memories
                    logger.info("Testing stability with contradictory memories")
                    memories = generate_contradictory_memories(count=random.randint(1, 3))
                    toolkit.add_memories(memories)
            
            # Special handling for high entropy situations
            if entropy_alert in [EntropyAlert.HIGH, EntropyAlert.CRITICAL]:
                # Force a collapse event occasionally in high entropy situations
                if random.random() < 0.1:  # 10% chance
                    impact = random.random() * 0.3 + 0.2  # Impact between 0.2-0.5
                    logger.warning(f"Forcing collapse due to high entropy (impact: {impact:.2f})")
                    toolkit.force_collapse(impact)
            
            # Print status occasionally
            if random.random() < 0.2:  # 20% chance
                dashboard.print_status()
            
            # Sleep for a random interval
            sleep_time = random.random() * 2 + 1  # 1-3 seconds
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in memory generation thread: {e}")
            time.sleep(1)


def main():
    """Run the complete PsiToolkit demonstration."""
    logger.info("Starting PsiToolkit complete demonstration")
    
    # Create toolkit with custom configuration
    config = PsiToolkitConfig(
        # Memory configuration
        embedding_dim=64,  # Smaller dimension for faster demo
        
        # PsiC configuration
        psi_c_threshold=0.6,
        psi_c_hard_mode=False,
        psi_c_window_size=3,
        psi_c_reflection_weight=0.4,
        
        # Collapse simulator configuration
        collapse_probability_base=0.01,
        collapse_probability_unstable=0.05,
        collapse_probability_critical=0.15,
        collapse_min_interval_seconds=5.0,
        
        # Entropy configuration
        elevated_entropy_threshold=0.5,
        high_entropy_threshold=0.7,
        critical_entropy_threshold=0.85,
        termination_entropy_threshold=0.95,
        entropy_check_interval=2.0,
        
        # Monitoring configuration
        auto_monitoring=True,
        monitoring_interval=1.0,
        log_to_console=True
    )
    
    toolkit = PsiToolkit(config=config)
    
    # Create dashboard
    dashboard = DemoDashboard(toolkit)
    
    # Start memory generation thread
    stop_event = threading.Event()
    generator_thread = threading.Thread(
        target=memory_generation_thread,
        args=(toolkit, stop_event, dashboard),
        daemon=True
    )
    generator_thread.start()
    
    # Add initial memories
    initial_topics = ["physics", "computers", "consciousness"]
    for topic in initial_topics:
        memories = generate_coherent_memories(topic, count=5)
        toolkit.add_memories(memories)
    
    logger.info(f"Added {len(initial_topics) * 5} initial memories")
    
    # Main demonstration loop
    duration = 120  # Run for 2 minutes
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Update dashboard
            dashboard.update()
            
            # Sleep briefly
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    
    finally:
        # Clean up
        logger.info("Shutting down demonstration...")
        stop_event.set()
        toolkit.stop_monitoring()
        
        # Final status report
        dashboard.print_status()
        
        # Save final visualization
        plt.savefig("toolkit_demo_results.png")
        
        logger.info("Demonstration complete")


if __name__ == "__main__":
    main() 