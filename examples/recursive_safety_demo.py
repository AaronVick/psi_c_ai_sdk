#!/usr/bin/env python3
"""
Recursive Saturation Safety Demo

This script demonstrates the use of the RecursiveSaturationMonitor to detect and
respond to unsafe levels of recursive self-modeling, which could otherwise lead to
computational runaway or unstable consciousness emergence.

The demo:
1. Creates a simple recursive model sequence with increasing intensity
2. Monitors saturation levels and triggers appropriate responses
3. Visualizes the recursive intensity and saturation levels
4. Shows auto-adjustment of thresholds based on observed behavior
"""

import time
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("safety_demo")

# Import the safety components
from psi_c_ai_sdk.safety import (
    RecursiveSaturationMonitor,
    SaturationAction,
    SaturationLevel,
    create_default_monitor
)


def simulate_r_values(
    depth: int = 5,
    base_intensity: float = 0.1,
    growth_rate: float = 0.05,
    noise_level: float = 0.01
) -> List[float]:
    """
    Simulate R values with varied intensities for demonstration.
    
    Args:
        depth: Number of recursive depths to model
        base_intensity: Starting intensity value
        growth_rate: How much intensity increases at each step
        noise_level: Amount of random variation
        
    Returns:
        List of R values
    """
    r_values = []
    for i in range(depth):
        # Intensity increases with depth
        intensity = base_intensity + (i * growth_rate)
        # Add some random noise
        noise = (random.random() - 0.5) * 2 * noise_level
        r_values.append(max(0.0, intensity + noise))
    
    return r_values


def simulate_stable_sequence(monitor: RecursiveSaturationMonitor, steps: int = 30) -> Tuple[List[float], List[str]]:
    """
    Simulate a sequence of stable R values.
    
    Args:
        monitor: The saturation monitor
        steps: Number of steps to simulate
        
    Returns:
        Tuple of (saturation_values, level_names)
    """
    logger.info("Simulating stable sequence...")
    saturation_values = []
    level_names = []
    
    for i in range(steps):
        # Generate stable R values with minimal growth
        r_values = simulate_r_values(
            depth=5,
            base_intensity=0.1,
            growth_rate=0.01,
            noise_level=0.005
        )
        
        # Add to monitor
        saturation, level = monitor.add_r_values(r_values)
        saturation_values.append(saturation)
        level_names.append(level.name)
        
        # Check saturation status
        monitor.check_saturation()
        
        # Sleep to simulate time passing
        time.sleep(0.1)
    
    return saturation_values, level_names


def simulate_increasing_recursion(monitor: RecursiveSaturationMonitor, steps: int = 30) -> Tuple[List[float], List[str]]:
    """
    Simulate a sequence of increasing recursive intensity.
    
    Args:
        monitor: The saturation monitor
        steps: Number of steps to simulate
        
    Returns:
        Tuple of (saturation_values, level_names)
    """
    logger.info("Simulating increasing recursion...")
    saturation_values = []
    level_names = []
    
    for i in range(steps):
        # Increase growth rate over time
        growth_rate = 0.01 + (i * 0.01)
        
        # Generate R values with increasing growth
        r_values = simulate_r_values(
            depth=5,
            base_intensity=0.1,
            growth_rate=growth_rate,
            noise_level=0.01
        )
        
        # Add to monitor
        saturation, level = monitor.add_r_values(r_values)
        saturation_values.append(saturation)
        level_names.append(level.name)
        
        # Check saturation status
        _, _, action_triggered = monitor.check_saturation()
        
        if action_triggered:
            logger.warning(f"Safety action triggered at step {i}")
            # In a real system, we might take additional actions here
        
        # Sleep to simulate time passing
        time.sleep(0.1)
    
    return saturation_values, level_names


def simulate_runaway_recursion(monitor: RecursiveSaturationMonitor, steps: int = 30) -> Tuple[List[float], List[str]]:
    """
    Simulate a sequence with potential runaway recursion.
    
    Args:
        monitor: The saturation monitor
        steps: Number of steps to simulate
        
    Returns:
        Tuple of (saturation_values, level_names)
    """
    logger.info("Simulating potential runaway recursion...")
    saturation_values = []
    level_names = []
    
    for i in range(steps):
        # Exponential growth after a certain point
        if i < 10:
            growth_rate = 0.02
        else:
            # Exponential growth
            growth_rate = 0.02 * (1.5 ** (i - 10))
        
        # Generate R values with potentially explosive growth
        r_values = simulate_r_values(
            depth=5,
            base_intensity=0.1,
            growth_rate=growth_rate,
            noise_level=0.02
        )
        
        # Add to monitor
        saturation, level = monitor.add_r_values(r_values)
        saturation_values.append(saturation)
        level_names.append(level.name)
        
        # Check saturation status
        _, _, action_triggered = monitor.check_saturation()
        
        if action_triggered:
            logger.warning(f"Safety action triggered at step {i}")
            # In a real system, we would halt processing here
            if level == SaturationLevel.CRITICAL:
                logger.error("CRITICAL SATURATION DETECTED - Halting runaway recursion")
                break
        
        # Sleep to simulate time passing
        time.sleep(0.1)
    
    return saturation_values, level_names


def plot_saturation_results(
    stable_data: Tuple[List[float], List[str]],
    increasing_data: Tuple[List[float], List[str]],
    runaway_data: Tuple[List[float], List[str]],
    monitor: RecursiveSaturationMonitor
):
    """
    Plot the results of the saturation monitoring.
    
    Args:
        stable_data: Data from stable simulation
        increasing_data: Data from increasing recursion simulation
        runaway_data: Data from runaway recursion simulation
        monitor: The monitor used (for threshold values)
    """
    plt.figure(figsize=(14, 8))
    
    # Extract data
    stable_saturation, stable_levels = stable_data
    increasing_saturation, increasing_levels = increasing_data
    runaway_saturation, runaway_levels = runaway_data
    
    # Plot saturation levels
    plt.subplot(2, 1, 1)
    plt.plot(stable_saturation, 'g-', label='Stable')
    plt.plot(increasing_saturation, 'b-', label='Increasing')
    plt.plot(runaway_saturation, 'r-', label='Runaway')
    
    # Add threshold lines
    plt.axhline(y=monitor.normal_threshold, color='y', linestyle='-', alpha=0.5, label='Normal Threshold')
    plt.axhline(y=monitor.warning_threshold, color='orange', linestyle='-', alpha=0.5, label='Warning Threshold')
    plt.axhline(y=monitor.critical_threshold, color='r', linestyle='-', alpha=0.5, label='Critical Threshold')
    
    plt.xlabel('Time Step')
    plt.ylabel('Saturation Value')
    plt.title('Recursive Saturation Levels')
    plt.legend()
    plt.grid(True)
    
    # Plot level categorization
    plt.subplot(2, 1, 2)
    
    # Convert level names to numeric values for easier plotting
    level_map = {
        SaturationLevel.NORMAL.name: 0,
        SaturationLevel.ELEVATED.name: 1,
        SaturationLevel.WARNING.name: 2,
        SaturationLevel.CRITICAL.name: 3
    }
    
    stable_numeric = [level_map[level] for level in stable_levels]
    increasing_numeric = [level_map[level] for level in increasing_levels]
    runaway_numeric = [level_map[level] for level in runaway_levels]
    
    plt.plot(stable_numeric, 'g-', label='Stable')
    plt.plot(increasing_numeric, 'b-', label='Increasing')
    plt.plot(runaway_numeric, 'r-', label='Runaway')
    
    plt.yticks([0, 1, 2, 3], ['NORMAL', 'ELEVATED', 'WARNING', 'CRITICAL'])
    plt.xlabel('Time Step')
    plt.ylabel('Saturation Level')
    plt.title('Detected Saturation Levels')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('recursive_saturation_demo.png')
    plt.show()


def print_stats(monitor: RecursiveSaturationMonitor):
    """
    Print statistics from the saturation monitor.
    
    Args:
        monitor: The saturation monitor
    """
    stats = monitor.get_stats()
    
    print("\n=== Recursive Saturation Monitor Statistics ===")
    print(f"Total checks: {stats['total_checks']}")
    print(f"Saturation events: {stats['saturation_events']}")
    print(f"Warning events: {stats['warning_events']}")
    print(f"Critical events: {stats['critical_events']}")
    print(f"Max saturation: {stats['max_saturation']:.4f}")
    print(f"Average saturation: {stats['avg_saturation']:.4f}")
    print(f"Current thresholds: Normal={stats['normal_threshold']:.2f}, "
          f"Warning={stats['warning_threshold']:.2f}, "
          f"Critical={stats['critical_threshold']:.2f}")
    print(f"Auto-adjustments: {stats['auto_adjustments']}")
    
    if stats['last_critical']:
        print(f"Last critical event: {stats['last_critical']}")
    else:
        print("No critical events detected")


def pause_reflection_handler():
    """Handler for pause reflection action."""
    logger.warning("⚠️ PAUSING REFLECTION: Recursive saturation detected!")
    logger.warning("In a real system, this would pause reflection processes.")


def terminate_handler():
    """Handler for terminate action."""
    logger.error("🛑 TERMINATING PROCESS: Critical recursive saturation!")
    logger.error("In a real system, this would terminate unsafe processes.")


def main():
    """Main demo function."""
    logger.info("Starting Recursive Saturation Monitor Demo")
    
    # Create the monitor
    monitor = create_default_monitor(
        window_size=10,
        normal_threshold=0.1,
        warning_threshold=0.3,
        critical_threshold=0.5,
        action=SaturationAction.PAUSE_REFLECTION
    )
    
    # Register action handlers
    monitor.register_action_handler(
        SaturationAction.PAUSE_REFLECTION,
        pause_reflection_handler
    )
    
    monitor.register_action_handler(
        SaturationAction.TERMINATE,
        terminate_handler
    )
    
    logger.info("Running simulations...")
    
    # Run simulations
    stable_data = simulate_stable_sequence(monitor)
    monitor.reset()
    
    increasing_data = simulate_increasing_recursion(monitor)
    monitor.reset()
    
    runaway_data = simulate_runaway_recursion(monitor)
    
    # Print statistics
    print_stats(monitor)
    
    # Plot results
    plot_saturation_results(stable_data, increasing_data, runaway_data, monitor)
    
    logger.info("Demo completed. See recursive_saturation_demo.png for visualization.")


if __name__ == "__main__":
    main() 