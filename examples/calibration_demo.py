#!/usr/bin/env python3
"""
Calibration Logger Demo

This script demonstrates how to use the Empirical ΨC Calibration Logger to:
1. Track ΨC scores and activation patterns
2. Analyze correlations between ΨC scores and cognitive events
3. Determine optimal thresholds based on empirical data
4. Generate visualizations and reports
"""

import time
import random
import os
from typing import List, Dict, Any

from psi_c_ai_sdk.memory.memory import MemoryStore, Memory
from psi_c_ai_sdk.psi_c.psi_operator import PsiCOperator
from psi_c_ai_sdk.psi_c.toolkit import PsiToolkit
from psi_c_ai_sdk.monitor.calibration_logger import CalibrationLogger


def create_memory(content: str, importance: float = 1.0) -> Memory:
    """Create a memory with the given content and importance."""
    return Memory(content=content, importance=importance)


def run_simulation_cycle(
    toolkit: PsiToolkit, 
    memory_store: MemoryStore,
    calibration_logger: CalibrationLogger,
    with_reflection: bool = False
) -> None:
    """
    Run a simulation cycle, adding memories and logging ΨC state.
    
    Args:
        toolkit: PsiToolkit instance
        memory_store: MemoryStore instance
        calibration_logger: CalibrationLogger instance
        with_reflection: Whether to trigger a reflection event
    """
    # Add some memories
    for i in range(5):
        importance = 0.5 + (random.random() * 0.5)
        memory = create_memory(
            content=f"Memory {i} from cycle at {time.time()}",
            importance=importance
        )
        memory_store.add_memory(memory)
        
        # Log the state after each memory addition
        calibration_logger.log_toolkit_state(toolkit)
        
        # Short delay for better temporal distribution
        time.sleep(0.1)
    
    # If reflection is enabled, simulate a reflection event
    if with_reflection:
        # In a real system, this would be triggered by coherence metrics
        # Here we just manually mark it as a reflection event
        event = {
            "timestamp": time.time(),
            "event_type": "reflection",
            "schema_changed": random.random() > 0.5
        }
        
        # Add to toolkit's activation log
        if hasattr(toolkit, "_activation_log"):
            toolkit._activation_log.insert(0, event)
        
        # Log the reflection event
        calibration_logger.log_toolkit_state(toolkit)
        
        print("Simulated reflection event")


def run_calibration_demo():
    """Run the main calibration demo."""
    # Set up the log directory
    log_dir = "logs/calibration_demo"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Starting ΨC calibration demo, logs will be saved to {log_dir}")
    
    # Create the memory store
    memory_store = MemoryStore()
    
    # Create the ΨC operator with dynamic threshold
    psi_operator = PsiCOperator(
        memory_store,
        use_dynamic_threshold=True,
        dynamic_threshold_config={
            "base_threshold": 0.6,
            "sensitivity": 0.3
        }
    )
    
    # Create the toolkit
    toolkit = PsiToolkit(psi_operator)
    
    # Create the calibration logger
    calibration_logger = CalibrationLogger(
        log_dir=log_dir,
        session_id=f"demo_{int(time.time())}",
        log_interval=0.1  # Allow frequent logging for demo purposes
    )
    
    print("Simulation starting - will run 20 cycles with varying reflection patterns")
    
    # Run simulation cycles
    for cycle in range(20):
        print(f"\nCycle {cycle+1}/20")
        
        # Every 3rd cycle, trigger a reflection
        with_reflection = (cycle % 3 == 0)
        
        # Run the cycle
        run_simulation_cycle(
            toolkit=toolkit,
            memory_store=memory_store,
            calibration_logger=calibration_logger,
            with_reflection=with_reflection
        )
        
        # Log current ΨC state
        psi_score = toolkit.get_psi_index()
        psi_state = toolkit.get_psi_state()
        print(f"ΨC score: {psi_score:.4f}, State: {psi_state.name}")
        
        # Short delay between cycles
        time.sleep(0.5)
    
    # Save all logs
    log_path = calibration_logger.save_logs()
    print(f"\nCalibration logs saved to {log_path}")
    
    # Calculate correlations
    correlations = calibration_logger.calculate_correlations()
    print("\nCorrelations between ΨC score and cognitive events:")
    for key, value in correlations.items():
        print(f"  {key}: {value:.4f}")
    
    # Find optimal threshold
    optimal = calibration_logger.find_optimal_threshold()
    print("\nOptimal threshold analysis:")
    for key, value in optimal.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Generate calibration report
    report = calibration_logger.generate_calibration_report()
    print(f"\nGenerated calibration report with {report['statistics']['total_events']} events")
    
    # Plot the calibration data
    print("\nGenerating calibration plot...")
    plot_path = os.path.join(log_dir, f"{calibration_logger.session_id}_plot.png")
    calibration_logger.plot_calibration_data(save_path=plot_path)
    print(f"Plot saved to {plot_path}")
    
    print("\nCalibration demo complete!")


if __name__ == "__main__":
    run_calibration_demo() 