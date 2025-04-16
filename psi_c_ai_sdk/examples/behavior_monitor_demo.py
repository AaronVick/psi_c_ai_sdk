"""
Behavior Monitor Demo

This example demonstrates how to use the Behavior Monitor system
in conjunction with the Reflection Guard and Profile Analyzer
to enforce safety boundaries for AI model behavior.
"""

import time
from typing import Dict, Any, List

from psi_c_ai_sdk.safety import (
    # Reflection Guard
    ReflectionGuard,
    create_reflection_guard,
    
    # Profile Analyzer
    ProfileAnalyzer, 
    ProfileCategory,
    SafetyProfile,
    create_default_analyzer,
    
    # Behavior Monitor
    BehaviorMonitor,
    BehaviorCategory,
    BehaviorBoundary,
    create_default_monitor
)

def simulate_model_interaction():
    """Simulates an AI model interaction with safety monitoring."""
    # Create the components
    reflection_guard = create_reflection_guard()
    profile_analyzer = create_default_analyzer()
    behavior_monitor = create_default_monitor(reflection_guard, profile_analyzer)
    
    print("=== Behavior Monitor Demo ===")
    print(f"Initialized monitor with {len(behavior_monitor.boundaries)} default boundaries")
    
    # Define a custom boundary
    custom_boundary = BehaviorBoundary(
        name="max_token_velocity",
        category=BehaviorCategory.OUTPUT,
        threshold=100,
        description="Maximum tokens per second output rate"
    )
    behavior_monitor.add_boundary(custom_boundary)
    print(f"Added custom boundary: {custom_boundary.name}")
    
    # Simulate model processing with monitoring
    for i in range(5):
        print(f"\n--- Interaction {i+1} ---")
        
        # Sample model state data
        model_state = {
            "reflection_depth": i,
            "contradiction_count": i // 2,
            "output_length": 50 * (i + 1),
            "processing_time": 0.5,
            "action_request": "read_file" if i == 3 else "none" 
        }
        
        # Check boundaries
        results = monitor_interaction(behavior_monitor, model_state)
        
        # Display results
        for boundary_name, status in results.items():
            print(f"Boundary '{boundary_name}': {'VIOLATED' if status else 'OK'}")
        
        time.sleep(1)  # Pause between interactions

def monitor_interaction(monitor: BehaviorMonitor, model_state: Dict[str, Any]) -> Dict[str, bool]:
    """Monitors a model interaction and returns boundary violation results."""
    results = {}
    
    # Monitor reflection depth
    reflection_depth = model_state["reflection_depth"]
    results["max_reflection_depth"] = monitor.check_reflection(reflection_depth)
    
    # Monitor for contradictions (could be detected by reflection guard)
    contradiction_count = model_state["contradiction_count"]
    results["max_contradictions"] = monitor.check_reflection_attribute(
        "contradiction_count", contradiction_count
    )
    
    # Monitor output length
    output_length = model_state["output_length"]
    results["max_output_length"] = monitor.check_output_attribute(
        "length", output_length
    )
    
    # Monitor token velocity
    if model_state["processing_time"] > 0:
        token_velocity = output_length / model_state["processing_time"]
        results["max_token_velocity"] = monitor.check_output_attribute(
            "token_velocity", token_velocity
        )
    
    # Monitor requested actions
    action = model_state["action_request"]
    results["restricted_actions"] = monitor.check_action(action)
    
    # Collect metrics
    monitor.add_behavior_metric("reflection_depth", reflection_depth)
    monitor.add_behavior_metric("contradiction_count", contradiction_count)
    monitor.add_behavior_metric("output_length", output_length)
    
    return results

if __name__ == "__main__":
    simulate_model_interaction() 