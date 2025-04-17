#!/usr/bin/env python3
"""
ΨC-AI SDK Orchestration Demo

This example demonstrates how to use the Master Orchestration Script to run a simple
experiment with the ΨC system. It shows:
1. How to set up all components
2. How to run the coherence cycle
3. How to visualize the results over time

This serves as a minimal working example of the complete system.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Ensure the psi_c_ai_sdk package is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.schema import SchemaGraph
from psi_c_ai_sdk.coherence import CoherenceCalculator
from psi_c_ai_sdk.reflection import ReflectionEngine, ReflectionScheduler
from psi_c_ai_sdk.contradiction import ContradictionDetector
from psi_c_ai_sdk.core.orchestration import CycleController, InputProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestration_demo")


def create_system_components():
    """Create all components for the ΨC system."""
    # Create memory store
    memory_store = MemoryStore()
    
    # Create schema graph
    schema_graph = SchemaGraph(memory_store)
    
    # Create reflection components
    reflection_scheduler = ReflectionScheduler(
        coherence_threshold=0.5,
        min_interval=1.0  # Low for demo purposes
    )
    
    reflection_engine = ReflectionEngine(
        memory_store=memory_store,
        scheduler=reflection_scheduler
    )
    
    # Create contradiction detector
    contradiction_detector = ContradictionDetector(
        memory_store=memory_store,
        schema_graph=schema_graph
    )
    
    # Create coherence calculator
    coherence_calculator = CoherenceCalculator(
        memory_store=memory_store,
        schema_graph=schema_graph
    )
    
    # Create input processor
    input_processor = InputProcessor(
        memory_store=memory_store,
        source_trust_levels={
            "high_trust_source": 0.9,
            "low_trust_source": 0.3
        }
    )
    
    # Create cycle controller
    cycle_controller = CycleController(
        memory_store=memory_store,
        schema_graph=schema_graph,
        reflection_engine=reflection_engine,
        contradiction_detector=contradiction_detector,
        coherence_calculator=coherence_calculator,
        input_processor=input_processor,
        log_dir="./demo_logs"
    )
    
    return {
        "memory_store": memory_store,
        "schema_graph": schema_graph,
        "reflection_engine": reflection_engine,
        "contradiction_detector": contradiction_detector,
        "coherence_calculator": coherence_calculator,
        "input_processor": input_processor,
        "cycle_controller": cycle_controller
    }


def run_demo():
    """Run the orchestration demo."""
    print("====== ΨC-AI SDK Orchestration Demo ======")
    print("This demo shows the complete ΨC cycle in action.")
    
    # Create components
    print("\n[1] Creating system components...")
    components = create_system_components()
    cycle_controller = components["cycle_controller"]
    
    # Prepare demo data - simple scenario with contradictions
    print("\n[2] Preparing data and running cycles...")
    demo_inputs = [
        {
            "content": "The sky is blue during a clear day.",
            "importance": 0.8
        },
        {
            "content": "Water boils at 100 degrees Celsius at sea level.",
            "importance": 0.9
        },
        {
            "content": "The Earth revolves around the Sun.",
            "importance": 0.95
        },
        {
            "content": "The boiling point of water depends on atmospheric pressure.",
            "importance": 0.85
        },
        {
            "content": "At high altitudes, water boils at lower temperatures.",
            "importance": 0.8
        },
        {
            "content": "The sky appears red during sunset.",
            "importance": 0.7
        },
        {
            "content": "The sky is green during a clear day.",
            "importance": 0.6,
            "source": "low_trust_source"
        },
        {
            "content": "Water boils at exactly 90 degrees Celsius at sea level.",
            "importance": 0.5,
            "source": "low_trust_source"
        }
    ]
    
    # Process inputs
    memory_ids = []
    for i, input_data in enumerate(demo_inputs):
        print(f"  Ingesting input {i+1}/{len(demo_inputs)}: {input_data['content']}")
        memory_id = cycle_controller.ingest(
            content=input_data["content"],
            importance=input_data.get("importance"),
            source=input_data.get("source")
        )
        memory_ids.append(memory_id)
        
        # Run a cycle after every other input
        if i % 2 == 1:
            print(f"  Running cycle after input {i+1}...")
            cycle_controller.run_cycle()
            
            # Print current coherence
            coherence_history = cycle_controller.get_coherence_history()
            if coherence_history:
                current_coherence = coherence_history[-1][1]
                print(f"  Current coherence: {current_coherence:.4f}")
    
    # Run a few more cycles to stabilize
    print("\n[3] Running additional cycles for stabilization...")
    
    for i in range(3):
        print(f"  Running stabilization cycle {i+1}/3...")
        cycle_controller.run_cycle()
        
        # Print current coherence
        coherence_history = cycle_controller.get_coherence_history()
        if coherence_history:
            current_coherence = coherence_history[-1][1]
            print(f"  Current coherence: {current_coherence:.4f}")
    
    # Print results
    print("\n[4] Experiment completed!")
    cycle_stats = cycle_controller.get_cycle_stats()
    
    print("\nCoherence progression:")
    coherence_history = cycle_controller.get_coherence_history()
    for i, (timestamp, coherence) in enumerate(coherence_history):
        print(f"  Cycle {i+1}: {coherence:.4f}")
    
    print("\nFinal statistics:")
    print(f"  Total cycles run: {cycle_stats['cycles_run']}")
    print(f"  Average cycle duration: {cycle_stats['avg_duration']:.2f}s")
    print(f"  Initial coherence: {cycle_stats['initial_coherence']:.4f}")
    print(f"  Final coherence: {cycle_stats['latest_coherence']:.4f}")
    print(f"  Coherence change: {cycle_stats['latest_coherence'] - cycle_stats['initial_coherence']:.4f}")
    print(f"  Coherence trend: {cycle_stats['coherence_trend']:.4f}")
    
    # Create visualization if bokeh is available
    try:
        import bokeh.plotting as plt
        from bokeh.models import ColumnDataSource
        from bokeh.embed import file_html
        from bokeh.resources import CDN
        
        print("\n[5] Creating visualization...")
        
        # Create coherence plot
        p = plt.figure(
            title="Coherence Over Time", 
            x_axis_label="Cycle",
            y_axis_label="Coherence Score",
            width=800, height=400
        )
        
        # Extract cycle numbers and coherence values
        cycles = list(range(len(coherence_history)))
        coherence_values = [v for _, v in coherence_history]
        
        source = ColumnDataSource(data=dict(cycles=cycles, coherence=coherence_values))
        
        p.line('cycles', 'coherence', source=source, line_width=2)
        p.circle('cycles', 'coherence', source=source, size=8, color="red")
        
        # Save visualization
        os.makedirs("./demo_logs", exist_ok=True)
        viz_file = os.path.join("./demo_logs", f"coherence_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        with open(viz_file, 'w') as f:
            f.write(file_html(p, CDN, "ΨC Coherence Plot"))
            
        print(f"  Visualization saved to {viz_file}")
        
    except ImportError:
        print("\n[5] Bokeh not available, skipping visualization.")
    
    print("\n====== Demo completed ======")
    print(f"Logs and data saved to './demo_logs' directory.")


if __name__ == "__main__":
    run_demo() 