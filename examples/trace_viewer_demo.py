#!/usr/bin/env python
"""
Trace Viewer Demo

This script demonstrates how to use the ΨC trace viewer to visualize 
introspection data from a running system.
"""

import time
import random
import uuid
from pathlib import Path

# Import the necessary components
from psi_c_ai_sdk.memory import MemoryStore, Memory
from psi_c_ai_sdk.coherence import BasicCoherenceScorer
from psi_c_ai_sdk.psi_c import PsiCOperator, CollapseSimulator, PsiCState
from psi_c_ai_sdk.logging import IntrospectionLogger, EventType, view_trace, export_trace

# Setup basic logging configuration
def setup_logger():
    logger = IntrospectionLogger()
    logger.start_trace(trace_id="demo-trace")
    return logger

# Create some random memories
def generate_memories(count=20):
    memories = []
    topics = ["AI Ethics", "Cognitive Architecture", "Self-Reflection", 
              "Consciousness", "Learning", "Memory Systems", "Decision Making"]
    
    for i in range(count):
        content = f"Insight about {random.choice(topics)}: {uuid.uuid4().hex[:8]}"
        embedding = [random.random() for _ in range(10)]  # Simple random embedding
        memory = Memory(
            id=f"mem_{i}",
            content=content,
            embedding=embedding,
            metadata={
                "importance": random.random(),
                "recency": random.random(),
                "source": random.choice(["user", "system", "reflection"])
            }
        )
        memories.append(memory)
    
    return memories

# Simulate a ΨC system with active memory processing
def run_simulation(duration=30, memory_count=20):
    print("Setting up introspection logger...")
    logger = setup_logger()
    
    print("Creating memory store...")
    memory_store = MemoryStore()
    
    print("Setting up coherence scorer...")
    coherence_scorer = BasicCoherenceScorer()
    
    print("Initializing ΨC operator...")
    psic = PsiCOperator(
        memory_store=memory_store,
        threshold=0.5,
        window_size=5,
        reflection_weight=0.7
    )
    
    print("Setting up collapse simulator...")
    collapse_sim = CollapseSimulator(
        psic_operator=psic,
        collapse_threshold=0.8
    )
    
    print(f"Generating {memory_count} initial memories...")
    memories = generate_memories(memory_count)
    
    # Add initial memories
    for memory in memories:
        memory_store.add(memory)
        logger.log_event(
            event_type=EventType.MEMORY_ADDED,
            description=f"Added memory: {memory.id}",
            data={"memory_id": memory.id, "content": memory.content}
        )
    
    print(f"Running simulation for {duration} seconds...")
    start_time = time.time()
    
    # Log system startup
    logger.log_event(
        event_type=EventType.SYSTEM_STARTUP,
        description="ΨC system initialized",
        data={
            "config": {
                "threshold": psic.threshold,
                "window_size": psic.window_size,
                "reflection_weight": psic.reflection_weight
            }
        }
    )
    
    # Simulation loop
    while time.time() - start_time < duration:
        # Randomly select memories to access
        for _ in range(random.randint(1, 3)):
            if memory_store.memories:
                memory = random.choice(list(memory_store.memories.values()))
                memory_store.access(memory.id)
                logger.log_event(
                    event_type=EventType.MEMORY_ACCESSED,
                    description=f"Accessed memory: {memory.id}",
                    data={"memory_id": memory.id, "content": memory.content}
                )
        
        # Calculate coherence
        coherence_value = coherence_scorer.score_coherence(list(memory_store.memories.values()))
        logger.log_event(
            event_type=EventType.COHERENCE_CALCULATED,
            description=f"Calculated coherence: {coherence_value:.3f}",
            data={"coherence_value": coherence_value}
        )
        
        # Update ΨC activation
        prev_state = psic.state
        activation_score = psic.calculate_activation()
        
        logger.log_event(
            event_type=EventType.PSIC_ACTIVATION,
            description=f"ΨC activation score: {activation_score:.3f}",
            data={"activation_score": activation_score}
        )
        
        # Check for state changes
        if prev_state != psic.state:
            logger.log_event(
                event_type=EventType.PSIC_STATE_CHANGE,
                description=f"ΨC state changed from {prev_state.name} to {psic.state.name}",
                data={
                    "old_state": prev_state.name,
                    "new_state": psic.state.name,
                    "activation_score": activation_score
                }
            )
        
        # Occasionally add a new memory
        if random.random() < 0.3:
            new_memory = generate_memories(1)[0]
            memory_store.add(new_memory)
            logger.log_event(
                event_type=EventType.MEMORY_ADDED,
                description=f"Added new memory: {new_memory.id}",
                data={"memory_id": new_memory.id, "content": new_memory.content}
            )
        
        # Check for collapse events
        if psic.state == PsiCState.UNSTABLE and random.random() < 0.4:
            collapse_decision = collapse_sim.should_collapse()
            logger.log_event(
                event_type=EventType.COLLAPSE_DECISION,
                description=f"Collapse decision: {'Yes' if collapse_decision else 'No'}",
                data={"collapse": collapse_decision}
            )
            
            if collapse_decision:
                # Simulate collapse (e.g., prune some memories)
                pruned_memories = []
                for _ in range(random.randint(1, 3)):
                    if memory_store.memories:
                        mem_id = random.choice(list(memory_store.memories.keys()))
                        memory_store.delete(mem_id)
                        pruned_memories.append(mem_id)
                
                logger.log_event(
                    event_type=EventType.REFLECTION_INSIGHT,
                    description="Identified contradictory memories and pruned them",
                    data={
                        "pruned_memories": pruned_memories,
                        "confidence": random.uniform(0.7, 0.95)
                    }
                )
        
        # Add some random pauses to simulate processing time
        time.sleep(random.uniform(0.2, 0.8))
    
    print("Simulation completed!")
    
    # Stop the trace
    logger.end_trace("demo-trace")
    
    return logger

def main():
    print("Starting ΨC trace viewer demo")
    print("==============================")
    
    # Run the simulation
    logger = run_simulation(duration=15, memory_count=15)
    
    # Export the trace to a file
    output_dir = Path.cwd() / "trace_output"
    output_dir.mkdir(exist_ok=True)
    
    export_path = output_dir / "demo_trace.json"
    export_trace(output_path=export_path, trace_id="demo-trace", logger=logger)
    print(f"Trace exported to: {export_path}")
    
    # View the trace in the browser
    print("Opening trace viewer in your browser...")
    html_path = view_trace(trace_id="demo-trace", logger=logger)
    print(f"Trace viewer HTML saved to: {html_path}")
    
    print("\nDemo completed!")
    print("You can view the trace data in your browser.")
    print("The JSON trace file can be loaded again later using:")
    print("  from psi_c_ai_sdk.logging import TraceViewerWeb")
    print("  viewer = TraceViewerWeb()")
    print(f"  viewer.load_from_file('{export_path}')")
    print("  viewer.view_trace()")

if __name__ == "__main__":
    main() 