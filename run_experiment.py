#!/usr/bin/env python3
"""
ΨC-AI SDK Experiment Runner

This script serves as the main entry point for running ΨC system experiments.
It integrates all five pillars of the ΨC system and demonstrates the feedback loop
of: input → reflect → check contradictions → update memory → log coherence.

Usage:
  python run_experiment.py --scenario=[scenario_name] --cycles=[num_cycles] --log-dir=[log_directory]

Example:
  python run_experiment.py --scenario=contradiction_resolution --cycles=10 --log-dir=./experiment_logs
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.schema import SchemaGraph
from psi_c_ai_sdk.coherence import CoherenceCalculator
from psi_c_ai_sdk.reflection import ReflectionEngine, ReflectionScheduler
from psi_c_ai_sdk.contradiction import ContradictionDetector
from psi_c_ai_sdk.identity import self_entropy
from psi_c_ai_sdk.core.orchestration import CycleController, InputProcessor
from psi_c_ai_sdk.runtime.complexity_controller import ComplexityController, ComplexityTier, FeatureActivation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment.log")
    ]
)
logger = logging.getLogger("experiment_runner")

# Define example scenarios
SCENARIOS = {
    "contradiction_resolution": {
        "description": "Tests the system's ability to resolve contradictions and maintain coherence",
        "inputs": [
            {"content": "The capital of France is Paris.", "importance": 0.8},
            {"content": "Paris is known for the Eiffel Tower.", "importance": 0.7},
            {"content": "The Eiffel Tower was built in 1889.", "importance": 0.6},
            {"content": "The capital of France is Lyon.", "importance": 0.9, "source": "unreliable_source"},
            {"content": "Paris has many museums including the Louvre.", "importance": 0.8},
            {"content": "The Louvre houses the Mona Lisa painting.", "importance": 0.7},
            {"content": "The Eiffel Tower was built in 1999.", "importance": 0.5, "source": "unreliable_source"},
            {"content": "France is in Western Europe.", "importance": 0.9}
        ]
    },
    "identity_evolution": {
        "description": "Tests how the system maintains identity over time despite evolving beliefs",
        "inputs": [
            {"content": "I prefer quiet environments for working.", "importance": 0.7},
            {"content": "I enjoy reading science fiction books.", "importance": 0.8},
            {"content": "I believe in evidence-based decision making.", "importance": 0.9},
            {"content": "I've started to enjoy working with background noise.", "importance": 0.6},
            {"content": "Background noise helps me focus better.", "importance": 0.7},
            {"content": "I find that I'm most productive in coffee shops.", "importance": 0.8},
            {"content": "I still value evidence-based approaches to problems.", "importance": 0.9},
            {"content": "My reading preferences have shifted to non-fiction.", "importance": 0.7},
            {"content": "I now primarily read historical accounts and biographies.", "importance": 0.8}
        ]
    },
    "progressive_learning": {
        "description": "Tests the system's ability to build knowledge incrementally and coherently",
        "inputs": [
            {"content": "Atoms are the basic units of matter.", "importance": 0.9},
            {"content": "Atoms consist of protons, neutrons, and electrons.", "importance": 0.8},
            {"content": "Protons have a positive charge.", "importance": 0.7},
            {"content": "Electrons have a negative charge.", "importance": 0.7},
            {"content": "Neutrons have no charge.", "importance": 0.7},
            {"content": "Protons and neutrons form the nucleus of an atom.", "importance": 0.8},
            {"content": "Electrons orbit the nucleus in shells.", "importance": 0.8},
            {"content": "The atomic number is equal to the number of protons.", "importance": 0.7},
            {"content": "Elements in the same column of the periodic table have similar properties.", "importance": 0.6},
            {"content": "Chemical bonds form when atoms share or transfer electrons.", "importance": 0.8}
        ]
    }
}


def create_experiment_components(args):
    """
    Create all the necessary components for running an experiment.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with all created components
    """
    # Create core components
    memory_store = MemoryStore()
    
    # Build schema graph
    schema_graph = SchemaGraph(memory_store)
    
    # Create reflection components
    reflection_scheduler = ReflectionScheduler(
        coherence_threshold=0.6,
        min_interval=1.0  # Low interval for experiment purposes
    )
    
    reflection_engine = ReflectionEngine(
        memory_store=memory_store,
        scheduler=reflection_scheduler
    )
    
    # Create contradiction components
    contradiction_detector = ContradictionDetector(
        memory_store=memory_store,
        schema_graph=schema_graph
    )
    
    # Create coherence components
    coherence_calculator = CoherenceCalculator(
        memory_store=memory_store,
        schema_graph=schema_graph
    )
    
    # Create complexity controller
    complexity_controller = ComplexityController(
        memory_store=memory_store,
        schema_graph=schema_graph,
        reflection_engine=reflection_engine,
        tier_thresholds={
            ComplexityTier.TIER_0: 0.0,    # Always active
            ComplexityTier.TIER_1: 10.0,   # Low threshold for experiments
            ComplexityTier.TIER_2: 20.0,
            ComplexityTier.TIER_3: 30.0
        },
        min_interval=1.0  # Low interval for experiment purposes
    )
    
    # Create input processor
    input_processor = InputProcessor(
        memory_store=memory_store,
        source_trust_levels={
            "reliable_source": 0.9,
            "unreliable_source": 0.3
        }
    )
    
    # Create main cycle controller
    cycle_controller = CycleController(
        memory_store=memory_store,
        schema_graph=schema_graph,
        reflection_engine=reflection_engine,
        contradiction_detector=contradiction_detector,
        coherence_calculator=coherence_calculator,
        input_processor=input_processor,
        complexity_controller=complexity_controller,
        cycle_frequency=None,  # Don't enforce frequency for experiments
        log_dir=args.log_dir
    )
    
    return {
        "memory_store": memory_store,
        "schema_graph": schema_graph,
        "reflection_engine": reflection_engine,
        "contradiction_detector": contradiction_detector,
        "coherence_calculator": coherence_calculator,
        "complexity_controller": complexity_controller,
        "input_processor": input_processor,
        "cycle_controller": cycle_controller
    }


def run_scenario(scenario_name, components, args):
    """
    Run a specific scenario experiment.
    
    Args:
        scenario_name: Name of the scenario to run
        components: Dictionary of system components
        args: Command line arguments
        
    Returns:
        Experiment results
    """
    logger.info(f"Running scenario: {scenario_name}")
    
    # Extract components
    cycle_controller = components["cycle_controller"]
    
    # Get scenario data
    if scenario_name not in SCENARIOS:
        logger.error(f"Unknown scenario: {scenario_name}")
        return None
        
    scenario = SCENARIOS[scenario_name]
    logger.info(f"Scenario description: {scenario['description']}")
    
    # Process inputs
    inputs = scenario["inputs"]
    logger.info(f"Processing {len(inputs)} inputs...")
    
    memory_ids = []
    for i, input_data in enumerate(inputs):
        logger.info(f"Ingesting input {i+1}/{len(inputs)}: {input_data['content'][:50]}...")
        
        # Extract metadata
        metadata = {k: v for k, v in input_data.items() if k not in ["content", "importance", "source"]}
        
        # Ingest the input
        memory_id = cycle_controller.ingest(
            content=input_data["content"],
            metadata=metadata,
            source=input_data.get("source"),
            importance=input_data.get("importance")
        )
        
        memory_ids.append(memory_id)
        
        # Run a cycle after each input if requested
        if args.cycle_per_input:
            logger.info(f"Running cycle for input {i+1}...")
            cycle_controller.run_cycle(memory_id=memory_id)
    
    # Run requested number of cycles
    if not args.cycle_per_input:
        logger.info(f"Running {args.cycles} cycles...")
        cycle_results = cycle_controller.run_continuous(max_cycles=args.cycles)
        logger.info(f"Completed {len(cycle_results)} cycles")
    
    # Gather and return results
    coherence_history = cycle_controller.get_coherence_history()
    cycle_stats = cycle_controller.get_cycle_stats()
    
    results = {
        "scenario": scenario_name,
        "description": scenario["description"],
        "input_count": len(inputs),
        "cycles_run": cycle_stats["cycles_run"],
        "coherence_history": [(t, v) for t, v in coherence_history],
        "initial_coherence": cycle_stats["initial_coherence"],
        "final_coherence": cycle_stats["latest_coherence"],
        "coherence_change": cycle_stats["latest_coherence"] - cycle_stats["initial_coherence"],
        "coherence_trend": cycle_stats["coherence_trend"],
        "avg_duration": cycle_stats["avg_duration"]
    }
    
    logger.info(f"Scenario results: Initial coherence: {results['initial_coherence']:.4f}, "
                f"Final coherence: {results['final_coherence']:.4f}, "
                f"Change: {results['coherence_change']:.4f}")
    
    return results


def save_results(results, args):
    """
    Save experiment results to file.
    
    Args:
        results: Experiment results to save
        args: Command line arguments
    """
    if not args.log_dir:
        logger.info("No log directory specified, results not saved.")
        return
        
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save results to JSON file
    results_file = os.path.join(args.log_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Results saved to {results_file}")
    
    # Create a simple visualization if bokeh is available
    try:
        import bokeh.plotting as plt
        from bokeh.models import ColumnDataSource
        from bokeh.layouts import column
        from bokeh.embed import file_html
        from bokeh.resources import CDN
        
        # Create coherence plot
        p = plt.figure(title="Coherence Over Time", 
                      x_axis_label="Cycle",
                      y_axis_label="Coherence Score",
                      width=800, height=400)
        
        # Extract cycle numbers and coherence values
        cycles = list(range(len(results["coherence_history"])))
        coherence = [v for _, v in results["coherence_history"]]
        
        source = ColumnDataSource(data=dict(cycles=cycles, coherence=coherence))
        
        p.line('cycles', 'coherence', source=source, line_width=2)
        p.circle('cycles', 'coherence', source=source, size=8, color="red")
        
        # Save visualization
        viz_file = os.path.join(args.log_dir, f"coherence_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        with open(viz_file, 'w') as f:
            f.write(file_html(p, CDN, "ΨC Coherence Plot"))
            
        logger.info(f"Visualization saved to {viz_file}")
        
    except ImportError:
        logger.info("Bokeh not available, skipping visualization.")


def main():
    """Main function to parse arguments and run the experiment."""
    parser = argparse.ArgumentParser(description="ΨC-AI SDK Experiment Runner")
    
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="contradiction_resolution",
        help="Experiment scenario to run"
    )
    
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of ΨC cycles to run"
    )
    
    parser.add_argument(
        "--log-dir",
        default="./experiment_logs",
        help="Directory to store experiment logs and results"
    )
    
    parser.add_argument(
        "--cycle-per-input",
        action="store_true",
        help="Run a cycle after each input instead of waiting"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting ΨC experiment runner")
    logger.info(f"Scenario: {args.scenario}, Cycles: {args.cycles}, Log directory: {args.log_dir}")
    
    # Create experiment components
    logger.info("Creating experiment components...")
    components = create_experiment_components(args)
    
    # Run the experiment
    logger.info("Running experiment...")
    start_time = time.time()
    results = run_scenario(args.scenario, components, args)
    duration = time.time() - start_time
    
    logger.info(f"Experiment completed in {duration:.2f} seconds")
    
    # Save results
    save_results(results, args)
    
    logger.info("Experiment run completed")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 