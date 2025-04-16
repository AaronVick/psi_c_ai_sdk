#!/usr/bin/env python3
"""
Multi-Agent Schema Visualization Demo for ΨC-AI SDK

This script demonstrates the multi-agent visualization capabilities in the ΨC-AI SDK,
including shared schema visualization, belief contradiction analysis, and conceptual
alignment metrics.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from psi_c_ai_sdk.memory.memory import Memory
from psi_c_ai_sdk.schema.schema import SchemaGraph, NodeType
from psi_c_ai_sdk.multi_agent.shared_schema import SharedSchemaBuilder
from psi_c_ai_sdk.multi_agent.shared_visualization import (
    SchemaOverlapVisualizer, 
    CollectiveSchemaVisualizer
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_memory(content: str, importance: float = 0.8) -> Memory:
    """Create a test memory with the given content."""
    return Memory(
        content=content,
        importance=importance,
        creation_time=datetime.now(),
        metadata={"source": "test"}
    )


def create_test_schema_graph(agent_id: str, scenario: str = "default") -> SchemaGraph:
    """
    Create a test schema graph for demonstrations.
    
    Args:
        agent_id: ID of the agent
        scenario: Scenario type to create
        
    Returns:
        A populated schema graph
    """
    schema = SchemaGraph(agent_id=agent_id)
    
    if scenario == "physics":
        # Physics agent with focus on physics concepts
        schema.add_node(
            "gravity", 
            label="Gravity", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "physics"}
        )
        schema.add_node(
            "mass", 
            label="Mass", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "physics"}
        )
        schema.add_node(
            "energy", 
            label="Energy", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "physics"}
        )
        schema.add_node(
            "force", 
            label="Force", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "physics"}
        )
        schema.add_node(
            "physics_laws", 
            label="Laws of Physics", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "physics"}
        )
        
        # Add a few beliefs
        schema.add_node(
            "gravity_belief", 
            label="Gravity Always Attracts", 
            node_type=NodeType.BELIEF,
            content="Objects with mass always attract each other due to gravity",
            metadata={"confidence": 0.9}
        )
        schema.add_node(
            "energy_belief", 
            label="Energy Conservation", 
            node_type=NodeType.BELIEF,
            content="Energy cannot be created or destroyed, only transformed",
            metadata={"confidence": 0.95}
        )
        
        # Add relationships
        schema.add_edge("gravity", "mass", label="depends on")
        schema.add_edge("force", "mass", label="proportional to")
        schema.add_edge("energy", "force", label="related to")
        schema.add_edge("physics_laws", "gravity", label="includes")
        schema.add_edge("physics_laws", "energy", label="includes")
        schema.add_edge("gravity_belief", "gravity", label="about")
        schema.add_edge("energy_belief", "energy", label="about")
        
    elif scenario == "biology":
        # Biology agent with focus on biology concepts
        schema.add_node(
            "cell", 
            label="Cell", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "biology"}
        )
        schema.add_node(
            "dna", 
            label="DNA", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "biology"}
        )
        schema.add_node(
            "evolution", 
            label="Evolution", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "biology"}
        )
        schema.add_node(
            "organism", 
            label="Organism", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "biology"}
        )
        schema.add_node(
            "energy", 
            label="Energy", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "biology"}
        )
        
        # Add a few beliefs
        schema.add_node(
            "cell_belief", 
            label="Cells are Fundamental", 
            node_type=NodeType.BELIEF,
            content="Cells are the fundamental units of all living organisms",
            metadata={"confidence": 0.95}
        )
        schema.add_node(
            "evolution_belief", 
            label="Evolution is Gradual", 
            node_type=NodeType.BELIEF,
            content="Evolution occurs through gradual changes over many generations",
            metadata={"confidence": 0.8}
        )
        
        # Add relationships
        schema.add_edge("cell", "dna", label="contains")
        schema.add_edge("organism", "cell", label="composed of")
        schema.add_edge("evolution", "organism", label="affects")
        schema.add_edge("cell", "energy", label="requires")
        schema.add_edge("cell_belief", "cell", label="about")
        schema.add_edge("evolution_belief", "evolution", label="about")
        
    elif scenario == "chemistry":
        # Chemistry agent with focus on chemistry concepts
        schema.add_node(
            "atom", 
            label="Atom", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "chemistry"}
        )
        schema.add_node(
            "molecule", 
            label="Molecule", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "chemistry"}
        )
        schema.add_node(
            "reaction", 
            label="Chemical Reaction", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "chemistry"}
        )
        schema.add_node(
            "energy", 
            label="Energy", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "chemistry"}
        )
        schema.add_node(
            "periodic_table", 
            label="Periodic Table", 
            node_type=NodeType.CONCEPT,
            metadata={"domain": "chemistry"}
        )
        
        # Add a few beliefs
        schema.add_node(
            "atom_belief", 
            label="Atoms are Building Blocks", 
            node_type=NodeType.BELIEF,
            content="Atoms are the basic building blocks of all matter",
            metadata={"confidence": 0.9}
        )
        schema.add_node(
            "energy_belief", 
            label="Energy in Reactions", 
            node_type=NodeType.BELIEF,
            content="Chemical reactions involve the release or absorption of energy",
            metadata={"confidence": 0.85}
        )
        
        # Add relationships
        schema.add_edge("molecule", "atom", label="composed of")
        schema.add_edge("reaction", "molecule", label="involves")
        schema.add_edge("reaction", "energy", label="releases/absorbs")
        schema.add_edge("periodic_table", "atom", label="classifies")
        schema.add_edge("atom_belief", "atom", label="about")
        schema.add_edge("energy_belief", "energy", label="about")
        
    else:  # default scenario
        # Default agent with general concepts
        schema.add_node(
            "knowledge", 
            label="Knowledge", 
            node_type=NodeType.CONCEPT
        )
        schema.add_node(
            "science", 
            label="Science", 
            node_type=NodeType.CONCEPT
        )
        schema.add_node(
            "nature", 
            label="Nature", 
            node_type=NodeType.CONCEPT
        )
        schema.add_node(
            "energy", 
            label="Energy", 
            node_type=NodeType.CONCEPT
        )
        
        # Add a few beliefs
        schema.add_node(
            "science_belief", 
            label="Science is Methodical", 
            node_type=NodeType.BELIEF,
            content="Science is a methodical approach to understanding the natural world",
            metadata={"confidence": 0.9}
        )
        
        # Add relationships
        schema.add_edge("science", "knowledge", label="produces")
        schema.add_edge("science", "nature", label="studies")
        schema.add_edge("nature", "energy", label="contains")
        schema.add_edge("science_belief", "science", label="about")
    
    logger.info(f"Created schema for agent '{agent_id}' with {len(schema.graph.nodes)} nodes")
    return schema


def create_agent_schemas() -> Dict[str, SchemaGraph]:
    """
    Create a set of agent schemas for demonstration.
    
    Returns:
        Dictionary mapping agent IDs to schema graphs
    """
    agent_schemas = {
        "PhysicsAgent": create_test_schema_graph("PhysicsAgent", "physics"),
        "BiologyAgent": create_test_schema_graph("BiologyAgent", "biology"),
        "ChemistryAgent": create_test_schema_graph("ChemistryAgent", "chemistry"),
        "GeneralAgent": create_test_schema_graph("GeneralAgent", "default")
    }
    
    return agent_schemas


def visualize_overlap_network(agent_schemas: Dict[str, SchemaGraph], output_dir: str) -> None:
    """
    Visualize the overlap network between agent schemas.
    
    Args:
        agent_schemas: Dictionary mapping agent IDs to schema graphs
        output_dir: Directory to save visualizations
    """
    # Create shared builder
    shared_builder = SharedSchemaBuilder()
    
    # Add agent schemas
    for agent_id, schema in agent_schemas.items():
        shared_builder.add_agent_schema(agent_id, schema)
    
    # Create visualizer
    visualizer = SchemaOverlapVisualizer(shared_builder)
    
    # Generate visualization
    output_file = os.path.join(output_dir, "overlap_network.png")
    visualizer.visualize_overlap_network(
        title="Multi-Agent Knowledge Overlap",
        output_file=output_file
    )
    
    logger.info(f"Saved overlap network visualization to {output_file}")


def visualize_conceptual_alignment(agent_schemas: Dict[str, SchemaGraph], output_dir: str) -> None:
    """
    Visualize the conceptual alignment between agent schemas.
    
    Args:
        agent_schemas: Dictionary mapping agent IDs to schema graphs
        output_dir: Directory to save visualizations
    """
    # Create shared builder
    shared_builder = SharedSchemaBuilder()
    
    # Add agent schemas
    for agent_id, schema in agent_schemas.items():
        shared_builder.add_agent_schema(agent_id, schema)
    
    # Create visualizer
    visualizer = SchemaOverlapVisualizer(shared_builder)
    
    # Generate visualization
    output_file = os.path.join(output_dir, "conceptual_alignment.png")
    visualizer.visualize_conceptual_alignment(
        output_file=output_file
    )
    
    logger.info(f"Saved conceptual alignment visualization to {output_file}")


def visualize_belief_contradictions(agent_schemas: Dict[str, SchemaGraph], output_dir: str) -> None:
    """
    Visualize contradictions in beliefs between agents.
    
    Args:
        agent_schemas: Dictionary mapping agent IDs to schema graphs
        output_dir: Directory to save visualizations
    """
    # Create shared builder
    shared_builder = SharedSchemaBuilder()
    
    # Add agent schemas
    for agent_id, schema in agent_schemas.items():
        shared_builder.add_agent_schema(agent_id, schema)
    
    # Create visualizer
    visualizer = SchemaOverlapVisualizer(shared_builder)
    
    # Generate visualization
    output_file = os.path.join(output_dir, "belief_contradictions.png")
    visualizer.visualize_belief_contradictions(
        output_file=output_file
    )
    
    logger.info(f"Saved belief contradictions visualization to {output_file}")


def visualize_collective_schema(agent_schemas: Dict[str, SchemaGraph], output_dir: str) -> None:
    """
    Visualize the collective schema of all agents.
    
    Args:
        agent_schemas: Dictionary mapping agent IDs to schema graphs
        output_dir: Directory to save visualizations
    """
    # Create collective visualizer
    visualizer = CollectiveSchemaVisualizer(agent_schemas)
    
    # Generate visualization
    output_file = os.path.join(output_dir, "collective_schema.png")
    visualizer.visualize_collective_schema(
        output_file=output_file,
        highlight_overlaps=True
    )
    
    logger.info(f"Saved collective schema visualization to {output_file}")


def main():
    """Main entry point for the demo script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multi-Agent Schema Visualization Demo")
    parser.add_argument(
        "--output-dir", 
        default="multi_agent_visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--save-schemas", 
        action="store_true",
        help="Save agent schemas to files"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Create agent schemas
    agent_schemas = create_agent_schemas()
    
    # Save schemas if requested
    if args.save_schemas:
        for agent_id, schema in agent_schemas.items():
            schema_file = os.path.join(output_dir, f"{agent_id}_schema.json")
            schema.to_json_file(schema_file)
            logger.info(f"Saved schema for agent '{agent_id}' to {schema_file}")
    
    # Generate visualizations
    logger.info("Generating overlap network visualization...")
    visualize_overlap_network(agent_schemas, output_dir)
    
    logger.info("Generating conceptual alignment visualization...")
    visualize_conceptual_alignment(agent_schemas, output_dir)
    
    logger.info("Generating belief contradictions visualization...")
    visualize_belief_contradictions(agent_schemas, output_dir)
    
    logger.info("Generating collective schema visualization...")
    visualize_collective_schema(agent_schemas, output_dir)
    
    logger.info("Visualization demo complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 