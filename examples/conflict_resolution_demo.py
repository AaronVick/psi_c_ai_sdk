#!/usr/bin/env python3
"""
Multi-Agent Conflict Resolution Demo

This demo showcases the conflict resolution system for multi-agent coordination in the ΨC-AI SDK.
It demonstrates how the system detects and resolves conflicts between agents with contradictory beliefs,
using various resolution strategies like trust-based resolution, evidence-based resolution, 
and conservative resolution.
"""

import logging
import random
import argparse
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any

from psi_c_ai_sdk.schema.schema import SchemaGraph, NodeType
from psi_c_ai_sdk.multi_agent.conflict_resolution import (
    ConflictDetector, ConflictResolver, MultiAgentCoordinator,
    ConflictType, ResolutionStrategy
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_agent_schema(agent_id: str, domain: str) -> SchemaGraph:
    """
    Create a schema graph for an agent with domain-specific beliefs.
    
    Args:
        agent_id: ID of the agent
        domain: Domain expertise of the agent
        
    Returns:
        SchemaGraph populated with beliefs and concepts
    """
    schema = SchemaGraph()
    
    # Add domain as a root concept
    schema.add_node(
        f"{domain}_concept",
        label=domain,
        node_type=NodeType.CONCEPT,
        importance=0.9,
        metadata={"domain": domain}
    )
    
    # Maps of contradictory beliefs for different domains
    physics_contradictions = [
        ("Light is a wave", "Light is a particle"),
        ("Quantum mechanics is deterministic", "Quantum mechanics is probabilistic"),
        ("The universe is expanding", "The universe is in steady state"),
        ("Dark matter exists", "Dark matter is a mathematical artifact"),
    ]
    
    biology_contradictions = [
        ("Inheritance is purely genetic", "Epigenetic factors affect inheritance"),
        ("Consciousness arises from the brain", "Consciousness is non-physical"),
        ("Evolution occurs gradually", "Evolution occurs in rapid bursts"),
        ("Viruses are living organisms", "Viruses are not living organisms"),
    ]
    
    psychology_contradictions = [
        ("Nature determines personality", "Nurture determines personality"),
        ("Intelligence is fixed", "Intelligence is malleable"),
        ("Memory is reliable", "Memory is reconstructive"),
        ("Emotions are universal", "Emotions are culturally constructed"),
    ]
    
    # Select contradictions based on domain
    if domain == "physics":
        contradictions = physics_contradictions
    elif domain == "biology":
        contradictions = biology_contradictions
    elif domain == "psychology":
        contradictions = psychology_contradictions
    else:
        # Mix of all if no specific domain
        contradictions = (
            random.sample(physics_contradictions, 1) +
            random.sample(biology_contradictions, 2) +
            random.sample(psychology_contradictions, 1)
        )
    
    # Add beliefs with some contradictions
    for i, (belief1, belief2) in enumerate(contradictions):
        # Add the first belief to this agent with high confidence
        belief_id = f"{agent_id}_belief_{i}"
        schema.add_node(
            belief_id,
            label=belief1 if random.random() < 0.7 else belief2,  # Randomly choose one side of contradiction
            node_type=NodeType.BELIEF,
            content=belief1 if random.random() < 0.7 else belief2,
            importance=random.uniform(0.5, 0.9),
            metadata={
                "confidence": random.uniform(0.7, 0.95),
                "creation_time": datetime.now().isoformat(),
                "source": f"expert_{random.randint(1, 5)}"
            }
        )
        
        # Connect this belief to the domain concept
        schema.add_edge(
            belief_id,
            f"{domain}_concept",
            label="belongs_to",
            weight=0.8
        )
        
        # Add related concepts for each belief
        concept_id = f"{agent_id}_concept_{i}"
        concept_name = belief1.split(' ')[0] if random.random() < 0.5 else belief2.split(' ')[0]
        schema.add_node(
            concept_id,
            label=concept_name,
            node_type=NodeType.CONCEPT,
            importance=random.uniform(0.4, 0.8),
            metadata={"domain": domain}
        )
        
        # Connect belief to concept
        schema.add_edge(
            belief_id,
            concept_id,
            label="relates_to",
            weight=0.7
        )
        
        # Connect concept to domain
        schema.add_edge(
            concept_id,
            f"{domain}_concept",
            label="subclass_of",
            weight=0.9
        )
    
    return schema


def visualize_conflict_resolution(
    agent_schemas: Dict[str, SchemaGraph],
    conflicts: List[Any],
    output_dir: Path
) -> None:
    """
    Visualize the conflicts and resolutions between agents.
    
    Args:
        agent_schemas: Dictionary of agent schemas
        conflicts: List of detected conflicts
        output_dir: Directory to save visualization outputs
    """
    # Create a graph to visualize conflicts
    G = nx.Graph()
    
    # Add agent nodes
    for agent_id in agent_schemas:
        G.add_node(agent_id, node_type="agent")
    
    # Add edges for conflicts
    for conflict in conflicts:
        agent_ids = list(conflict.agent_beliefs.keys())
        if len(agent_ids) >= 2:
            G.add_edge(
                agent_ids[0], 
                agent_ids[1],
                conflict_id=conflict.conflict_id,
                conflict_type=conflict.conflict_type.name,
                resolution=conflict.resolution_strategy.name if conflict.resolution_strategy else "Unresolved"
            )
    
    # Prepare node colors
    node_colors = ['skyblue' for _ in range(len(G.nodes()))]
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(G, pos)
    
    # Color edges by resolution strategy
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if "resolution" not in data:
            edge_colors.append('gray')
        elif data["resolution"] == "TRUST_BASED":
            edge_colors.append('green')
        elif data["resolution"] == "EVIDENCE_BASED":
            edge_colors.append('blue')
        elif data["resolution"] == "CONSERVATIVE":
            edge_colors.append('orange')
        else:
            edge_colors.append('red')
    
    nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors)
    
    # Add edge labels showing conflict type
    edge_labels = {(u, v): data.get("conflict_type", "Unknown") 
                  for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Agent Conflicts and Resolutions")
    plt.axis('off')
    
    # Save the visualization
    output_file = output_dir / "conflict_resolution.png"
    plt.savefig(output_file)
    logger.info(f"Saved conflict visualization to {output_file}")
    plt.close()
    
    # Create a bar chart of conflict types
    conflict_types = {}
    for conflict in conflicts:
        conflict_type = conflict.conflict_type.name
        if conflict_type not in conflict_types:
            conflict_types[conflict_type] = 0
        conflict_types[conflict_type] += 1
    
    plt.figure(figsize=(10, 6))
    plt.bar(conflict_types.keys(), conflict_types.values(), color='skyblue')
    plt.title("Conflict Types")
    plt.xlabel("Type")
    plt.ylabel("Count")
    
    output_file = output_dir / "conflict_types.png"
    plt.savefig(output_file)
    logger.info(f"Saved conflict types chart to {output_file}")
    plt.close()
    
    # Create a pie chart of resolution strategies
    resolution_strategies = {}
    for conflict in conflicts:
        if not conflict.resolution_strategy:
            continue
        strategy = conflict.resolution_strategy.name
        if strategy not in resolution_strategies:
            resolution_strategies[strategy] = 0
        resolution_strategies[strategy] += 1
    
    if resolution_strategies:
        plt.figure(figsize=(8, 8))
        plt.pie(
            resolution_strategies.values(), 
            labels=resolution_strategies.keys(),
            autopct='%1.1f%%', 
            startangle=90,
            colors=['green', 'blue', 'orange', 'red', 'purple', 'gray']
        )
        plt.title("Resolution Strategies")
        
        output_file = output_dir / "resolution_strategies.png"
        plt.savefig(output_file)
        logger.info(f"Saved resolution strategies chart to {output_file}")
        plt.close()


def export_resolution_report(
    conflicts: List[Any], 
    output_dir: Path
) -> None:
    """
    Export a detailed JSON report of all conflicts and their resolutions.
    
    Args:
        conflicts: List of resolved conflicts
        output_dir: Directory to save the report
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_conflicts": len(conflicts),
        "conflicts": [conflict.to_dict() for conflict in conflicts]
    }
    
    output_file = output_dir / "conflict_resolution_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved detailed conflict report to {output_file}")


def run_demo(output_dir: Path, num_agents: int = 5) -> None:
    """
    Run the multi-agent conflict resolution demo.
    
    Args:
        output_dir: Directory to save outputs
        num_agents: Number of agents to simulate
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create agents with different domain expertise
    domains = ["physics", "biology", "psychology", "mixed"]
    agent_schemas = {}
    
    logger.info(f"Creating {num_agents} agents with different domain expertise...")
    for i in range(num_agents):
        agent_id = f"agent_{i}"
        domain = domains[i % len(domains)]
        agent_schemas[agent_id] = create_agent_schema(agent_id, domain)
        logger.info(f"Created {agent_id} with {domain} expertise")
    
    # Create trust scores for agents (in a real system, this would be based on past performance)
    trust_scores = {
        agent_id: random.uniform(0.5, 0.95) for agent_id in agent_schemas
    }
    
    logger.info("Initial trust scores:")
    for agent_id, score in trust_scores.items():
        logger.info(f"  {agent_id}: {score:.2f}")
    
    # Create the conflict detector and resolver
    conflict_detector = ConflictDetector(agent_schemas)
    conflict_resolver = ConflictResolver(
        agent_schemas, 
        agent_trust_scores=trust_scores
    )
    
    # Create the multi-agent coordinator
    coordinator = MultiAgentCoordinator(agent_schemas)
    
    # Run sequential conflict detection and resolution
    logger.info("Detecting conflicts between agents...")
    conflicts = conflict_detector.find_direct_contradictions()
    logger.info(f"Found {len(conflicts)} direct contradictions")
    
    # Resolve each conflict with different strategies
    resolution_strategies = [
        ResolutionStrategy.TRUST_BASED,
        ResolutionStrategy.EVIDENCE_BASED,
        ResolutionStrategy.CONSERVATIVE,
        ResolutionStrategy.MAJORITY_CONSENSUS
    ]
    
    resolved_conflicts = []
    for i, conflict in enumerate(conflicts):
        strategy = resolution_strategies[i % len(resolution_strategies)]
        logger.info(f"Resolving conflict {i+1} with {strategy.name} strategy...")
        
        resolved = conflict_resolver.resolve_conflict(conflict, strategy)
        conflict_resolver.apply_resolution(resolved)
        resolved_conflicts.append(resolved)
        
        # Log the resolution details
        if resolved.resolution:
            resolution_type = resolved.resolution.get("resolution_type")
            if resolution_type == "trust_based":
                winner = resolved.resolution.get("winner")
                logger.info(f"  Resolution: Trust-based, winner: {winner}")
            elif resolution_type == "conservative":
                logger.info(f"  Resolution: Conservative, keeping all beliefs with warnings")
            else:
                logger.info(f"  Resolution: {resolution_type}")
    
    # Run the coordinator to detect and resolve all conflicts at once
    logger.info("\nRunning the multi-agent coordinator to detect and resolve all conflicts...")
    coordinator_resolved = coordinator.detect_and_resolve_conflicts()
    logger.info(f"Coordinator resolved {len(coordinator_resolved)} conflicts")
    
    # Synchronize knowledge across agents
    logger.info("\nSynchronizing shared knowledge across agents...")
    sync_results = coordinator.synchronize_knowledge()
    logger.info(f"Synchronized {sync_results['shared_concepts_count']} shared concepts")
    for agent_id, updates in sync_results['updates_per_agent'].items():
        logger.info(f"  {agent_id} received {updates} new concepts")
    
    # Find collaborative opportunities
    logger.info("\nIdentifying collaboration opportunities...")
    opportunities = coordinator.find_collaborative_opportunities()
    logger.info(f"Found {len(opportunities)} collaboration opportunities")
    for opp in opportunities:
        logger.info(f"  Concept: {opp['concept']}, Experts: {', '.join(opp['experts'])}")
    
    # Combine all conflicts for visualization and reporting
    all_conflicts = resolved_conflicts + [c for c in coordinator_resolved if c.conflict_id not in [rc.conflict_id for rc in resolved_conflicts]]
    
    # Visualize the results
    logger.info("\nVisualizing conflict resolution results...")
    visualize_conflict_resolution(agent_schemas, all_conflicts, output_dir)
    
    # Export detailed report
    export_resolution_report(all_conflicts, output_dir)
    
    logger.info(f"\nDemo complete! Results saved to {output_dir}")


def main():
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(description='Multi-Agent Conflict Resolution Demo')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./conflict_resolution_output',
        help='Directory to save output files'
    )
    parser.add_argument(
        '--num-agents',
        type=int,
        default=5,
        help='Number of agents to simulate'
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    run_demo(output_dir, args.num_agents)


if __name__ == "__main__":
    main() 