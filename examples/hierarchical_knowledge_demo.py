#!/usr/bin/env python3
"""
Hierarchical Knowledge Management Demo.

This script demonstrates the use of the HierarchicalKnowledgeManager to create,
manage, and analyze concept hierarchies, taxonomies, and ontological relationships.
"""

import os
import sys
import time
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

# Ensure the parent directory is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from psi_c_ai_sdk.schema import SchemaGraph
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.belief.justification_chain import JustificationChain, Belief, JustificationLink
from psi_c_ai_sdk.knowledge.hierarchical_manager import HierarchicalKnowledgeManager

# Set up console for nice output
console = Console()


def initialize_components():
    """Initialize the necessary components for the demo."""
    console.print("[bold blue]Initializing components...[/bold blue]")
    
    # Create schema graph
    schema_graph = SchemaGraph()
    
    # Create memory store
    memory_store = MemoryStore()
    
    # Create justification chain
    justification_chain = JustificationChain(schema_graph=schema_graph, memory_store=memory_store)
    
    # Create hierarchical knowledge manager
    manager = HierarchicalKnowledgeManager(
        schema_graph=schema_graph,
        memory_store=memory_store,
        justification_chain=justification_chain
    )
    
    console.print("[bold green]Components initialized successfully![/bold green]")
    return schema_graph, memory_store, justification_chain, manager


def create_knowledge_structure(manager):
    """Create a knowledge structure for the demo."""
    console.print("\n[bold blue]Creating knowledge structure...[/bold blue]")
    
    # Create top-level domains
    domains = [
        "Mathematics",
        "Physics",
        "Biology",
        "Computer Science"
    ]
    
    # Create branches for each domain
    branches = {}
    
    # Mathematics
    branches["math"] = manager.create_taxonomic_branch(
        parent_concept="Mathematics",
        child_concepts=[
            "Algebra",
            "Geometry",
            "Calculus",
            "Number Theory",
            "Statistics"
        ],
        branch_name="mathematics_branch"
    )
    
    # Add subconcepts to algebra
    algebra_subconcepts = manager.create_taxonomic_branch(
        parent_concept="Algebra",
        child_concepts=[
            "Linear Algebra",
            "Abstract Algebra",
            "Elementary Algebra"
        ],
        branch_name="algebra_branch"
    )
    
    # Physics
    branches["physics"] = manager.create_taxonomic_branch(
        parent_concept="Physics",
        child_concepts=[
            "Classical Mechanics",
            "Quantum Mechanics",
            "Thermodynamics",
            "Electromagnetism",
            "Relativity"
        ],
        branch_name="physics_branch"
    )
    
    # Biology
    branches["biology"] = manager.create_taxonomic_branch(
        parent_concept="Biology",
        child_concepts=[
            "Genetics",
            "Ecology",
            "Molecular Biology",
            "Evolutionary Biology",
            "Cell Biology"
        ],
        branch_name="biology_branch"
    )
    
    # Computer Science
    branches["cs"] = manager.create_taxonomic_branch(
        parent_concept="Computer Science",
        child_concepts=[
            "Algorithms",
            "Data Structures",
            "Artificial Intelligence",
            "Operating Systems",
            "Computer Networks"
        ],
        branch_name="cs_branch"
    )
    
    # Add subconcepts to artificial intelligence
    ai_subconcepts = manager.create_taxonomic_branch(
        parent_concept="Artificial Intelligence",
        child_concepts=[
            "Machine Learning",
            "Natural Language Processing",
            "Computer Vision",
            "Robotics",
            "Knowledge Representation"
        ],
        branch_name="ai_branch"
    )
    
    # Add subconcepts to machine learning
    ml_subconcepts = manager.create_taxonomic_branch(
        parent_concept="Machine Learning",
        child_concepts=[
            "Supervised Learning",
            "Unsupervised Learning",
            "Reinforcement Learning",
            "Deep Learning",
            "Transfer Learning"
        ],
        branch_name="ml_branch"
    )
    
    console.print("[bold green]Knowledge structure created successfully![/bold green]")
    return branches


def add_beliefs_and_justifications(manager, justification_chain):
    """Add beliefs and justifications to the knowledge structure."""
    console.print("\n[bold blue]Adding beliefs and justifications...[/bold blue]")
    
    # Get concept IDs
    hierarchy = manager.get_concept_hierarchy()
    concept_map = {}
    for concept in hierarchy["concepts"]:
        concept_map[concept["label"]] = concept["id"]
    
    # Create beliefs about concepts
    beliefs = {}
    
    # Mathematics belief
    beliefs["math"] = Belief(
        content="Mathematics is the study of numbers, quantities, and spaces.",
        confidence=0.95,
        creation_time=datetime.now(),
        category="definition",
        tags=["mathematics", "definition"]
    )
    justification_chain.add_belief(beliefs["math"])
    
    # Add justification link to mathematics concept
    if "Mathematics" in concept_map:
        justification_chain.add_justification(
            beliefs["math"].id,
            source_type="concept",
            source_id=concept_map["Mathematics"],
            relation="defined_by",
            weight=0.9
        )
    
    # Machine learning belief
    beliefs["ml"] = Belief(
        content="Machine learning is a subfield of AI that focuses on algorithms that can learn from data.",
        confidence=0.9,
        creation_time=datetime.now(),
        category="definition",
        tags=["machine learning", "AI", "definition"]
    )
    justification_chain.add_belief(beliefs["ml"])
    
    # Add justification links
    if "Machine Learning" in concept_map:
        justification_chain.add_justification(
            beliefs["ml"].id,
            source_type="concept",
            source_id=concept_map["Machine Learning"],
            relation="defined_by",
            weight=0.9
        )
    
    if "Artificial Intelligence" in concept_map:
        justification_chain.add_justification(
            beliefs["ml"].id,
            source_type="concept",
            source_id=concept_map["Artificial Intelligence"],
            relation="part_of",
            weight=0.8
        )
    
    # Deep learning belief
    beliefs["dl"] = Belief(
        content="Deep learning uses neural networks with multiple layers to learn from data.",
        confidence=0.85,
        creation_time=datetime.now(),
        category="definition",
        tags=["deep learning", "neural networks", "definition"]
    )
    justification_chain.add_belief(beliefs["dl"])
    
    # Add justification links
    if "Deep Learning" in concept_map:
        justification_chain.add_justification(
            beliefs["dl"].id,
            source_type="concept",
            source_id=concept_map["Deep Learning"],
            relation="defined_by",
            weight=0.9
        )
    
    # Deep learning is justified by machine learning belief
    justification_chain.add_justification(
        beliefs["dl"].id,
        source_type="belief",
        source_id=beliefs["ml"].id,
        relation="supported_by",
        weight=0.7
    )
    
    console.print("[bold green]Beliefs and justifications added successfully![/bold green]")
    return beliefs


def analyze_knowledge_structure(manager):
    """Analyze the knowledge structure."""
    console.print("\n[bold blue]Analyzing knowledge structure...[/bold blue]")
    
    # Get hierarchy
    hierarchy = manager.get_concept_hierarchy()
    
    # Print statistics
    console.print(f"Total concepts: [bold]{len(hierarchy['concepts'])}[/bold]")
    
    # Count concepts by level
    levels = {}
    for concept in hierarchy["concepts"]:
        depth = len(concept["parents"])
        if depth not in levels:
            levels[depth] = 0
        levels[depth] += 1
    
    # Create a table for level statistics
    level_table = Table(title="Concept Distribution by Level")
    level_table.add_column("Level", style="cyan")
    level_table.add_column("Count", style="green")
    level_table.add_column("Percentage", style="magenta")
    
    for level, count in sorted(levels.items()):
        percentage = (count / len(hierarchy["concepts"])) * 100
        level_table.add_row(
            str(level),
            str(count),
            f"{percentage:.2f}%"
        )
    
    console.print(level_table)
    
    # Analyze impact of key concepts
    console.print("\n[bold]Impact of Key Concepts:[/bold]")
    
    # Find concept IDs for key concepts
    key_concepts = ["Mathematics", "Computer Science", "Artificial Intelligence", "Machine Learning"]
    concept_map = {concept["label"]: concept["id"] for concept in hierarchy["concepts"]}
    
    # Create impact table
    impact_table = Table(title="Concept Impact Analysis")
    impact_table.add_column("Concept", style="cyan")
    impact_table.add_column("Direct Connections", style="green")
    impact_table.add_column("Hierarchy Influence", style="magenta")
    impact_table.add_column("Overall Impact", style="yellow")
    
    for concept_name in key_concepts:
        if concept_name in concept_map:
            concept_id = concept_map[concept_name]
            impact = manager.get_concept_impact(concept_id)
            
            impact_table.add_row(
                concept_name,
                str(impact["direct_connections"]),
                f"{impact['hierarchy_influence']:.2f}",
                f"{impact['overall_impact']:.2f}"
            )
    
    console.print(impact_table)
    
    # Find common ancestor of machine learning and deep learning
    if "Machine Learning" in concept_map and "Deep Learning" in concept_map:
        ml_id = concept_map["Machine Learning"]
        dl_id = concept_map["Deep Learning"]
        
        common_ancestor_id = manager.find_common_ancestor([ml_id, dl_id])
        common_ancestor_name = ""
        
        for concept in hierarchy["concepts"]:
            if concept["id"] == common_ancestor_id:
                common_ancestor_name = concept["label"]
                break
                
        console.print(f"\nCommon ancestor of Machine Learning and Deep Learning: [bold]{common_ancestor_name}[/bold]")
    
    # Get lineage of AI
    if "Artificial Intelligence" in concept_map:
        ai_id = concept_map["Artificial Intelligence"]
        lineage = manager.get_lineage(ai_id)
        
        console.print("\n[bold]Lineage of Artificial Intelligence:[/bold]")
        console.print(f"Ancestors: {len(lineage['ancestors'])}")
        console.print(f"Descendants: {len(lineage['descendants'])}")
        
        # Create a tree representation
        ai_tree = Tree("Artificial Intelligence")
        
        # Add ancestors
        if lineage["ancestors"]:
            ancestors_branch = ai_tree.add("Ancestors")
            for ancestor in lineage["ancestors"]:
                ancestors_branch.add(f"[blue]{ancestor['label']}[/blue]")
        
        # Add descendants
        if lineage["descendants"]:
            descendants_branch = ai_tree.add("Descendants")
            
            # Group by depth
            descendants_by_depth = {}
            for descendant in lineage["descendants"]:
                depth = descendant["depth"]
                if depth not in descendants_by_depth:
                    descendants_by_depth[depth] = []
                descendants_by_depth[depth].append(descendant)
            
            # Add to tree by depth
            for depth, descendants in sorted(descendants_by_depth.items()):
                depth_branch = descendants_branch.add(f"Depth {depth}")
                for descendant in descendants:
                    depth_branch.add(f"[green]{descendant['label']}[/green]")
        
        console.print(ai_tree)


def visualize_hierarchy(manager, schema_graph):
    """Visualize the concept hierarchy as a graph."""
    console.print("\n[bold blue]Visualizing concept hierarchy...[/bold blue]")
    
    try:
        # Create a visualization graph
        G = nx.DiGraph()
        
        # Add nodes from schema graph
        for node_id, data in schema_graph.graph.nodes(data=True):
            if data.get("node_type") == "concept":
                label = data.get("label", node_id)
                G.add_node(node_id, label=label)
        
        # Add edges for parent-child relationships
        for source, target, data in schema_graph.graph.edges(data=True):
            if data.get("edge_type") == "parent_child":
                G.add_edge(source, target)
        
        # Create the figure
        plt.figure(figsize=(14, 10))
        
        # Use a hierarchical layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue", alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=15)
        
        # Draw labels
        labels = {n: G.nodes[n]["label"] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        # Set title
        plt.title("Concept Hierarchy Visualization")
        plt.axis("off")
        
        # Save the figure
        plt.savefig("concept_hierarchy.png", dpi=300, bbox_inches="tight")
        console.print("[bold green]Visualization saved as 'concept_hierarchy.png'[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error visualizing hierarchy: {e}[/bold red]")


def perform_branch_operations(manager):
    """Demonstrate branch operations like merging."""
    console.print("\n[bold blue]Performing branch operations...[/bold blue]")
    
    # Create new branches for demonstration
    physics_classical = manager.create_taxonomic_branch(
        parent_concept="Classical Mechanics",
        child_concepts=["Newtonian Mechanics", "Lagrangian Mechanics", "Hamiltonian Mechanics"],
        branch_name="classical_mechanics_branch"
    )
    
    physics_modern = manager.create_taxonomic_branch(
        parent_concept="Modern Physics",
        child_concepts=["Relativity", "Quantum Mechanics"],
        branch_name="modern_physics_branch"
    )
    
    # Create alternative classical mechanics branch with different structure
    physics_classical_alt = manager.create_taxonomic_branch(
        parent_concept="Classical Physics",
        child_concepts=["Newtonian Physics", "Waves", "Optics", "Thermodynamics"],
        branch_name="classical_physics_branch"
    )
    
    # Merge branches
    console.print("[bold]Merging 'classical_physics_branch' into 'modern_physics_branch'...[/bold]")
    merge_results = manager.merge_branches(
        source_branch="classical_physics_branch",
        target_branch="modern_physics_branch",
        conflict_resolution="create_both"
    )
    
    # Print merge results
    merge_table = Table(title="Branch Merge Results")
    merge_table.add_column("Metric", style="cyan")
    merge_table.add_column("Value", style="green")
    
    for key, value in merge_results.items():
        merge_table.add_row(key, str(value))
    
    console.print(merge_table)


def main():
    """Run the hierarchical knowledge management demo."""
    console.print("[bold yellow]===== Hierarchical Knowledge Management Demo =====[/bold yellow]")
    
    # Initialize components
    schema_graph, memory_store, justification_chain, manager = initialize_components()
    
    # Create knowledge structure
    branches = create_knowledge_structure(manager)
    
    # Add beliefs and justifications
    beliefs = add_beliefs_and_justifications(manager, justification_chain)
    
    # Analyze knowledge structure
    analyze_knowledge_structure(manager)
    
    # Perform branch operations
    perform_branch_operations(manager)
    
    # Visualize hierarchy
    visualize_hierarchy(manager, schema_graph)
    
    console.print("\n[bold yellow]===== Demo Completed =====[/bold yellow]")


if __name__ == "__main__":
    main() 