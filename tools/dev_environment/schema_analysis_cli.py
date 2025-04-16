#!/usr/bin/env python3
"""
Schema Analysis CLI

Command-line interface for the Schema Analysis Tool, which provides
advanced analysis functionality for memory schema graphs and their evolution
over time.

Usage:
    python schema_analysis_cli.py compare --schema1 path/to/schema1.json --schema2 path/to/schema2.json
    python schema_analysis_cli.py drift --schema1 path/to/schema1.json --schema2 path/to/schema2.json
    python schema_analysis_cli.py evolution --schemas path/to/schema1.json path/to/schema2.json path/to/schema3.json
    python schema_analysis_cli.py visualize-evolution --schemas path/to/schema*.json --output evolution.png
    python schema_analysis_cli.py pivot-concepts --schema path/to/schema.json --top 5
"""

import os
import sys
import json
import glob
import argparse
import logging
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Optional, Dict, Any

# Ensure the dev_environment package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tools.dev_environment.schema_analysis import (
    SchemaAnalysis, 
    export_schema_evolution_report,
    import_schema_evolution_report
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_schema_from_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a schema graph from a JSON file.
    
    Args:
        file_path: Path to the schema file
        
    Returns:
        Dict representation of the schema graph or None if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            schema = json.load(f)
        return schema
    except Exception as e:
        logger.error(f"Error loading schema from {file_path}: {e}")
        return None

def save_report_to_file(report: Dict[str, Any], output_path: str) -> bool:
    """
    Save an analysis report to a file.
    
    Args:
        report: Analysis report to save
        output_path: Path to save the report
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving report to {output_path}: {e}")
        return False

def compare_schemas_command(args):
    """
    Compare two schema graphs and print/save the comparison report.
    
    Args:
        args: Command-line arguments
    """
    # Load schemas
    schema1 = load_schema_from_file(args.schema1)
    schema2 = load_schema_from_file(args.schema2)
    
    if not schema1 or not schema2:
        logger.error("Failed to load schemas for comparison")
        return 1
    
    # Create analyzer and compare schemas
    analyzer = SchemaAnalysis()
    comparison_report = analyzer.compare_schemas(schema1, schema2)
    
    # Print summary
    print("\nSchema Comparison Report Summary:")
    print(f"Timestamp: {comparison_report['timestamp']}")
    print("\nNode Changes:")
    print(f"- Nodes in Schema 1: {comparison_report['summary']['nodes_in_graph1']}")
    print(f"- Nodes in Schema 2: {comparison_report['summary']['nodes_in_graph2']}")
    print(f"- Unique to Schema 1: {comparison_report['summary']['unique_to_graph1']}")
    print(f"- Unique to Schema 2: {comparison_report['summary']['unique_to_graph2']}")
    print(f"- Common Nodes: {comparison_report['summary']['common_nodes']}")
    print(f"- Nodes with Attribute Changes: {comparison_report['summary']['node_changes']}")
    
    print("\nEdge Changes:")
    print(f"- Edges in Schema 1: {comparison_report['summary']['edges_in_graph1']}")
    print(f"- Edges in Schema 2: {comparison_report['summary']['edges_in_graph2']}")
    print(f"- Unique to Schema 1: {comparison_report['summary']['unique_edges_to_graph1']}")
    print(f"- Unique to Schema 2: {comparison_report['summary']['unique_edges_to_graph2']}")
    print(f"- Common Edges: {comparison_report['summary']['common_edges']}")
    print(f"- Edges with Attribute Changes: {comparison_report['summary']['edge_changes']}")
    
    # Print concept changes
    print("\nConcept Changes:")
    print(f"- Concepts in Schema 1: {comparison_report['concept_changes']['concept_count_in_graph1']}")
    print(f"- Concepts in Schema 2: {comparison_report['concept_changes']['concept_count_in_graph2']}")
    print(f"- Common Concepts: {len(comparison_report['concept_changes']['common_concept_names'])}")
    print(f"- Unique to Schema 1: {len(comparison_report['concept_changes']['unique_to_graph1'])}")
    print(f"- Unique to Schema 2: {len(comparison_report['concept_changes']['unique_to_graph2'])}")
    
    # Save report if output path is specified
    if args.output:
        save_report_to_file(comparison_report, args.output)
    
    return 0

def analyze_concept_drift_command(args):
    """
    Analyze concept drift between two schema graphs.
    
    Args:
        args: Command-line arguments
    """
    # Load schemas
    schema1 = load_schema_from_file(args.schema1)
    schema2 = load_schema_from_file(args.schema2)
    
    if not schema1 or not schema2:
        logger.error("Failed to load schemas for concept drift analysis")
        return 1
    
    # Create analyzer and analyze concept drift
    analyzer = SchemaAnalysis()
    drift_report = analyzer.analyze_concept_drift(
        schema1, schema2, concept_threshold=args.threshold
    )
    
    # Print summary
    print("\nConcept Drift Analysis Report Summary:")
    print(f"Timestamp: {drift_report['timestamp']}")
    print(f"Similarity Threshold: {drift_report['similarity_threshold']}")
    print(f"Concepts in Schema 1: {drift_report['summary']['concepts_in_schema1']}")
    print(f"Concepts in Schema 2: {drift_report['summary']['concepts_in_schema2']}")
    print(f"Evolved Concepts: {drift_report['summary']['evolved_concepts']}")
    print(f"New Concepts: {drift_report['summary']['new_concepts']}")
    print(f"Removed Concepts: {drift_report['summary']['removed_concepts']}")
    
    # Print evolved concepts
    if drift_report['evolved_concepts'] and not args.quiet:
        print("\nEvolved Concepts:")
        for i, evolution in enumerate(drift_report['evolved_concepts'], 1):
            print(f"{i}. {evolution['from']['name']} → {evolution['to']['name']} " +
                  f"(similarity: {evolution['similarity']:.2f})")
    
    # Print new concepts
    if drift_report['new_concepts'] and not args.quiet:
        print("\nNew Concepts:")
        for i, concept in enumerate(drift_report['new_concepts'], 1):
            print(f"{i}. {concept['name']} " +
                  f"(connected memories: {concept['connected_memories']})")
    
    # Print removed concepts
    if drift_report['removed_concepts'] and not args.quiet:
        print("\nRemoved Concepts:")
        for i, concept in enumerate(drift_report['removed_concepts'], 1):
            print(f"{i}. {concept['name']} " +
                  f"(connected memories: {concept['connected_memories']})")
    
    # Save report if output path is specified
    if args.output:
        save_report_to_file(drift_report, args.output)
    
    return 0

def analyze_schema_evolution_command(args):
    """
    Analyze evolution of schema across multiple time points.
    
    Args:
        args: Command-line arguments
    """
    # Load schemas
    schemas = []
    for schema_path in args.schemas:
        schema = load_schema_from_file(schema_path)
        if not schema:
            logger.error(f"Failed to load schema from {schema_path}")
            return 1
        schemas.append(schema)
    
    if len(schemas) < 2:
        logger.error("Need at least 2 schemas for evolution analysis")
        return 1
    
    # Parse time points if provided
    time_points = args.time_points.split(',') if args.time_points else None
    
    # Create analyzer and analyze schema evolution
    analyzer = SchemaAnalysis()
    evolution_report = analyzer.analyze_schema_evolution(schemas, time_points)
    
    # Print summary
    print("\nSchema Evolution Analysis Report Summary:")
    print(f"Timestamp: {evolution_report['timestamp']}")
    print(f"Number of Snapshots: {evolution_report['summary']['num_snapshots']}")
    print(f"Time Span: {evolution_report['summary']['time_span']}")
    print(f"Total New Nodes: {evolution_report['summary']['total_new_nodes']}")
    print(f"Total Removed Nodes: {evolution_report['summary']['total_removed_nodes']}")
    print(f"Total New Edges: {evolution_report['summary']['total_new_edges']}")
    print(f"Total New Concepts: {evolution_report['summary']['total_new_concepts']}")
    print(f"Total Removed Concepts: {evolution_report['summary']['total_removed_concepts']}")
    
    # Print evolution steps
    if not args.quiet:
        print("\nEvolution Steps:")
        for i, step in enumerate(evolution_report['evolution_steps'], 1):
            print(f"\nStep {i}: {step['from_time']} → {step['to_time']}")
            print(f"- New Nodes: {step['comparison']['summary']['unique_to_graph2']}")
            print(f"- Removed Nodes: {step['comparison']['summary']['unique_to_graph1']}")
            print(f"- New Concepts: {len(step['concept_drift']['new_concepts'])}")
            print(f"- Removed Concepts: {len(step['concept_drift']['removed_concepts'])}")
            print(f"- Evolved Concepts: {len(step['concept_drift']['evolved_concepts'])}")
    
    # Save report if output path is specified
    if args.output:
        save_report_to_file(evolution_report, args.output)
    
    return 0

def visualize_schema_evolution_command(args):
    """
    Visualize schema evolution across multiple time points.
    
    Args:
        args: Command-line arguments
    """
    # Load schemas
    schemas = []
    for schema_path in args.schemas:
        schema = load_schema_from_file(schema_path)
        if not schema:
            logger.error(f"Failed to load schema from {schema_path}")
            return 1
        schemas.append(schema)
    
    if len(schemas) < 2:
        logger.error("Need at least 2 schemas for evolution visualization")
        return 1
    
    # Parse time points if provided
    time_points = args.time_points.split(',') if args.time_points else None
    
    # Create analyzer and visualize schema evolution
    analyzer = SchemaAnalysis()
    fig = analyzer.visualize_schema_evolution(
        schemas, time_points, args.output, not args.no_display
    )
    
    if fig:
        print(f"Schema evolution visualization {'displayed' if not args.no_display else ''}" +
              f"{' and ' if not args.no_display and args.output else ''}" +
              f"{'saved to ' + args.output if args.output else ''}")
        return 0
    else:
        logger.error("Failed to create schema evolution visualization")
        return 1

def identify_pivot_concepts_command(args):
    """
    Identify pivot concepts in a schema graph.
    
    Args:
        args: Command-line arguments
    """
    # Load schema
    schema = load_schema_from_file(args.schema)
    
    if not schema:
        logger.error("Failed to load schema for pivot concept identification")
        return 1
    
    # Create analyzer and identify pivot concepts
    analyzer = SchemaAnalysis()
    pivot_concepts = analyzer.identify_pivot_concepts(schema, args.top)
    
    # Print pivot concepts
    print(f"\nTop {len(pivot_concepts)} Pivot Concepts:")
    for i, concept in enumerate(pivot_concepts, 1):
        print(f"{i}. {concept['name']} (pivot score: {concept['pivot_score']:.4f})")
        print(f"   Connected Memories: {concept['connected_memories']}")
        print(f"   Memory Types: {concept['memory_types']}")
        print(f"   Centrality: {concept['centrality']:.4f}")
        print(f"   Type Diversity: {concept['type_diversity']:.4f}")
        print()
    
    # Save report if output path is specified
    if args.output:
        save_report_to_file({
            "timestamp": datetime.now().isoformat(),
            "pivot_concepts": pivot_concepts
        }, args.output)
    
    return 0

def main():
    """
    Main entry point for the Schema Analysis CLI.
    """
    parser = argparse.ArgumentParser(
        description="Command-line interface for Schema Analysis Tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Compare schemas command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two schema graphs"
    )
    compare_parser.add_argument(
        "--schema1", required=True, help="Path to first schema file"
    )
    compare_parser.add_argument(
        "--schema2", required=True, help="Path to second schema file"
    )
    compare_parser.add_argument(
        "--output", help="Path to save the comparison report"
    )
    compare_parser.set_defaults(func=compare_schemas_command)
    
    # Analyze concept drift command
    drift_parser = subparsers.add_parser(
        "drift", help="Analyze concept drift between two schema graphs"
    )
    drift_parser.add_argument(
        "--schema1", required=True, help="Path to first schema file"
    )
    drift_parser.add_argument(
        "--schema2", required=True, help="Path to second schema file"
    )
    drift_parser.add_argument(
        "--threshold", type=float, default=0.7, 
        help="Similarity threshold for concept matching (default: 0.7)"
    )
    drift_parser.add_argument(
        "--output", help="Path to save the concept drift report"
    )
    drift_parser.add_argument(
        "--quiet", action="store_true", help="Only print summary information"
    )
    drift_parser.set_defaults(func=analyze_concept_drift_command)
    
    # Analyze schema evolution command
    evolution_parser = subparsers.add_parser(
        "evolution", help="Analyze evolution of schema across multiple time points"
    )
    evolution_parser.add_argument(
        "--schemas", required=True, nargs="+", 
        help="Paths to schema files in chronological order"
    )
    evolution_parser.add_argument(
        "--time-points", help="Comma-separated time labels for each schema"
    )
    evolution_parser.add_argument(
        "--output", help="Path to save the evolution report"
    )
    evolution_parser.add_argument(
        "--quiet", action="store_true", help="Only print summary information"
    )
    evolution_parser.set_defaults(func=analyze_schema_evolution_command)
    
    # Visualize schema evolution command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize schema evolution across multiple time points"
    )
    visualize_parser.add_argument(
        "--schemas", required=True, nargs="+", 
        help="Paths to schema files in chronological order"
    )
    visualize_parser.add_argument(
        "--time-points", help="Comma-separated time labels for each schema"
    )
    visualize_parser.add_argument(
        "--output", help="Path to save the visualization image"
    )
    visualize_parser.add_argument(
        "--no-display", action="store_true", help="Do not display the visualization"
    )
    visualize_parser.set_defaults(func=visualize_schema_evolution_command)
    
    # Identify pivot concepts command
    pivot_parser = subparsers.add_parser(
        "pivot-concepts", help="Identify pivot concepts in a schema graph"
    )
    pivot_parser.add_argument(
        "--schema", required=True, help="Path to schema file"
    )
    pivot_parser.add_argument(
        "--top", type=int, default=5, help="Number of top pivot concepts to return"
    )
    pivot_parser.add_argument(
        "--output", help="Path to save the pivot concepts report"
    )
    pivot_parser.set_defaults(func=identify_pivot_concepts_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if command is specified
    if not args.command:
        parser.print_help()
        return 1
    
    # Run the specified command
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main()) 