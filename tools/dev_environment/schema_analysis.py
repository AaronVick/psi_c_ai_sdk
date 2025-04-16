#!/usr/bin/env python3
"""
Schema Analysis Tool for Memory Structure Evolution

This module provides advanced analysis tools for comparing schema graphs and
detecting evolutionary changes in memory structures over time. It helps
identify concept drift, emergent patterns, and structural changes in
knowledge organization.

Usage:
    from tools.dev_environment.schema_analysis import SchemaAnalysis

    # Create analyzer
    analyzer = SchemaAnalysis(schema_integration)
    
    # Compare two schema graphs
    diff_report = analyzer.compare_schemas(schema1, schema2)
    
    # Analyze concept drift
    drift_report = analyzer.analyze_concept_drift(schema1, schema2)
"""

import os
import sys
import json
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SchemaAnalysis:
    """
    Schema Analysis for Memory Structure Evolution
    
    This class provides tools for analyzing and comparing schema graphs to
    detect evolutionary changes in memory structures over time.
    """
    
    def __init__(self, schema_integration=None):
        """
        Initialize the schema analysis tool.
        
        Args:
            schema_integration: Optional MemorySchemaIntegration instance
        """
        self.schema_integration = schema_integration
    
    def compare_schemas(self, schema1, schema2):
        """
        Compare two schema graphs and identify differences.
        
        Args:
            schema1: First schema graph (networkx.Graph or schema data dict)
            schema2: Second schema graph (networkx.Graph or schema data dict)
            
        Returns:
            dict: Comparison report with differences
        """
        # Convert to networkx graphs if needed
        graph1 = self._ensure_graph(schema1)
        graph2 = self._ensure_graph(schema2)
        
        # Get node sets
        nodes1 = set(graph1.nodes())
        nodes2 = set(graph2.nodes())
        
        # Find nodes present only in one graph
        unique_to_graph1 = nodes1 - nodes2
        unique_to_graph2 = nodes2 - nodes1
        common_nodes = nodes1.intersection(nodes2)
        
        # Get edge sets
        edges1 = set(graph1.edges())
        edges2 = set(graph2.edges())
        
        # Find edges present only in one graph
        unique_edges_to_graph1 = edges1 - edges2
        unique_edges_to_graph2 = edges2 - edges1
        common_edges = edges1.intersection(edges2)
        
        # Analyze node and edge changes
        node_changes = self._analyze_node_changes(
            graph1, graph2, common_nodes
        )
        
        edge_changes = self._analyze_edge_changes(
            graph1, graph2, common_edges
        )
        
        # Create comparison report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "nodes_in_graph1": len(nodes1),
                "nodes_in_graph2": len(nodes2),
                "unique_to_graph1": len(unique_to_graph1),
                "unique_to_graph2": len(unique_to_graph2),
                "common_nodes": len(common_nodes),
                "edges_in_graph1": len(edges1),
                "edges_in_graph2": len(edges2),
                "unique_edges_to_graph1": len(unique_edges_to_graph1),
                "unique_edges_to_graph2": len(unique_edges_to_graph2),
                "common_edges": len(common_edges),
                "node_changes": len(node_changes),
                "edge_changes": len(edge_changes)
            },
            "unique_nodes": {
                "graph1": list(unique_to_graph1),
                "graph2": list(unique_to_graph2)
            },
            "unique_edges": {
                "graph1": [{"source": s, "target": t} for s, t in unique_edges_to_graph1],
                "graph2": [{"source": s, "target": t} for s, t in unique_edges_to_graph2]
            },
            "node_changes": node_changes,
            "edge_changes": edge_changes
        }
        
        # Add memory type analysis
        report["memory_type_changes"] = self._analyze_memory_type_changes(graph1, graph2)
        
        # Add concept analysis
        report["concept_changes"] = self._analyze_concept_changes(graph1, graph2)
        
        return report
    
    def analyze_concept_drift(self, schema1, schema2, concept_threshold=0.7):
        """
        Analyze concept drift between two schema graphs.
        
        Args:
            schema1: First schema graph (networkx.Graph or schema data dict)
            schema2: Second schema graph (networkx.Graph or schema data dict)
            concept_threshold: Threshold for concept similarity
            
        Returns:
            dict: Concept drift analysis report
        """
        # Convert to networkx graphs if needed
        graph1 = self._ensure_graph(schema1)
        graph2 = self._ensure_graph(schema2)
        
        # Get concept nodes
        concepts1 = [n for n, attr in graph1.nodes(data=True) 
                    if attr.get('node_type') == 'concept']
        concepts2 = [n for n, attr in graph2.nodes(data=True) 
                    if attr.get('node_type') == 'concept']
        
        # Analyze concept evolution
        evolved_concepts = []
        new_concepts = []
        removed_concepts = []
        
        # Track concepts that have been matched
        matched_concepts2 = set()
        
        # For each concept in graph1, find similar concepts in graph2
        for concept_id1 in concepts1:
            concept1 = graph1.nodes[concept_id1]
            concept_name1 = concept1.get('name', concept_id1)
            
            # Get connected memories
            connected_memories1 = [n for n in graph1.neighbors(concept_id1)
                                 if graph1.nodes[n].get('node_type') == 'memory']
            
            # Find matching concept in graph2
            best_match = None
            best_similarity = 0
            
            for concept_id2 in concepts2:
                if concept_id2 in matched_concepts2:
                    continue
                
                concept2 = graph2.nodes[concept_id2]
                concept_name2 = concept2.get('name', concept_id2)
                
                # Get connected memories
                connected_memories2 = [n for n in graph2.neighbors(concept_id2)
                                     if graph2.nodes[n].get('node_type') == 'memory']
                
                # Calculate concept similarity
                name_similarity = self._calculate_text_similarity(concept_name1, concept_name2)
                
                # Calculate memory overlap
                common_memories = set(connected_memories1).intersection(set(connected_memories2))
                memory_overlap = len(common_memories) / max(1, min(len(connected_memories1), len(connected_memories2)))
                
                # Combined similarity
                similarity = 0.5 * name_similarity + 0.5 * memory_overlap
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (concept_id2, similarity)
            
            # If a good match was found
            if best_match and best_similarity >= concept_threshold:
                concept_id2, similarity = best_match
                matched_concepts2.add(concept_id2)
                
                # Check for changes
                evolution = {
                    "from": {
                        "id": concept_id1,
                        "name": concept_name1,
                        "connected_memories": len(connected_memories1)
                    },
                    "to": {
                        "id": concept_id2,
                        "name": graph2.nodes[concept_id2].get('name', concept_id2),
                        "connected_memories": len([n for n in graph2.neighbors(concept_id2)
                                                if graph2.nodes[n].get('node_type') == 'memory'])
                    },
                    "similarity": best_similarity
                }
                evolved_concepts.append(evolution)
            else:
                # Concept was removed
                removed_concepts.append({
                    "id": concept_id1,
                    "name": concept_name1,
                    "connected_memories": len(connected_memories1)
                })
        
        # Find new concepts (those in graph2 that weren't matched)
        for concept_id2 in concepts2:
            if concept_id2 not in matched_concepts2:
                concept2 = graph2.nodes[concept_id2]
                concept_name2 = concept2.get('name', concept_id2)
                connected_memories2 = [n for n in graph2.neighbors(concept_id2)
                                     if graph2.nodes[n].get('node_type') == 'memory']
                
                new_concepts.append({
                    "id": concept_id2,
                    "name": concept_name2,
                    "connected_memories": len(connected_memories2)
                })
        
        # Create drift report
        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "concepts_in_schema1": len(concepts1),
                "concepts_in_schema2": len(concepts2),
                "evolved_concepts": len(evolved_concepts),
                "new_concepts": len(new_concepts),
                "removed_concepts": len(removed_concepts)
            },
            "evolved_concepts": evolved_concepts,
            "new_concepts": new_concepts,
            "removed_concepts": removed_concepts,
            "similarity_threshold": concept_threshold
        }
        
        return drift_report
    
    def analyze_schema_evolution(self, schema_snapshots, time_points=None):
        """
        Analyze evolution of schema across multiple time points.
        
        Args:
            schema_snapshots: List of schema graphs ordered by time
            time_points: Optional list of time labels for each snapshot
            
        Returns:
            dict: Evolution analysis report
        """
        if not schema_snapshots:
            return {"error": "No schema snapshots provided"}
        
        if len(schema_snapshots) < 2:
            return {"error": "Need at least 2 snapshots for evolution analysis"}
        
        # Ensure we have time points
        if time_points is None:
            time_points = [f"T{i}" for i in range(len(schema_snapshots))]
        elif len(time_points) != len(schema_snapshots):
            time_points = time_points[:len(schema_snapshots)]
            while len(time_points) < len(schema_snapshots):
                time_points.append(f"T{len(time_points)}")
        
        # Analyze pairwise evolution
        evolution_steps = []
        
        for i in range(len(schema_snapshots) - 1):
            step_report = {
                "from_time": time_points[i],
                "to_time": time_points[i+1],
                "comparison": self.compare_schemas(schema_snapshots[i], schema_snapshots[i+1]),
                "concept_drift": self.analyze_concept_drift(schema_snapshots[i], schema_snapshots[i+1])
            }
            evolution_steps.append(step_report)
        
        # Calculate overall statistics
        total_new_nodes = sum(step["comparison"]["summary"]["unique_to_graph2"] 
                             for step in evolution_steps)
        total_removed_nodes = sum(step["comparison"]["summary"]["unique_to_graph1"] 
                                 for step in evolution_steps)
        total_new_edges = sum(len(step["comparison"]["unique_edges"]["graph2"]) 
                             for step in evolution_steps)
        total_new_concepts = sum(len(step["concept_drift"]["new_concepts"]) 
                                for step in evolution_steps)
        total_removed_concepts = sum(len(step["concept_drift"]["removed_concepts"]) 
                                    for step in evolution_steps)
        
        # Create evolution report
        evolution_report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "num_snapshots": len(schema_snapshots),
                "time_span": f"{time_points[0]} to {time_points[-1]}",
                "total_new_nodes": total_new_nodes,
                "total_removed_nodes": total_removed_nodes,
                "total_new_edges": total_new_edges,
                "total_new_concepts": total_new_concepts,
                "total_removed_concepts": total_removed_concepts
            },
            "evolution_steps": evolution_steps,
            "time_points": time_points
        }
        
        return evolution_report
    
    def visualize_schema_evolution(self, schema_snapshots, time_points=None, 
                                  output_path=None, show=True):
        """
        Visualize schema evolution across multiple time points.
        
        Args:
            schema_snapshots: List of schema graphs ordered by time
            time_points: Optional list of time labels for each snapshot
            output_path: Path to save the visualization
            show: Whether to display the visualization
            
        Returns:
            matplotlib.Figure: The figure object
        """
        if not schema_snapshots:
            return None
        
        # Ensure we have time points
        if time_points is None:
            time_points = [f"T{i}" for i in range(len(schema_snapshots))]
        elif len(time_points) != len(schema_snapshots):
            time_points = time_points[:len(schema_snapshots)]
            while len(time_points) < len(schema_snapshots):
                time_points.append(f"T{len(time_points)}")
        
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        
        # Track metrics across time
        node_counts = []
        edge_counts = []
        concept_counts = []
        memory_type_counts = []
        
        # Analyze each snapshot
        for i, schema in enumerate(schema_snapshots):
            graph = self._ensure_graph(schema)
            
            # Count nodes and edges
            node_counts.append(len(graph.nodes()))
            edge_counts.append(len(graph.edges()))
            
            # Count concepts
            concept_count = len([n for n, attr in graph.nodes(data=True) 
                               if attr.get('node_type') == 'concept'])
            concept_counts.append(concept_count)
            
            # Count memory types
            memory_types = Counter([attr.get('memory_type', 'unknown') 
                                  for n, attr in graph.nodes(data=True)
                                  if attr.get('node_type') == 'memory'])
            memory_type_counts.append(memory_types)
        
        # Plot node and edge counts
        ax1 = fig.add_subplot(221)
        ax1.plot(time_points, node_counts, 'b-o', label='Nodes')
        ax1.plot(time_points, edge_counts, 'r-o', label='Edges')
        ax1.set_title('Graph Size Evolution')
        ax1.set_xlabel('Time Point')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot concept counts
        ax2 = fig.add_subplot(222)
        ax2.plot(time_points, concept_counts, 'g-o')
        ax2.set_title('Concept Evolution')
        ax2.set_xlabel('Time Point')
        ax2.set_ylabel('Number of Concepts')
        ax2.grid(True, alpha=0.3)
        
        # Plot memory type distribution
        ax3 = fig.add_subplot(223)
        memory_types = set()
        for counts in memory_type_counts:
            memory_types.update(counts.keys())
        
        for memory_type in memory_types:
            type_counts = [counts.get(memory_type, 0) for counts in memory_type_counts]
            ax3.plot(time_points, type_counts, 'o-', label=memory_type)
        
        ax3.set_title('Memory Type Distribution')
        ax3.set_xlabel('Time Point')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot concept drift rate
        ax4 = fig.add_subplot(224)
        concept_change_rates = []
        
        for i in range(len(schema_snapshots) - 1):
            drift = self.analyze_concept_drift(schema_snapshots[i], schema_snapshots[i+1])
            new_rate = len(drift["new_concepts"]) / max(1, concept_counts[i])
            removed_rate = len(drift["removed_concepts"]) / max(1, concept_counts[i])
            evolved_rate = len(drift["evolved_concepts"]) / max(1, concept_counts[i])
            concept_change_rates.append((new_rate, removed_rate, evolved_rate))
        
        # Add a zero point for the last time point (nothing to compare with)
        concept_change_rates.append((0, 0, 0))
        
        new_rates = [r[0] for r in concept_change_rates]
        removed_rates = [r[1] for r in concept_change_rates]
        evolved_rates = [r[2] for r in concept_change_rates]
        
        ax4.plot(time_points, new_rates, 'g-o', label='New Concepts')
        ax4.plot(time_points, removed_rates, 'r-o', label='Removed Concepts')
        ax4.plot(time_points, evolved_rates, 'b-o', label='Evolved Concepts')
        ax4.set_title('Concept Change Rates')
        ax4.set_xlabel('Time Point')
        ax4.set_ylabel('Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def identify_pivot_concepts(self, schema_graph, top_n=5):
        """
        Identify pivot concepts that bridge different memory clusters.
        
        Args:
            schema_graph: Schema graph to analyze
            top_n: Number of top pivot concepts to return
            
        Returns:
            list: Top pivot concepts
        """
        graph = self._ensure_graph(schema_graph)
        
        # Get concept nodes
        concept_nodes = [n for n, attr in graph.nodes(data=True) 
                        if attr.get('node_type') == 'concept']
        
        pivot_scores = {}
        
        for concept_id in concept_nodes:
            # Get connected memories
            connected_memories = [n for n in graph.neighbors(concept_id)
                               if graph.nodes[n].get('node_type') == 'memory']
            
            if len(connected_memories) <= 1:
                continue
            
            # Get memory types of connected memories
            memory_types = Counter([graph.nodes[m].get('memory_type', 'unknown')
                                 for m in connected_memories])
            
            # Calculate diversity of memory types
            type_diversity = len(memory_types) / max(1, len(connected_memories))
            
            # Calculate betweenness centrality score
            try:
                betweenness = nx.betweenness_centrality(graph, k=min(100, len(graph.nodes)))
                centrality_score = betweenness.get(concept_id, 0)
            except:
                centrality_score = 0
            
            # Calculate pivot score based on centrality and diversity
            pivot_score = 0.6 * centrality_score + 0.4 * type_diversity
            
            # Store the score
            pivot_scores[concept_id] = {
                "id": concept_id,
                "name": graph.nodes[concept_id].get('name', concept_id),
                "connected_memories": len(connected_memories),
                "memory_types": dict(memory_types),
                "centrality": centrality_score,
                "type_diversity": type_diversity,
                "pivot_score": pivot_score
            }
        
        # Sort by pivot score
        top_pivots = sorted(pivot_scores.values(), 
                          key=lambda x: x["pivot_score"], 
                          reverse=True)[:top_n]
        
        return top_pivots
    
    def _ensure_graph(self, schema):
        """
        Ensure the schema is a networkx graph.
        
        Args:
            schema: Schema graph or dict
            
        Returns:
            networkx.Graph: The schema graph
        """
        if isinstance(schema, nx.Graph):
            return schema
        
        # Convert dict to graph
        graph = nx.Graph()
        
        # Add nodes
        for node_data in schema.get('nodes', []):
            node_id = node_data.pop('id')
            graph.add_node(node_id, **node_data)
        
        # Add edges
        for edge_data in schema.get('edges', []):
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            graph.add_edge(source, target, **edge_data)
        
        return graph
    
    def _analyze_node_changes(self, graph1, graph2, common_nodes):
        """
        Analyze changes in node attributes between graphs.
        
        Args:
            graph1: First graph
            graph2: Second graph
            common_nodes: Set of common node IDs
            
        Returns:
            list: Node attribute changes
        """
        changes = []
        
        for node_id in common_nodes:
            attrs1 = graph1.nodes[node_id]
            attrs2 = graph2.nodes[node_id]
            
            # Compare attributes
            diff_attrs = {}
            for key in set(attrs1.keys()).union(set(attrs2.keys())):
                if key not in attrs1:
                    diff_attrs[key] = {"added": attrs2[key]}
                elif key not in attrs2:
                    diff_attrs[key] = {"removed": attrs1[key]}
                elif attrs1[key] != attrs2[key]:
                    diff_attrs[key] = {"from": attrs1[key], "to": attrs2[key]}
            
            if diff_attrs:
                changes.append({
                    "node_id": node_id,
                    "node_type": attrs1.get('node_type', attrs2.get('node_type', 'unknown')),
                    "attribute_changes": diff_attrs
                })
        
        return changes
    
    def _analyze_edge_changes(self, graph1, graph2, common_edges):
        """
        Analyze changes in edge attributes between graphs.
        
        Args:
            graph1: First graph
            graph2: Second graph
            common_edges: Set of common edges
            
        Returns:
            list: Edge attribute changes
        """
        changes = []
        
        for edge in common_edges:
            source, target = edge
            attrs1 = graph1.edges[source, target]
            attrs2 = graph2.edges[source, target]
            
            # Compare attributes
            diff_attrs = {}
            for key in set(attrs1.keys()).union(set(attrs2.keys())):
                if key not in attrs1:
                    diff_attrs[key] = {"added": attrs2[key]}
                elif key not in attrs2:
                    diff_attrs[key] = {"removed": attrs1[key]}
                elif attrs1[key] != attrs2[key]:
                    diff_attrs[key] = {"from": attrs1[key], "to": attrs2[key]}
            
            if diff_attrs:
                changes.append({
                    "source": source,
                    "target": target,
                    "attribute_changes": diff_attrs
                })
        
        return changes
    
    def _analyze_memory_type_changes(self, graph1, graph2):
        """
        Analyze changes in memory type distribution.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            dict: Memory type distribution changes
        """
        # Count memory types in first graph
        memory_types1 = Counter([
            attr.get('memory_type', 'unknown') 
            for n, attr in graph1.nodes(data=True)
            if attr.get('node_type') == 'memory'
        ])
        
        # Count memory types in second graph
        memory_types2 = Counter([
            attr.get('memory_type', 'unknown') 
            for n, attr in graph2.nodes(data=True)
            if attr.get('node_type') == 'memory'
        ])
        
        # Calculate changes
        all_types = set(memory_types1.keys()).union(set(memory_types2.keys()))
        
        type_changes = {}
        for memory_type in all_types:
            count1 = memory_types1.get(memory_type, 0)
            count2 = memory_types2.get(memory_type, 0)
            
            if count1 != count2:
                type_changes[memory_type] = {
                    "from": count1,
                    "to": count2,
                    "change": count2 - count1
                }
        
        return {
            "types_in_graph1": dict(memory_types1),
            "types_in_graph2": dict(memory_types2),
            "changes": type_changes
        }
    
    def _analyze_concept_changes(self, graph1, graph2):
        """
        Analyze changes in concepts.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            dict: Concept changes
        """
        # Get concept nodes in first graph
        concepts1 = [
            (n, attr.get('name', n))
            for n, attr in graph1.nodes(data=True)
            if attr.get('node_type') == 'concept'
        ]
        
        # Get concept nodes in second graph
        concepts2 = [
            (n, attr.get('name', n))
            for n, attr in graph2.nodes(data=True)
            if attr.get('node_type') == 'concept'
        ]
        
        # Get concept counts
        concept_count1 = len(concepts1)
        concept_count2 = len(concepts2)
        
        # Extract concept names
        concept_names1 = [name for _, name in concepts1]
        concept_names2 = [name for _, name in concepts2]
        
        # Find common concept names
        common_names = set(concept_names1).intersection(set(concept_names2))
        unique_to_graph1 = set(concept_names1) - set(concept_names2)
        unique_to_graph2 = set(concept_names2) - set(concept_names1)
        
        return {
            "concept_count_in_graph1": concept_count1,
            "concept_count_in_graph2": concept_count2,
            "common_concept_names": list(common_names),
            "unique_to_graph1": list(unique_to_graph1),
            "unique_to_graph2": list(unique_to_graph2),
            "concept_change": concept_count2 - concept_count1
        }
    
    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score (0-1)
        """
        # Simple word overlap-based similarity
        if not text1 or not text2:
            return 0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / max(1, union)
    
def export_schema_evolution_report(report, output_path):
    """
    Export a schema evolution report to a file.
    
    Args:
        report: Schema evolution report
        output_path: Path to save the report
        
    Returns:
        bool: True if export was successful
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        return False

def import_schema_evolution_report(input_path):
    """
    Import a schema evolution report from a file.
    
    Args:
        input_path: Path to load the report from
        
    Returns:
        dict: Loaded report
    """
    try:
        with open(input_path, 'r') as f:
            report = json.load(f)
        return report
    except Exception as e:
        logger.error(f"Error importing report: {e}")
        return None 