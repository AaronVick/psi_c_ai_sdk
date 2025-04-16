#!/usr/bin/env python3
"""
Memory Schema Integration for ΨC-AI SDK Development Environment

This module provides integration between the Memory Sandbox and schema visualization,
allowing for analysis of memory relationships, concept generation, and visual representation
of the memory-concept graph structure.
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict, Counter

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Import schema math functions
try:
    from tools.dev_environment.schema_math import (
        reflective_control_score,
        meta_objective_function,
        cognitive_debt,
        reflective_confidence_heuristic,
        complexity_budget_function,
        memory_energy_budget,
        schema_health_index,
        calculate_memory_embedding_entropy,
        calculate_graph_connectivity,
        identity_drift_measure,
        schema_annealing_temperature
    )
    HAS_SCHEMA_MATH = True
except ImportError:
    HAS_SCHEMA_MATH = False
    logging.warning("Schema math functions not available. Some advanced features will be disabled.")

logger = logging.getLogger(__name__)

def is_schema_integration_available():
    """Check if all required dependencies for schema integration are available"""
    return HAS_SKLEARN


class MemorySchemaIntegration:
    """
    Memory Schema Integration for the ΨC-AI SDK Memory Sandbox.
    
    This class provides functionality to analyze memory structures,
    build knowledge graphs, detect memory clusters, and identify
    emergent concepts from agent memory data.
    
    Attributes:
        sandbox (MemorySandbox): The memory sandbox instance
        graph (networkx.Graph): The schema graph for memory relationships
        clusters (dict): Memory clusters identified in the memory store
        concepts (dict): Generated concept suggestions from clusters
    """
    
    def __init__(self, sandbox):
        """
        Initialize the Memory Schema Integration.
        
        Args:
            sandbox: The memory sandbox instance containing the memory store
        """
        self.sandbox = sandbox
        self.graph = nx.Graph()
        self.clusters = {}
        self.concepts = {}
        self.schema_history = []
        self.schema_timestamps = []
        self.reflection_history = []
        self.iteration_count = 0
        self._total_budget = 100.0  # Default energy budget
        self._feature_costs = {
            "build_graph": 10.0,
            "detect_clusters": 15.0,
            "generate_concepts": 20.0,
            "visualize": 5.0,
            "calculate_stats": 2.0
        }
    
    def build_schema_graph(self):
        """
        Build a schema graph from the memories in the memory store.
        
        The graph represents memories as nodes and relationships between 
        memories as edges. Node attributes include memory content, type,
        importance, and creation time.
        
        Returns:
            networkx.Graph: The constructed schema graph
        """
        # Clear existing graph
        self.graph.clear()
        
        # Get all memories from the memory store
        memories = self.sandbox.memory_store.memories
        
        # Add memory nodes to the graph
        for memory_id, memory in memories.items():
            self.graph.add_node(
                memory_id,
                content=memory.content,
                memory_type=memory.memory_type,
                importance=memory.importance,
                creation_time=memory.creation_time,
                embedding=memory.embedding,
                node_type="memory"
            )
        
        # Add edges based on memory relationships
        for memory_id, memory in memories.items():
            for related_id in memory.related_memories:
                if related_id in memories:
                    self.graph.add_edge(memory_id, related_id, weight=1.0)
        
        # Add edges based on semantic similarity
        self._add_semantic_edges()
        
        # Add edges based on shared tags
        self._add_tag_based_edges()
        
        # Save current graph state to history
        self._save_current_state()
        
        # Increment iteration count
        self.iteration_count += 1
        
        return self.graph
    
    def _save_current_state(self):
        """Save the current state of the schema graph to history."""
        # Create a deep copy of the graph
        graph_copy = self.graph.copy()
        
        # Save graph to history
        self.schema_history.append(graph_copy)
        self.schema_timestamps.append(datetime.datetime.now().isoformat())
        
        # Keep history to a reasonable size
        max_history = 10
        if len(self.schema_history) > max_history:
            self.schema_history = self.schema_history[-max_history:]
            self.schema_timestamps = self.schema_timestamps[-max_history:]
    
    def _add_semantic_edges(self, similarity_threshold=0.7):
        """
        Add edges between memories based on semantic similarity.
        
        Args:
            similarity_threshold (float): Threshold for adding an edge
        """
        memories = self.sandbox.memory_store.memories
        memory_ids = list(memories.keys())
        
        # Calculate embeddings for all memories
        embeddings = []
        for memory_id in memory_ids:
            if memories[memory_id].embedding is not None:
                embeddings.append(memories[memory_id].embedding)
            else:
                # Use a zero vector if embedding is None
                embeddings.append(np.zeros(128))
        
        # Calculate similarity matrix
        if embeddings:
            similarity_matrix = cosine_similarity(embeddings)
            
            # Add edges for similar memories
            for i in range(len(memory_ids)):
                for j in range(i+1, len(memory_ids)):
                    similarity = similarity_matrix[i][j]
                    if similarity > similarity_threshold:
                        self.graph.add_edge(
                            memory_ids[i], 
                            memory_ids[j], 
                            weight=similarity,
                            edge_type="semantic"
                        )
    
    def _add_tag_based_edges(self):
        """Add edges between memories based on shared tags."""
        memories = self.sandbox.memory_store.memories
        
        # Build a dictionary of tags to memory IDs
        tag_to_memories = {}
        for memory_id, memory in memories.items():
            if hasattr(memory, 'tags') and memory.tags:
                for tag in memory.tags:
                    if tag not in tag_to_memories:
                        tag_to_memories[tag] = set()
                    tag_to_memories[tag].add(memory_id)
        
        # Add edges between memories with shared tags
        for tag, memory_ids in tag_to_memories.items():
            memory_id_list = list(memory_ids)
            for i in range(len(memory_id_list)):
                for j in range(i+1, len(memory_id_list)):
                    mem_id1, mem_id2 = memory_id_list[i], memory_id_list[j]
                    
                    # If an edge already exists, update it
                    if self.graph.has_edge(mem_id1, mem_id2):
                        current_weight = self.graph[mem_id1][mem_id2]['weight']
                        self.graph[mem_id1][mem_id2]['weight'] = current_weight + 0.2
                    else:
                        self.graph.add_edge(
                            mem_id1,
                            mem_id2,
                            weight=0.5,
                            edge_type="tag",
                            shared_tag=tag
                        )
    
    def detect_memory_clusters(self, eps=0.5, min_samples=2):
        """
        Detect clusters of memories based on embeddings.
        
        Uses DBSCAN clustering algorithm to identify groups of
        semantically similar memories.
        
        Args:
            eps (float): The maximum distance between two samples for
                         them to be considered in the same neighborhood
            min_samples (int): Minimum number of samples in a neighborhood
                              for a point to be a core point
        
        Returns:
            dict: Dictionary of clusters with memory IDs and statistics
        """
        memories = self.sandbox.memory_store.memories
        memory_ids = list(memories.keys())
        
        # Extract embeddings and handle None values
        embeddings = []
        valid_memory_ids = []
        
        for memory_id in memory_ids:
            memory = memories[memory_id]
            if memory.embedding is not None:
                embeddings.append(memory.embedding)
                valid_memory_ids.append(memory_id)
            
        if not embeddings:
            self.clusters = {}
            return self.clusters
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        labels = clustering.labels_
        
        # Process clusters
        unique_labels = set(labels)
        self.clusters = {}
        
        for cluster_id in unique_labels:
            # Skip noise points (cluster_id = -1)
            if cluster_id == -1:
                continue
                
            # Get memory IDs in this cluster
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            cluster_memory_ids = [valid_memory_ids[i] for i in cluster_indices]
            
            # Calculate cluster statistics
            memory_types = [memories[mid].memory_type for mid in cluster_memory_ids]
            avg_importance = np.mean([memories[mid].importance for mid in cluster_memory_ids])
            
            # Collect memory contents for concept generation
            memory_contents = [memories[mid].content for mid in cluster_memory_ids]
            
            # Store cluster data
            self.clusters[str(cluster_id)] = {
                "memory_ids": cluster_memory_ids,
                "memory_types": Counter(memory_types),
                "avg_importance": float(avg_importance),
                "size": len(cluster_memory_ids),
                "contents": memory_contents
            }
        
        return self.clusters
    
    def generate_concept_suggestions(self):
        """
        Generate concept suggestions for memory clusters.
        
        Analyzes memory clusters to suggest potential concepts that
        emerge from the grouped memories.
        
        Returns:
            dict: Dictionary of concept suggestions
        """
        if not self.clusters:
            self.detect_memory_clusters()
        
        self.concepts = {}
        
        for cluster_id, cluster_data in self.clusters.items():
            # Skip small clusters
            if cluster_data["size"] < 2:
                continue
            
            # Determine dominant memory type
            memory_type_counts = cluster_data["memory_types"]
            dominant_type = max(memory_type_counts.items(), key=lambda x: x[1])[0]
            
            # Extract keywords from memory contents
            contents = " ".join(cluster_data["contents"])
            
            # Simple keyword extraction (in a real system, use NLP techniques)
            words = contents.lower().split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about'}
            keywords = [word for word in words if word not in stop_words]
            
            # Count occurrences of each keyword
            keyword_counts = Counter(keywords)
            most_common_keywords = [kw for kw, count in keyword_counts.most_common(5)]
            
            # Generate a concept name
            concept_name = f"{dominant_type.capitalize()} - {most_common_keywords[0] if most_common_keywords else 'Unnamed'}"
            
            # Store concept suggestion
            self.concepts[cluster_id] = {
                "concept_name": concept_name,
                "keywords": most_common_keywords,
                "dominant_type": dominant_type,
                "memory_ids": cluster_data["memory_ids"],
                "importance": cluster_data["avg_importance"]
            }
            
            # Add concept node to the graph
            self.add_concept_node(concept_name, cluster_data["memory_ids"])
        
        return self.concepts
    
    def add_concept_node(self, concept_name, memory_ids):
        """
        Add a concept node to the schema graph.
        
        Args:
            concept_name (str): Name of the concept
            memory_ids (list): List of memory IDs associated with the concept
        
        Returns:
            str: ID of the new concept node
        """
        # Generate a unique ID for the concept
        concept_id = f"concept_{datetime.now().timestamp()}"
        
        # Add the concept node to the graph
        self.graph.add_node(
            concept_id,
            name=concept_name,
            node_type="concept",
            creation_time=datetime.now()
        )
        
        # Add edges from the concept to its associated memories
        for memory_id in memory_ids:
            if memory_id in self.graph.nodes:
                self.graph.add_edge(
                    concept_id,
                    memory_id,
                    weight=1.0,
                    edge_type="concept_memory"
                )
        
        return concept_id
    
    def find_related_memories(self, memory_id, max_depth=2):
        """
        Find memories related to a given memory based on graph connectivity.
        
        Args:
            memory_id (str): ID of the target memory
            max_depth (int): Maximum path length to consider
        
        Returns:
            dict: Dictionary of related memories with relationship information
        """
        if memory_id not in self.graph.nodes:
            return {}
        
        related_memories = {}
        
        # Get directly connected memories (depth 1)
        direct_neighbors = list(self.graph.neighbors(memory_id))
        for neighbor_id in direct_neighbors:
            # Skip concept nodes
            if self.graph.nodes[neighbor_id].get('node_type') == 'concept':
                continue
                
            edge_data = self.graph[memory_id][neighbor_id]
            related_memories[neighbor_id] = {
                "relationship_type": edge_data.get('edge_type', 'direct'),
                "strength": edge_data.get('weight', 1.0),
                "path_length": 1
            }
        
        # Get memories at depth 2 if requested
        if max_depth >= 2:
            depth2_memories = set()
            
            for neighbor_id in direct_neighbors:
                neighbors_of_neighbor = list(self.graph.neighbors(neighbor_id))
                for nn_id in neighbors_of_neighbor:
                    # Skip already processed nodes and concept nodes
                    if nn_id == memory_id or nn_id in direct_neighbors or \
                       self.graph.nodes[nn_id].get('node_type') == 'concept':
                        continue
                    depth2_memories.add((neighbor_id, nn_id))
            
            # Add depth 2 memories to the result
            for intermediate_id, nn_id in depth2_memories:
                edge1_weight = self.graph[memory_id][intermediate_id].get('weight', 1.0)
                edge2_weight = self.graph[intermediate_id][nn_id].get('weight', 1.0)
                combined_weight = edge1_weight * edge2_weight
                
                related_memories[nn_id] = {
                    "relationship_type": "indirect",
                    "intermediate_id": intermediate_id,
                    "strength": combined_weight,
                    "path_length": 2
                }
        
        return related_memories
    
    def visualize_schema_graph(self, output_path=None, show=True):
        """
        Visualize the schema graph.
        
        Args:
            output_path (str): Path to save the visualization
            show (bool): Whether to display the visualization
        
        Returns:
            matplotlib.Figure: The figure object
        """
        # Create a spring layout
        pos = nx.spring_layout(self.graph)
        
        # Create figure and axis
        plt.figure(figsize=(12, 10))
        
        # Get node types
        memory_nodes = [n for n, attr in self.graph.nodes(data=True) 
                       if attr.get('node_type') == 'memory']
        concept_nodes = [n for n, attr in self.graph.nodes(data=True) 
                        if attr.get('node_type') == 'concept']
        
        # Draw memory nodes
        nx.draw_networkx_nodes(self.graph, pos, nodelist=memory_nodes, 
                              node_color='lightblue', node_size=100)
        
        # Draw concept nodes
        nx.draw_networkx_nodes(self.graph, pos, nodelist=concept_nodes, 
                              node_color='lightgreen', node_size=300)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)
        
        # Add labels for concept nodes only to avoid clutter
        concept_labels = {n: self.graph.nodes[n].get('name', n) for n in concept_nodes}
        nx.draw_networkx_labels(self.graph, pos, labels=concept_labels, font_size=10)
        
        plt.title("Memory Schema Graph")
        plt.axis('off')
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def export_schema_graph(self, output_path):
        """
        Export the schema graph to a JSON file.
        
        Args:
            output_path (str): Path to save the exported graph
        
        Returns:
            dict: The exported schema data
        """
        # Convert graph to dictionary
        nodes_data = []
        for node_id, attrs in self.graph.nodes(data=True):
            node_data = {"id": node_id}
            
            # Copy node attributes, handling non-serializable objects
            for key, value in attrs.items():
                if key == 'embedding':
                    # Convert numpy array to list
                    if isinstance(value, np.ndarray):
                        node_data[key] = value.tolist()
                elif key == 'creation_time':
                    # Convert datetime to string
                    if isinstance(value, datetime):
                        node_data[key] = value.isoformat()
                else:
                    node_data[key] = value
            
            nodes_data.append(node_data)
        
        # Extract edge data
        edges_data = []
        for u, v, attrs in self.graph.edges(data=True):
            edge_data = {"source": u, "target": v}
            
            # Copy edge attributes
            for key, value in attrs.items():
                edge_data[key] = value
            
            edges_data.append(edge_data)
        
        # Create the schema data
        schema_data = {
            "nodes": nodes_data,
            "edges": edges_data,
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(schema_data, f, indent=2)
        
        return schema_data
    
    def import_schema_graph(self, input_path):
        """
        Import a schema graph from a JSON file.
        
        Args:
            input_path (str): Path to the schema graph JSON file
        
        Returns:
            networkx.Graph: The imported schema graph
        """
        # Read from file
        with open(input_path, 'r') as f:
            schema_data = json.load(f)
        
        # Create a new graph
        self.graph = nx.Graph()
        
        # Add nodes
        for node_data in schema_data["nodes"]:
            node_id = node_data.pop("id")
            
            # Handle special attributes
            for key, value in node_data.items():
                if key == 'embedding' and value is not None:
                    # Convert list back to numpy array
                    node_data[key] = np.array(value)
            
            self.graph.add_node(node_id, **node_data)
        
        # Add edges
        for edge_data in schema_data["edges"]:
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            self.graph.add_edge(source, target, **edge_data)
        
        return self.graph
    
    def generate_knowledge_report(self):
        """
        Generate a comprehensive report of the knowledge structure.
        
        Returns:
            dict: Knowledge report with concepts, clusters, and statistics
        """
        if not self.concepts:
            self.generate_concept_suggestions()
        
        # Calculate graph statistics
        stats = self.calculate_schema_statistics()
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "concepts": self.concepts,
            "clusters": self.clusters,
            "statistics": stats,
            "summary": {
                "total_memories": len([n for n, attr in self.graph.nodes(data=True) 
                                     if attr.get('node_type') == 'memory']),
                "total_concepts": len([n for n, attr in self.graph.nodes(data=True) 
                                     if attr.get('node_type') == 'concept']),
                "total_clusters": len(self.clusters)
            }
        }
        
        return report
    
    def calculate_schema_statistics(self):
        """
        Calculate statistics for the current schema graph.
        
        Returns:
            dict: Statistics about the schema graph
        """
        if not self.graph:
            return {"node_count": 0, "edge_count": 0}
        
        # Get counts by memory type
        memory_types = {}
        for node, attrs in self.graph.nodes(data=True):
            if "memory_type" in attrs:
                memory_type = attrs["memory_type"]
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
        
        # Calculate edge weight statistics
        edge_weights = [data.get("weight", 0) for _, _, data in self.graph.edges(data=True)]
        avg_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 0
        
        # Get graph connectivity metrics
        if HAS_SCHEMA_MATH:
            connectivity = calculate_graph_connectivity(self.graph)
        else:
            # Basic connectivity metrics
            connectivity = {
                "average_degree": sum(dict(self.graph.degree()).values()) / len(self.graph) if len(self.graph) > 0 else 0,
                "connected_components": nx.number_connected_components(self.graph),
                "density": nx.density(self.graph)
            }
        
        # Calculate average importance
        importance_values = []
        for node, attrs in self.graph.nodes(data=True):
            if "importance" in attrs:
                importance_values.append(attrs["importance"])
        avg_importance = sum(importance_values) / len(importance_values) if importance_values else 0
        
        return {
            "node_count": len(self.graph.nodes()),
            "edge_count": len(self.graph.edges()),
            "memory_type_distribution": memory_types,
            "average_edge_weight": avg_weight,
            "average_importance": avg_importance,
            "connectivity": connectivity
        }
    
    def calculate_cognitive_debt(self):
        """
        Calculate cognitive debt based on reflection count and schema changes.
        
        Implements formula #24: D_t = R_t - λ · (S_t + C_t)
        
        Returns:
            float: Cognitive debt score
        """
        if not HAS_SCHEMA_MATH:
            logger.warning("Schema math not available, using simplified calculation")
            return 0.0
            
        # Count reflections in history
        reflection_count = len(self.reflection_history)
        
        # Count schema changes (based on graph size changes)
        schema_changes = 0
        for i in range(1, len(self.schema_history)):
            prev_graph = self.schema_history[i-1]
            curr_graph = self.schema_history[i]
            # Count added or removed nodes as changes
            schema_changes += abs(len(curr_graph.nodes()) - len(prev_graph.nodes()))
        
        # Calculate average coherence
        coherence = 0.0
        edge_weights = [data.get("weight", 0) for _, _, data in self.graph.edges(data=True)]
        if edge_weights:
            coherence = sum(edge_weights) / len(edge_weights)
        
        # Calculate cognitive debt
        return cognitive_debt(reflection_count, schema_changes, coherence)
    
    def calculate_schema_health(self):
        """
        Calculate schema health index (Ψ-index) based on formula #44.
        
        Ψ_index = ω_1 · C̄ - ω_2 · H + ω_3 · A - ω_4 · D
        
        Returns:
            float: Schema health index
        """
        if not HAS_SCHEMA_MATH:
            logger.warning("Schema math not available, using simplified calculation")
            return 0.5  # Default middle value
            
        # Calculate average coherence
        coherence = 0.0
        edge_weights = [data.get("weight", 0) for _, _, data in self.graph.edges(data=True)]
        if edge_weights:
            coherence = sum(edge_weights) / len(edge_weights)
        
        # Calculate average entropy
        entropy = 0.0
        entropy_values = []
        for node, attrs in self.graph.nodes(data=True):
            if "embedding" in attrs and attrs["embedding"] is not None:
                entropy_values.append(calculate_memory_embedding_entropy(attrs["embedding"]))
        if entropy_values:
            entropy = sum(entropy_values) / len(entropy_values)
        
        # Calculate alignment score (percentage of memories with high coherence)
        high_coherence_count = 0
        for _, _, data in self.graph.edges(data=True):
            if data.get("weight", 0) > 0.7:  # Threshold for high coherence
                high_coherence_count += 1
        alignment = high_coherence_count / len(self.graph.edges()) if self.graph.edges() else 0
        
        # Calculate schema drift rate
        drift_rate = 0.0
        if len(self.schema_history) > 1:
            prev_graph = self.schema_history[-2]
            curr_graph = self.schema_history[-1]
            # Calculate percentage of nodes that changed
            total_nodes = len(set(prev_graph.nodes()).union(set(curr_graph.nodes())))
            different_nodes = len(set(prev_graph.nodes()).symmetric_difference(set(curr_graph.nodes())))
            drift_rate = different_nodes / total_nodes if total_nodes > 0 else 0
        
        # Calculate schema health index
        return schema_health_index(coherence, entropy, alignment, drift_rate)
    
    def calculate_complexity_budget(self):
        """
        Calculate complexity budget based on formula #18.
        
        C_t = α_1 · M_t + α_2 · S_t + α_3 · D_t
        
        Returns:
            float: Complexity budget score
        """
        if not HAS_SCHEMA_MATH:
            logger.warning("Schema math not available, using simplified calculation")
            return 50.0  # Default middle value
            
        # Count active memories
        active_memories = len(self.graph.nodes())
        
        # Calculate average schema depth (using average path length as proxy)
        try:
            connected_components = list(nx.connected_components(self.graph))
            avg_path_lengths = []
            for component in connected_components:
                if len(component) > 1:
                    subgraph = self.graph.subgraph(component)
                    # Calculate average shortest path length
                    path_length = nx.average_shortest_path_length(subgraph)
                    avg_path_lengths.append(path_length)
            
            schema_depth = sum(avg_path_lengths) / len(avg_path_lengths) if avg_path_lengths else 1.0
        except:
            schema_depth = 1.0  # Default if calculation fails
        
        # Count contradictions (edges with negative weights or marked as contradictions)
        contradiction_count = 0
        for _, _, data in self.graph.edges(data=True):
            if data.get("edge_type") == "contradiction" or data.get("weight", 0) < 0:
                contradiction_count += 1
        
        # Calculate complexity budget
        return complexity_budget_function(active_memories, schema_depth, contradiction_count)
    
    def calculate_memory_energy_usage(self):
        """
        Calculate memory energy usage and available budget based on formula #20.
        
        E_available = B - ∑_i E_i
        
        Returns:
            dict: Energy usage statistics
        """
        if not HAS_SCHEMA_MATH:
            logger.warning("Schema math not available, using simplified calculation")
            return {"total_budget": 100.0, "used_energy": 50.0, "available_energy": 50.0}
            
        # Calculate energy usage based on feature costs
        feature_costs = self._feature_costs.copy()
        
        # Adjust costs based on complexity
        node_count = len(self.graph.nodes())
        edge_count = len(self.graph.edges())
        
        # Scale costs based on graph size
        scaling_factor = 1.0
        if node_count > 100 or edge_count > 500:
            scaling_factor = 1.5
        if node_count > 500 or edge_count > 2000:
            scaling_factor = 2.0
            
        for feature in feature_costs:
            feature_costs[feature] *= scaling_factor
            
        # Calculate available energy
        available_energy = memory_energy_budget(self._total_budget, feature_costs)
        
        return {
            "total_budget": self._total_budget,
            "feature_costs": feature_costs,
            "used_energy": self._total_budget - available_energy,
            "available_energy": available_energy,
            "scaling_factor": scaling_factor
        }
    
    def process_reflection(self, trigger_type, memory_ids=None, reflection_result=None):
        """
        Process a reflection event and update the schema accordingly.
        
        Args:
            trigger_type: Type of reflection trigger (coherence, contradiction, etc.)
            memory_ids: List of memory IDs involved in reflection
            reflection_result: Result of reflection process (if any)
            
        Returns:
            dict: Information about schema changes resulting from reflection
        """
        # Record reflection event
        reflection_event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "trigger_type": trigger_type,
            "memory_ids": memory_ids or [],
            "result": reflection_result
        }
        self.reflection_history.append(reflection_event)
        
        # Update schema based on reflection
        changes = {"nodes_added": 0, "nodes_modified": 0, "edges_added": 0, "edges_modified": 0}
        
        # If reflection resulted in concept generation
        if reflection_result and "concepts" in reflection_result:
            for concept_name, concept_data in reflection_result["concepts"].items():
                # Create concept node
                concept_id = f"concept_{len(self.graph.nodes())}"
                self.graph.add_node(
                    concept_id,
                    name=concept_name,
                    node_type="concept",
                    keywords=concept_data.get("keywords", []),
                    importance=concept_data.get("importance", 0.5)
                )
                changes["nodes_added"] += 1
                
                # Connect concept to memories
                for memory_id in concept_data.get("memory_ids", []):
                    if memory_id in self.graph.nodes():
                        self.graph.add_edge(
                            concept_id,
                            memory_id,
                            weight=0.8,
                            edge_type="concept_memory"
                        )
                        changes["edges_added"] += 1
        
        # If reflection resulted in memory modifications
        if memory_ids:
            for memory_id in memory_ids:
                if memory_id in self.graph.nodes():
                    # Update memory node
                    self.graph.nodes[memory_id]["reflection_count"] = self.graph.nodes[memory_id].get("reflection_count", 0) + 1
                    changes["nodes_modified"] += 1
        
        # Save current state after reflection
        self._save_current_state()
        
        return changes
    
    def recommend_cognitive_action(self, memory_id):
        """
        Recommend cognitive action for a memory based on formula #13.
        
        Ψ_t = {
            Reflect(M_i)     if ∂ℒ/∂M_i > θ_r
            Consolidate(M_i) if C_i < τ_c ∧ H_i < τ_h ∧ R_i > ρ
            ...
        }
        
        Args:
            memory_id: ID of memory to evaluate
            
        Returns:
            str: Recommended cognitive action
        """
        if not HAS_SCHEMA_MATH or memory_id not in self.graph.nodes():
            return "unknown"
            
        # Get memory attributes
        memory_node = self.graph.nodes[memory_id]
        memory_importance = memory_node.get("importance", 0.5)
        reflection_count = memory_node.get("reflection_count", 0)
        
        # Calculate coherence for this memory
        coherence = 0.0
        edge_weights = []
        for _, target, data in self.graph.edges(memory_id, data=True):
            edge_weights.append(data.get("weight", 0))
        if edge_weights:
            coherence = sum(edge_weights) / len(edge_weights)
            
        # Calculate entropy for this memory
        entropy = 0.0
        if "embedding" in memory_node and memory_node["embedding"] is not None:
            entropy = calculate_memory_embedding_entropy(memory_node["embedding"])
            
        # Get global coherence statistics
        global_coherence = 0.0
        all_edge_weights = [data.get("weight", 0) for _, _, data in self.graph.edges(data=True)]
        if all_edge_weights:
            global_coherence = sum(all_edge_weights) / len(all_edge_weights)
            
        # Calculate global coherence delta
        global_coherence_delta = 0.0
        if len(self.schema_history) > 1:
            prev_graph = self.schema_history[-2]
            prev_edge_weights = [data.get("weight", 0) for _, _, data in prev_graph.edges(data=True)]
            if prev_edge_weights:
                prev_global_coherence = sum(prev_edge_weights) / len(prev_edge_weights)
                global_coherence_delta = global_coherence - prev_global_coherence
                
        # Calculate scores for cognitive actions
        scores = reflective_control_score(
            memory_importance, 
            coherence, 
            entropy, 
            reflection_count,
            global_coherence,
            global_coherence_delta
        )
        
        # Return action with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
        
    def export_schema_snapshot(self, output_path=None):
        """
        Export the current schema graph as a JSON snapshot.
        
        Args:
            output_path: Path to export the snapshot (optional)
            
        Returns:
            dict: Serialized schema data
        """
        # Create snapshot data
        snapshot = {
            "timestamp": datetime.datetime.now().isoformat(),
            "iteration": self.iteration_count,
            "nodes": [],
            "edges": [],
            "stats": self.calculate_schema_statistics()
        }
        
        # Add nodes to snapshot
        for node, attrs in self.graph.nodes(data=True):
            node_data = {"id": node}
            for key, value in attrs.items():
                # Skip embeddings for snapshot export (too large)
                if key == "embedding":
                    continue
                node_data[key] = value
            snapshot["nodes"].append(node_data)
            
        # Add edges to snapshot
        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {"source": source, "target": target}
            for key, value in attrs.items():
                edge_data[key] = value
            snapshot["edges"].append(edge_data)
            
        # Add health metrics
        if HAS_SCHEMA_MATH:
            snapshot["health"] = {
                "cognitive_debt": self.calculate_cognitive_debt(),
                "schema_health": self.calculate_schema_health(),
                "complexity_budget": self.calculate_complexity_budget(),
                "energy_usage": self.calculate_memory_energy_usage()
            }
            
        # Export snapshot to file if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(snapshot, f, indent=2)
                
        return snapshot
    
    def import_schema_snapshot(self, snapshot_data=None, input_path=None):
        """
        Import a schema snapshot and restore the graph.
        
        Args:
            snapshot_data: Serialized schema data (optional)
            input_path: Path to import the snapshot from (optional)
            
        Returns:
            bool: Success of import operation
        """
        # Load snapshot from file if path provided
        if snapshot_data is None and input_path:
            try:
                with open(input_path, 'r') as f:
                    snapshot_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load snapshot from {input_path}: {e}")
                return False
                
        if not snapshot_data:
            logger.error("No snapshot data provided")
            return False
            
        # Create new graph
        new_graph = nx.Graph()
        
        # Add nodes to graph
        for node_data in snapshot_data.get("nodes", []):
            node_id = node_data.pop("id")
            new_graph.add_node(node_id, **node_data)
            
        # Add edges to graph
        for edge_data in snapshot_data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            new_graph.add_edge(source, target, **edge_data)
            
        # Replace current graph with imported graph
        self.graph = new_graph
        
        # Update iteration count if available
        if "iteration" in snapshot_data:
            self.iteration_count = snapshot_data["iteration"]
            
        # Save current state
        self._save_current_state()
        
        return True
    
    def find_schema_patterns(self):
        """
        Find recurring patterns in the schema.
        
        Returns:
            dict: Dictionary of identified patterns
        """
        patterns = {}
        
        # Find frequently co-occurring memory types
        memory_nodes = [n for n, attr in self.graph.nodes(data=True) 
                       if attr.get('node_type') == 'memory']
        
        type_pairs = []
        for memory_id in memory_nodes:
            memory_type = self.graph.nodes[memory_id].get('memory_type')
            
            for neighbor_id in self.graph.neighbors(memory_id):
                if neighbor_id in memory_nodes:
                    neighbor_type = self.graph.nodes[neighbor_id].get('memory_type')
                    if memory_type and neighbor_type and memory_id < neighbor_id:
                        type_pairs.append((memory_type, neighbor_type))
        
        # Count type pairs
        type_pair_counts = Counter(type_pairs)
        patterns["type_pairs"] = {str(pair): count for pair, count in type_pair_counts.most_common(5)}
        
        return patterns
    
    def analyze_memory_structure(self):
        """
        Analyze the memory structure and provide insights.
        
        Returns:
            dict: Dictionary of insights about the memory structure
        """
        if not self.concepts:
            self.generate_concept_suggestions()
        
        insights = {
            "dominant_memory_types": {},
            "concept_distribution": {},
            "memory_connectivity": {},
            "recommendations": []
        }
        
        # Analyze dominant memory types
        memory_nodes = [n for n, attr in self.graph.nodes(data=True) 
                       if attr.get('node_type') == 'memory']
        
        memory_types = {}
        for node in memory_nodes:
            memory_type = self.graph.nodes[node].get('memory_type')
            if memory_type:
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
        
        # Calculate percentages
        total_memories = len(memory_nodes)
        if total_memories > 0:
            for memory_type, count in memory_types.items():
                insights["dominant_memory_types"][memory_type] = {
                    "count": count,
                    "percentage": count / total_memories * 100
                }
        
        # Analyze concept distribution
        for concept_id, concept_data in self.concepts.items():
            concept_name = concept_data["concept_name"]
            memory_count = len(concept_data["memory_ids"])
            
            insights["concept_distribution"][concept_name] = {
                "memory_count": memory_count,
                "importance": concept_data["importance"]
            }
        
        # Memory connectivity analysis
        if memory_nodes:
            degrees = [self.graph.degree(n) for n in memory_nodes]
            avg_degree = sum(degrees) / len(degrees)
            max_degree = max(degrees) if degrees else 0
            
            insights["memory_connectivity"] = {
                "average_connections": avg_degree,
                "max_connections": max_degree,
                "isolated_memories": degrees.count(0)
            }
        
        # Generate recommendations
        if insights["memory_connectivity"].get("isolated_memories", 0) > 0:
            insights["recommendations"].append(
                "Consider strengthening connections for isolated memories"
            )
        
        if len(self.concepts) == 0 and total_memories > 5:
            insights["recommendations"].append(
                "No concepts detected despite having memories. Consider adjusting clustering parameters."
            )
        
        return insights


def integrate_with_memory_sandbox(memory_sandbox):
    """
    Integrate schema functionality with a Memory Sandbox instance
    
    Args:
        memory_sandbox: The Memory Sandbox instance
    
    Returns:
        The MemorySchemaIntegration instance
    """
    # Check if integration is available
    if not is_schema_integration_available():
        logger.warning("Schema integration requires scikit-learn. Please install it with 'pip install scikit-learn'")
    
    # Create the integration
    schema_integration = MemorySchemaIntegration(memory_sandbox)
    
    # Add to memory sandbox
    memory_sandbox.schema_integration = schema_integration
    
    # Add schema-related methods to the memory sandbox
    memory_sandbox.create_schema_graph = schema_integration.build_schema_graph
    memory_sandbox.detect_memory_clusters = schema_integration.detect_memory_clusters
    memory_sandbox.suggest_concepts = schema_integration.generate_concept_suggestions
    memory_sandbox.get_memory_connections = schema_integration.find_related_memories
    memory_sandbox.find_similar_memories = schema_integration.find_related_memories
    
    logger.info("Integrated schema functionality with Memory Sandbox")
    
    return schema_integration 