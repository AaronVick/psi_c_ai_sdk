"""
Schema Fingerprinting and Merkle Integrity for ΨC-AI SDK

This module provides tools for creating unique fingerprints of schema graphs,
enabling schema version control, diff calculation, and integrity verification.
Uses Merkle tree concepts to efficiently compute and compare schema state.
"""

import hashlib
import json
import time
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from collections import defaultdict

from psi_c_ai_sdk.schema.schema import SchemaGraph, SchemaNode, SchemaEdge


class SchemaFingerprint:
    """
    Creates and manages cryptographic fingerprints of schema graphs.
    
    Uses Merkle tree concepts to create a unique hash representing the
    entire schema state, allowing for efficient comparison and integrity verification.
    """
    
    def __init__(self, schema_graph: SchemaGraph):
        """
        Initialize the schema fingerprinter.
        
        Args:
            schema_graph: The schema graph to fingerprint
        """
        self.schema_graph = schema_graph
        self.last_computed_time = 0
        self.merkle_root = ""
        self.node_hashes = {}
        self.changed_nodes = set()
        self.changed_edges = set()
    
    def compute_node_hash(self, node_id: str) -> str:
        """
        Compute a hash for a single node.
        
        Args:
            node_id: ID of the node to hash
            
        Returns:
            SHA-256 hash of the node
        """
        if node_id not in self.schema_graph.graph:
            return ""
            
        # Get node data
        node_data = self.schema_graph.graph.nodes[node_id]
        
        # Create a deterministic representation
        node_dict = {
            "id": node_id,
            "label": node_data.get("label", ""),
            "type": node_data.get("node_type", "memory"),
            "importance": node_data.get("importance", 0.0),
            "memory_id": node_data.get("memory_id", None),
            "tags": sorted(node_data.get("tags", [])),
            # Include a stable subset of metadata
            "metadata": {
                k: v for k, v in node_data.get("metadata", {}).items()
                if k in ["memory_type", "source", "created_at", "is_reflection"]
            }
        }
        
        # Convert to JSON and hash
        node_json = json.dumps(node_dict, sort_keys=True)
        return hashlib.sha256(node_json.encode()).hexdigest()
    
    def compute_edge_hash(self, source_id: str, target_id: str) -> str:
        """
        Compute a hash for a single edge.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            SHA-256 hash of the edge
        """
        if not self.schema_graph.graph.has_edge(source_id, target_id):
            return ""
            
        # Get edge data
        edge_data = self.schema_graph.graph.edges[source_id, target_id]
        
        # Create a deterministic representation
        edge_dict = {
            "source": source_id,
            "target": target_id,
            "weight": edge_data.get("weight", 0.0),
            "type": edge_data.get("edge_type", "coherence"),
            # Include a stable subset of metadata
            "metadata": {
                k: v for k, v in edge_data.get("metadata", {}).items()
                if k in ["created_at", "last_updated"]
            }
        }
        
        # Convert to JSON and hash
        edge_json = json.dumps(edge_dict, sort_keys=True)
        return hashlib.sha256(edge_json.encode()).hexdigest()
    
    def compute_fingerprint(self) -> str:
        """
        Compute a Merkle tree-based fingerprint of the entire schema.
        
        Returns:
            Merkle root hash as a hexadecimal string
        """
        # Reset tracking of changes
        self.changed_nodes = set()
        self.changed_edges = set()
        
        # Compute hashes for all nodes
        node_hashes = {}
        for node_id in self.schema_graph.graph.nodes:
            node_hash = self.compute_node_hash(node_id)
            node_hashes[node_id] = node_hash
            
            # Check if node changed since last computation
            if node_id in self.node_hashes and node_hash != self.node_hashes[node_id]:
                self.changed_nodes.add(node_id)
        
        # Compute hashes for all edges
        edge_hashes = {}
        for source, target in self.schema_graph.graph.edges:
            edge_id = f"{source}-->{target}"
            edge_hash = self.compute_edge_hash(source, target)
            edge_hashes[edge_id] = edge_hash
            
            # Check if edge changed since last computation
            if edge_id in self.edge_hashes and edge_hash != self.edge_hashes[edge_id]:
                self.changed_edges.add((source, target))
        
        # Store the new hashes
        self.node_hashes = node_hashes
        self.edge_hashes = edge_hashes
        
        # Create a Merkle tree from the hashes
        merkle_root = self._compute_merkle_root(list(node_hashes.values()) + list(edge_hashes.values()))
        
        # Store the result
        self.merkle_root = merkle_root
        self.last_computed_time = time.time()
        
        return merkle_root
    
    def _compute_merkle_root(self, hash_list: List[str]) -> str:
        """
        Compute a Merkle root from a list of hashes.
        
        Args:
            hash_list: List of hashes to combine
            
        Returns:
            Merkle root hash
        """
        if not hash_list:
            # Empty schema
            return hashlib.sha256(b"empty_schema").hexdigest()
        
        if len(hash_list) == 1:
            # Only one hash, return it
            return hash_list[0]
        
        # Sort hashes for deterministic results
        sorted_hashes = sorted(hash_list)
        
        # Pair up hashes and compute parent hashes
        parent_hashes = []
        
        for i in range(0, len(sorted_hashes), 2):
            if i + 1 < len(sorted_hashes):
                # Concatenate two hashes and hash them
                combined = sorted_hashes[i] + sorted_hashes[i + 1]
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
            else:
                # Odd number of hashes, duplicate the last one
                combined = sorted_hashes[i] + sorted_hashes[i]
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
            
            parent_hashes.append(parent_hash)
        
        # Recursively compute the Merkle root
        return self._compute_merkle_root(parent_hashes)
    
    def get_changes_since_last_computation(self) -> Dict[str, Any]:
        """
        Get a summary of changes since the last fingerprint computation.
        
        Returns:
            Dictionary with change information
        """
        return {
            "changed_nodes": list(self.changed_nodes),
            "changed_edges": list(self.changed_edges),
            "num_changed_nodes": len(self.changed_nodes),
            "num_changed_edges": len(self.changed_edges)
        }


class SchemaDiffCalculator:
    """
    Calculates the differences between two schemas.
    
    This class enables detailed change detection between schema versions,
    supporting schema version control and drift analysis.
    """
    
    def __init__(
        self,
        current_schema: SchemaGraph,
        previous_schema: Optional[SchemaGraph] = None
    ):
        """
        Initialize the schema diff calculator.
        
        Args:
            current_schema: Current schema graph
            previous_schema: Previous schema graph to compare against
        """
        self.current_schema = current_schema
        self.previous_schema = previous_schema
        
        # Cached diff results
        self.last_diff_time = 0
        self.diff_results = {}
    
    def set_previous_schema(self, schema: SchemaGraph) -> None:
        """
        Set the previous schema for comparison.
        
        Args:
            schema: Schema graph to compare against
        """
        self.previous_schema = schema
        self.last_diff_time = 0  # Reset cache
    
    def calculate_diff(self) -> Dict[str, Any]:
        """
        Calculate the differences between current and previous schema.
        
        Returns:
            Dictionary with diff information
        """
        if not self.previous_schema:
            return {
                "error": "No previous schema set for comparison",
                "timestamp": time.time()
            }
        
        # Calculate node differences
        added_nodes = []
        removed_nodes = []
        modified_nodes = []
        
        # Current schema nodes
        current_nodes = set(self.current_schema.graph.nodes)
        
        # Previous schema nodes
        previous_nodes = set(self.previous_schema.graph.nodes)
        
        # Find added and removed nodes
        added_nodes = list(current_nodes - previous_nodes)
        removed_nodes = list(previous_nodes - current_nodes)
        
        # Find modified nodes
        common_nodes = current_nodes.intersection(previous_nodes)
        for node_id in common_nodes:
            current_data = self.current_schema.graph.nodes[node_id]
            previous_data = self.previous_schema.graph.nodes[node_id]
            
            # Check for changes in node attributes
            if self._is_node_modified(current_data, previous_data):
                modified_nodes.append(node_id)
        
        # Calculate edge differences
        added_edges = []
        removed_edges = []
        modified_edges = []
        
        # Current schema edges
        current_edges = set(self.current_schema.graph.edges)
        
        # Previous schema edges
        previous_edges = set(self.previous_schema.graph.edges)
        
        # Find added and removed edges
        added_edges = list(current_edges - previous_edges)
        removed_edges = list(previous_edges - current_edges)
        
        # Find modified edges
        common_edges = current_edges.intersection(previous_edges)
        for edge in common_edges:
            source, target = edge
            current_data = self.current_schema.graph.edges[source, target]
            previous_data = self.previous_schema.graph.edges[source, target]
            
            # Check for changes in edge attributes
            if self._is_edge_modified(current_data, previous_data):
                modified_edges.append(edge)
        
        # Calculate clustering and centrality changes
        cluster_changes = self._calculate_cluster_changes()
        
        # Calculate overall schema metrics changes
        metric_changes = self._calculate_metric_changes()
        
        # Compute fingerprints
        current_fingerprinter = SchemaFingerprint(self.current_schema)
        previous_fingerprinter = SchemaFingerprint(self.previous_schema)
        
        current_hash = current_fingerprinter.compute_fingerprint()
        previous_hash = previous_fingerprinter.compute_fingerprint()
        
        # Store diff results
        diff_results = {
            "timestamp": time.time(),
            "current_schema_hash": current_hash,
            "previous_schema_hash": previous_hash,
            "schemas_match": current_hash == previous_hash,
            "nodes": {
                "added": added_nodes,
                "removed": removed_nodes,
                "modified": modified_nodes,
                "added_count": len(added_nodes),
                "removed_count": len(removed_nodes),
                "modified_count": len(modified_nodes)
            },
            "edges": {
                "added": added_edges,
                "removed": removed_edges,
                "modified": modified_edges,
                "added_count": len(added_edges),
                "removed_count": len(removed_edges),
                "modified_count": len(modified_edges)
            },
            "clusters": cluster_changes,
            "metrics": metric_changes
        }
        
        # Cache results
        self.diff_results = diff_results
        self.last_diff_time = time.time()
        
        return diff_results
    
    def _is_node_modified(self, current_data: Dict, previous_data: Dict) -> bool:
        """
        Check if a node has been modified.
        
        Args:
            current_data: Current node data
            previous_data: Previous node data
            
        Returns:
            True if the node has been modified, False otherwise
        """
        # Key attributes to check
        attrs_to_check = ["label", "node_type", "importance", "tags"]
        
        for attr in attrs_to_check:
            current_value = current_data.get(attr)
            previous_value = previous_data.get(attr)
            
            # Special case for importance - use a threshold for floating point comparison
            if attr == "importance":
                if abs((current_value or 0) - (previous_value or 0)) > 0.01:
                    return True
            # Special case for tags - compare as sets
            elif attr == "tags":
                if set(current_value or []) != set(previous_value or []):
                    return True
            # Default comparison
            elif current_value != previous_value:
                return True
        
        return False
    
    def _is_edge_modified(self, current_data: Dict, previous_data: Dict) -> bool:
        """
        Check if an edge has been modified.
        
        Args:
            current_data: Current edge data
            previous_data: Previous edge data
            
        Returns:
            True if the edge has been modified, False otherwise
        """
        # Key attributes to check
        attrs_to_check = ["weight", "edge_type"]
        
        for attr in attrs_to_check:
            current_value = current_data.get(attr)
            previous_value = previous_data.get(attr)
            
            # Special case for weight - use a threshold for floating point comparison
            if attr == "weight":
                if abs((current_value or 0) - (previous_value or 0)) > 0.01:
                    return True
            # Default comparison
            elif current_value != previous_value:
                return True
        
        return False
    
    def _calculate_cluster_changes(self) -> Dict[str, Any]:
        """
        Calculate changes in schema clustering.
        
        Returns:
            Dictionary with cluster change information
        """
        import networkx as nx
        
        try:
            # Calculate clustering coefficients
            current_clustering = nx.average_clustering(self.current_schema.graph)
            previous_clustering = nx.average_clustering(self.previous_schema.graph)
            
            # Calculate connected components
            current_components = list(nx.connected_components(self.current_schema.graph))
            previous_components = list(nx.connected_components(self.previous_schema.graph))
            
            return {
                "clustering_coefficient": {
                    "current": current_clustering,
                    "previous": previous_clustering,
                    "change": current_clustering - previous_clustering
                },
                "connected_components": {
                    "current": len(current_components),
                    "previous": len(previous_components),
                    "change": len(current_components) - len(previous_components)
                }
            }
        except Exception as e:
            return {
                "error": f"Error calculating cluster changes: {str(e)}"
            }
    
    def _calculate_metric_changes(self) -> Dict[str, Any]:
        """
        Calculate changes in overall schema metrics.
        
        Returns:
            Dictionary with metric change information
        """
        # Node and edge counts
        current_nodes = len(self.current_schema.graph.nodes)
        previous_nodes = len(self.previous_schema.graph.nodes)
        
        current_edges = len(self.current_schema.graph.edges)
        previous_edges = len(self.previous_schema.graph.edges)
        
        # Node type distribution
        current_types = defaultdict(int)
        previous_types = defaultdict(int)
        
        for _, data in self.current_schema.graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            current_types[node_type] += 1
            
        for _, data in self.previous_schema.graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            previous_types[node_type] += 1
        
        # Calculate average importance
        current_importance = 0
        previous_importance = 0
        
        if current_nodes > 0:
            for _, data in self.current_schema.graph.nodes(data=True):
                current_importance += data.get("importance", 0)
            current_importance /= current_nodes
            
        if previous_nodes > 0:
            for _, data in self.previous_schema.graph.nodes(data=True):
                previous_importance += data.get("importance", 0)
            previous_importance /= previous_nodes
        
        return {
            "node_count": {
                "current": current_nodes,
                "previous": previous_nodes,
                "change": current_nodes - previous_nodes,
                "percent_change": ((current_nodes - previous_nodes) / max(1, previous_nodes)) * 100
            },
            "edge_count": {
                "current": current_edges,
                "previous": previous_edges,
                "change": current_edges - previous_edges,
                "percent_change": ((current_edges - previous_edges) / max(1, previous_edges)) * 100
            },
            "avg_importance": {
                "current": current_importance,
                "previous": previous_importance,
                "change": current_importance - previous_importance
            },
            "node_types": {
                "current": dict(current_types),
                "previous": dict(previous_types)
            }
        }
    
    def get_detailed_node_changes(self, node_id: str) -> Dict[str, Any]:
        """
        Get detailed information about changes to a specific node.
        
        Args:
            node_id: ID of the node to examine
            
        Returns:
            Dictionary with detailed change information
        """
        if not self.previous_schema:
            return {
                "error": "No previous schema set for comparison"
            }
            
        # Check if node exists in both schemas
        current_exists = node_id in self.current_schema.graph
        previous_exists = node_id in self.previous_schema.graph
        
        if not current_exists and not previous_exists:
            return {
                "error": f"Node {node_id} does not exist in either schema"
            }
            
        # Node was added
        if current_exists and not previous_exists:
            return {
                "status": "added",
                "current_data": dict(self.current_schema.graph.nodes[node_id])
            }
            
        # Node was removed
        if not current_exists and previous_exists:
            return {
                "status": "removed",
                "previous_data": dict(self.previous_schema.graph.nodes[node_id])
            }
            
        # Node exists in both - calculate detailed differences
        current_data = dict(self.current_schema.graph.nodes[node_id])
        previous_data = dict(self.previous_schema.graph.nodes[node_id])
        
        # Collect attribute changes
        changes = {}
        
        for key in set(list(current_data.keys()) + list(previous_data.keys())):
            current_value = current_data.get(key)
            previous_value = previous_data.get(key)
            
            if current_value != previous_value:
                changes[key] = {
                    "current": current_value,
                    "previous": previous_value
                }
        
        # Get edge changes for this node
        current_edges = list(self.current_schema.graph.edges(node_id, data=True))
        previous_edges = list(self.previous_schema.graph.edges(node_id, data=True))
        
        current_edge_nodes = {edge[1] for edge in current_edges}
        previous_edge_nodes = {edge[1] for edge in previous_edges}
        
        added_connections = current_edge_nodes - previous_edge_nodes
        removed_connections = previous_edge_nodes - current_edge_nodes
        
        return {
            "status": "modified" if changes else "unchanged",
            "attribute_changes": changes,
            "connections": {
                "added": list(added_connections),
                "removed": list(removed_connections)
            },
            "current_data": current_data,
            "previous_data": previous_data
        }
    
    def visualize_diff(self, filename: Optional[str] = None, show: bool = True) -> None:
        """
        Visualize the differences between schemas.
        
        Args:
            filename: Optional path to save the visualization
            show: Whether to display the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            if not self.previous_schema:
                raise ValueError("No previous schema set for comparison")
                
            # Calculate diff if not already done
            if not self.diff_results:
                self.calculate_diff()
                
            # Create a new graph for visualization
            diff_graph = nx.Graph()
            
            # Add all nodes from both schemas
            all_nodes = set(list(self.current_schema.graph.nodes) + 
                           list(self.previous_schema.graph.nodes))
            
            # Add nodes with status
            for node_id in all_nodes:
                current_exists = node_id in self.current_schema.graph
                previous_exists = node_id in self.previous_schema.graph
                
                if current_exists and previous_exists:
                    # Node exists in both schemas
                    status = "unchanged"
                    if node_id in self.diff_results.get("nodes", {}).get("modified", []):
                        status = "modified"
                    label = self.current_schema.graph.nodes[node_id].get("label", node_id)
                elif current_exists:
                    # Node was added
                    status = "added"
                    label = self.current_schema.graph.nodes[node_id].get("label", node_id)
                else:
                    # Node was removed
                    status = "removed"
                    label = self.previous_schema.graph.nodes[node_id].get("label", node_id)
                    
                diff_graph.add_node(node_id, status=status, label=label)
            
            # Add all edges from both schemas
            all_edges = set(list(self.current_schema.graph.edges) + 
                           list(self.previous_schema.graph.edges))
            
            # Add edges with status
            for source, target in all_edges:
                current_exists = self.current_schema.graph.has_edge(source, target)
                previous_exists = self.previous_schema.graph.has_edge(source, target)
                
                if current_exists and previous_exists:
                    # Edge exists in both schemas
                    status = "unchanged"
                    if (source, target) in self.diff_results.get("edges", {}).get("modified", []):
                        status = "modified"
                    weight = self.current_schema.graph.edges[source, target].get("weight", 0.1)
                elif current_exists:
                    # Edge was added
                    status = "added"
                    weight = self.current_schema.graph.edges[source, target].get("weight", 0.1)
                else:
                    # Edge was removed
                    status = "removed"
                    weight = self.previous_schema.graph.edges[source, target].get("weight", 0.1)
                    
                diff_graph.add_edge(source, target, status=status, weight=weight)
            
            # Create figure
            plt.figure(figsize=(14, 10))
            
            # Create layout
            layout = nx.spring_layout(diff_graph, seed=42)
            
            # Draw nodes based on status
            node_colors = []
            for node, data in diff_graph.nodes(data=True):
                status = data.get("status", "unchanged")
                if status == "added":
                    node_colors.append("green")
                elif status == "removed":
                    node_colors.append("red")
                elif status == "modified":
                    node_colors.append("orange")
                else:
                    node_colors.append("gray")
                    
            nx.draw_networkx_nodes(
                diff_graph,
                layout,
                node_color=node_colors,
                alpha=0.8
            )
            
            # Draw edges based on status
            edge_colors = {"added": "green", "removed": "red", "modified": "orange", "unchanged": "gray"}
            
            for status in edge_colors:
                edge_list = [(u, v) for u, v, d in diff_graph.edges(data=True) if d.get("status") == status]
                
                if edge_list:
                    edge_weights = [diff_graph.edges[u, v].get("weight", 0.1) * 2 for u, v in edge_list]
                    
                    nx.draw_networkx_edges(
                        diff_graph,
                        layout,
                        edgelist=edge_list,
                        width=edge_weights,
                        alpha=0.6,
                        edge_color=edge_colors[status]
                    )
            
            # Draw labels
            labels = {node: data.get("label", "")[:15] + ("..." if len(data.get("label", "")) > 15 else "") 
                     for node, data in diff_graph.nodes(data=True)}
            
            nx.draw_networkx_labels(
                diff_graph,
                layout,
                labels=labels,
                font_size=8
            )
            
            # Add legend and title
            plt.title("Schema Diff Visualization")
            
            # Add legend patches
            import matplotlib.patches as mpatches
            
            legend_patches = [
                mpatches.Patch(color="green", label="Added"),
                mpatches.Patch(color="red", label="Removed"),
                mpatches.Patch(color="orange", label="Modified"),
                mpatches.Patch(color="gray", label="Unchanged")
            ]
            
            plt.legend(handles=legend_patches, loc="upper right")
            
            # Add summary text
            summary_text = (
                f"Added nodes: {len(self.diff_results.get('nodes', {}).get('added', []))}\n"
                f"Removed nodes: {len(self.diff_results.get('nodes', {}).get('removed', []))}\n"
                f"Modified nodes: {len(self.diff_results.get('nodes', {}).get('modified', []))}\n"
                f"Added edges: {len(self.diff_results.get('edges', {}).get('added', []))}\n"
                f"Removed edges: {len(self.diff_results.get('edges', {}).get('removed', []))}\n"
                f"Modified edges: {len(self.diff_results.get('edges', {}).get('modified', []))}\n"
            )
            
            plt.figtext(0.02, 0.02, summary_text, fontsize=10)
            
            plt.axis("off")
            
            # Save or show
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                
            if show:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error visualizing diff: {e}")


class SchemaDriftMonitor:
    """
    Monitors schema drift over time.
    
    Tracks changes in the schema graph structure and provides
    alerts when significant changes occur.
    """
    
    def __init__(
        self,
        schema_graph: SchemaGraph,
        drift_threshold: float = 0.1,
        max_history: int = 10
    ):
        """
        Initialize the schema drift monitor.
        
        Args:
            schema_graph: The schema graph to monitor
            drift_threshold: Threshold for significant drift (0-1)
            max_history: Maximum number of schema snapshots to keep
        """
        self.schema_graph = schema_graph
        self.drift_threshold = drift_threshold
        self.max_history = max_history
        
        # Schema snapshot history
        self.snapshots = []
        self.merkle_history = []
        self.drift_scores = []
    
    def take_snapshot(self) -> Dict[str, Any]:
        """
        Take a snapshot of the current schema state.
        
        Returns:
            Dictionary with snapshot information
        """
        # Create a fingerprint of the current schema
        fingerprinter = SchemaFingerprint(self.schema_graph)
        merkle_root = fingerprinter.compute_fingerprint()
        
        # Get schema statistics
        stats = self.schema_graph.get_stats()
        
        # Create snapshot
        snapshot = {
            "timestamp": time.time(),
            "merkle_root": merkle_root,
            "node_count": stats.get("node_count", 0),
            "edge_count": stats.get("edge_count", 0),
            "avg_importance": stats.get("avg_importance", 0),
            "clustering": stats.get("clustering_coefficient", 0),
            "components": stats.get("connected_components", 0)
        }
        
        # Add to history
        self.snapshots.append(snapshot)
        self.merkle_history.append(merkle_root)
        
        # Trim history if needed
        if len(self.snapshots) > self.max_history:
            self.snapshots = self.snapshots[-self.max_history:]
            self.merkle_history = self.merkle_history[-self.max_history:]
            self.drift_scores = self.drift_scores[-self.max_history:]
        
        # Calculate drift if we have previous snapshots
        if len(self.snapshots) > 1:
            drift_score = self._calculate_drift(self.snapshots[-2], self.snapshots[-1])
            self.drift_scores.append(drift_score)
            snapshot["drift_score"] = drift_score
            snapshot["significant_drift"] = drift_score > self.drift_threshold
        else:
            snapshot["drift_score"] = 0
            snapshot["significant_drift"] = False
        
        return snapshot
    
    def _calculate_drift(self, previous: Dict, current: Dict) -> float:
        """
        Calculate schema drift score between two snapshots.
        
        Args:
            previous: Previous snapshot
            current: Current snapshot
            
        Returns:
            Drift score (0-1)
        """
        # Check if Merkle roots are the same
        if previous["merkle_root"] == current["merkle_root"]:
            return 0
        
        # Calculate weighted components of drift
        drift_components = {}
        
        # Node count change
        if previous["node_count"] > 0:
            node_ratio = abs(current["node_count"] - previous["node_count"]) / previous["node_count"]
            drift_components["node_count"] = min(1.0, node_ratio)
        else:
            drift_components["node_count"] = 1.0 if current["node_count"] > 0 else 0.0
        
        # Edge count change
        if previous["edge_count"] > 0:
            edge_ratio = abs(current["edge_count"] - previous["edge_count"]) / previous["edge_count"]
            drift_components["edge_count"] = min(1.0, edge_ratio)
        else:
            drift_components["edge_count"] = 1.0 if current["edge_count"] > 0 else 0.0
        
        # Clustering change
        if previous["clustering"] > 0:
            cluster_ratio = abs(current["clustering"] - previous["clustering"]) / previous["clustering"]
            drift_components["clustering"] = min(1.0, cluster_ratio)
        else:
            drift_components["clustering"] = 0.2 if current["clustering"] > 0 else 0.0
        
        # Component count change
        if previous["components"] > 0:
            component_ratio = abs(current["components"] - previous["components"]) / previous["components"]
            drift_components["components"] = min(1.0, component_ratio)
        else:
            drift_components["components"] = 0.2 if current["components"] > 0 else 0.0
        
        # Weighted average of drift components
        weights = {
            "node_count": 0.3,
            "edge_count": 0.3,
            "clustering": 0.2,
            "components": 0.2
        }
        
        drift_score = sum(drift_components[k] * weights[k] for k in weights)
        
        return drift_score
    
    def get_drift_report(self) -> Dict[str, Any]:
        """
        Get a report on schema drift.
        
        Returns:
            Dictionary with drift information
        """
        if len(self.snapshots) < 2:
            return {
                "error": "Insufficient history for drift analysis",
                "snapshot_count": len(self.snapshots)
            }
        
        # Calculate drift statistics
        recent_drift = self.drift_scores[-1] if self.drift_scores else 0
        max_drift = max(self.drift_scores) if self.drift_scores else 0
        avg_drift = sum(self.drift_scores) / len(self.drift_scores) if self.drift_scores else 0
        
        # Calculate drift trend
        trend = "stable"
        if len(self.drift_scores) >= 3:
            if self.drift_scores[-1] > self.drift_scores[-2] > self.drift_scores[-3]:
                trend = "increasing"
            elif self.drift_scores[-1] < self.drift_scores[-2] < self.drift_scores[-3]:
                trend = "decreasing"
        
        # Get significant drift events
        significant_drifts = []
        for i, score in enumerate(self.drift_scores):
            if score > self.drift_threshold:
                if i < len(self.snapshots) - 1:
                    significant_drifts.append({
                        "index": i,
                        "score": score,
                        "from_timestamp": self.snapshots[i]["timestamp"],
                        "to_timestamp": self.snapshots[i+1]["timestamp"]
                    })
        
        return {
            "current_drift": recent_drift,
            "max_drift": max_drift,
            "avg_drift": avg_drift,
            "trend": trend,
            "significant_drifts": significant_drifts,
            "snapshot_count": len(self.snapshots),
            "last_snapshot": self.snapshots[-1],
            "drift_threshold": self.drift_threshold
        }
