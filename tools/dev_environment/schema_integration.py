#!/usr/bin/env python3
"""
Schema Integration Module for ΨC-AI SDK Development Environment

This module provides integration between the Memory Sandbox and schema-based knowledge
representation systems. It enables building schema graphs from memory stores,
analyzing conceptual relationships, and tracking concept drift over time.

Features:
- Build schema graphs from memory content
- Detect clusters of related memories
- Analyze relationships between concepts
- Track concept drift over time
- Generate knowledge summaries
- Visualize knowledge structures
"""

import os
import json
import uuid
import datetime
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

class SchemaGraph:
    """
    Represents a schema graph with concepts as nodes and relationships as edges.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.metadata = {
            "creation_date": datetime.datetime.now().isoformat(),
            "version": "1.0",
            "num_concepts": 0,
            "num_relationships": 0
        }
    
    def add_concept(self, concept_id: str, properties: Dict[str, Any]) -> None:
        """Add a concept node to the schema graph."""
        self.graph.add_node(concept_id, **properties)
        self.metadata["num_concepts"] = len(self.graph.nodes)
    
    def add_relationship(self, source_id: str, target_id: str, 
                        rel_type: str, weight: float, 
                        properties: Dict[str, Any] = None) -> None:
        """Add a relationship edge to the schema graph."""
        if properties is None:
            properties = {}
        
        self.graph.add_edge(source_id, target_id, 
                           type=rel_type, 
                           weight=weight, 
                           **properties)
        self.metadata["num_relationships"] = len(self.graph.edges)
    
    def get_central_concepts(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Return the top N central concepts based on centrality measures."""
        centrality = nx.eigenvector_centrality(self.graph, weight='weight')
        return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_concept_clusters(self, resolution: float = 1.0) -> Dict[int, List[str]]:
        """Identify clusters of related concepts using community detection."""
        communities = nx.community.louvain_communities(
            self.graph.to_undirected(), weight='weight', resolution=resolution
        )
        return {i: list(community) for i, community in enumerate(communities)}
    
    def visualize(self, output_path: str = None) -> plt.Figure:
        """Visualize the schema graph."""
        plt.figure(figsize=(12, 10))
        
        # Calculate node size based on centrality
        centrality = nx.eigenvector_centrality(self.graph, weight='weight')
        node_size = [centrality[n] * 5000 for n in self.graph.nodes()]
        
        # Calculate edge width based on weight
        edge_width = [self.graph[u][v]['weight'] * 2 for u, v in self.graph.edges()]
        
        # Generate layout
        pos = nx.spring_layout(self.graph, k=0.3, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size, 
                               node_color='skyblue', alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=edge_width, 
                              alpha=0.5, edge_color='darkblue', 
                              arrows=True, arrowsize=15)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
        
        plt.axis('off')
        plt.title(f"Schema Graph - {len(self.graph.nodes)} concepts, {len(self.graph.edges)} relationships")
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the schema graph to a dictionary for serialization."""
        return {
            "metadata": self.metadata,
            "nodes": [
                {
                    "id": node,
                    "properties": {k: v for k, v in self.graph.nodes[node].items()}
                }
                for node in self.graph.nodes
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "type": self.graph[u][v].get("type", ""),
                    "weight": self.graph[u][v].get("weight", 1.0),
                    "properties": {
                        k: v for k, v in self.graph[u][v].items() 
                        if k not in ["type", "weight"]
                    }
                }
                for u, v in self.graph.edges
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaGraph':
        """Create a schema graph from a dictionary representation."""
        schema = cls()
        schema.metadata = data.get("metadata", {})
        
        # Add nodes
        for node_data in data.get("nodes", []):
            node_id = node_data["id"]
            properties = node_data.get("properties", {})
            schema.graph.add_node(node_id, **properties)
        
        # Add edges
        for edge_data in data.get("edges", []):
            source = edge_data["source"]
            target = edge_data["target"]
            rel_type = edge_data.get("type", "")
            weight = edge_data.get("weight", 1.0)
            properties = edge_data.get("properties", {})
            
            schema.graph.add_edge(source, target, type=rel_type, weight=weight, **properties)
        
        schema.metadata["num_concepts"] = len(schema.graph.nodes)
        schema.metadata["num_relationships"] = len(schema.graph.edges)
        
        return schema


class SchemaIntegration:
    """
    Integrates schema-based knowledge representation with the Memory Sandbox.
    """
    def __init__(self, memory_sandbox):
        self.memory_sandbox = memory_sandbox
        self.snapshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         "schema_snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self.schema_snapshots = {}
    
    def build_schema_from_memory_store(self, memory_store_id: Optional[str] = None, 
                                      concept_threshold: float = 0.3,
                                      relation_threshold: float = 0.2) -> SchemaGraph:
        """
        Build a schema graph from the memories in a memory store.
        
        Args:
            memory_store_id: ID of the memory store to use, or None for current store
            concept_threshold: Threshold for concept formation
            relation_threshold: Threshold for relationship formation
            
        Returns:
            SchemaGraph: The constructed schema graph
        """
        # Get the memory store
        if memory_store_id:
            memory_store = self.memory_sandbox.get_memory_store(memory_store_id)
        else:
            memory_store = self.memory_sandbox.memory_store
        
        # Extract memories
        memories = memory_store.get_all_memories()
        if not memories:
            return SchemaGraph()
        
        # Create a new schema graph
        schema = SchemaGraph()
        
        # Build concept vectors from semantic memories
        semantic_memories = [m for m in memories if m.memory_type == "semantic"]
        
        # Extract key concepts from semantic memories
        concepts = {}
        for memory in semantic_memories:
            concept_name = memory.content.get("concept", "")
            if not concept_name:
                continue
                
            if concept_name not in concepts:
                concepts[concept_name] = {
                    "importance": memory.importance,
                    "count": 1,
                    "memories": [memory],
                    "attributes": {},
                    "vector": memory.embedding if hasattr(memory, "embedding") else None
                }
            else:
                concepts[concept_name]["count"] += 1
                concepts[concept_name]["importance"] += memory.importance
                concepts[concept_name]["memories"].append(memory)
                
                # Average embeddings if available
                if hasattr(memory, "embedding") and memory.embedding is not None:
                    if concepts[concept_name]["vector"] is not None:
                        concepts[concept_name]["vector"] = np.mean(
                            [concepts[concept_name]["vector"], memory.embedding], axis=0
                        )
                    else:
                        concepts[concept_name]["vector"] = memory.embedding
                
            # Extract attributes
            for key, value in memory.content.items():
                if key != "concept" and isinstance(value, (str, int, float, bool)):
                    if key not in concepts[concept_name]["attributes"]:
                        concepts[concept_name]["attributes"][key] = [value]
                    else:
                        concepts[concept_name]["attributes"][key].append(value)
        
        # Filter concepts based on importance
        important_concepts = {
            name: data for name, data in concepts.items() 
            if data["importance"] >= concept_threshold
        }
        
        # Add concept nodes to schema
        for concept_name, concept_data in important_concepts.items():
            # Consolidate attributes
            consolidated_attributes = {}
            for attr_name, attr_values in concept_data["attributes"].items():
                # For simplicity, use the most common value
                from collections import Counter
                value_counts = Counter(attr_values)
                consolidated_attributes[attr_name] = value_counts.most_common(1)[0][0]
            
            schema.add_concept(concept_name, {
                "importance": concept_data["importance"],
                "memory_count": concept_data["count"],
                "attributes": consolidated_attributes
            })
        
        # Build relationships between concepts
        for source_name in important_concepts:
            source_vector = important_concepts[source_name]["vector"]
            if source_vector is None:
                continue
                
            for target_name in important_concepts:
                if source_name == target_name:
                    continue
                    
                target_vector = important_concepts[target_name]["vector"]
                if target_vector is None:
                    continue
                
                # Calculate similarity
                similarity = self._cosine_similarity(source_vector, target_vector)
                
                # Add relationship if above threshold
                if similarity >= relation_threshold:
                    schema.add_relationship(
                        source_name, target_name, 
                        "semantic_similarity", 
                        similarity,
                        {"description": f"Semantic similarity: {similarity:.2f}"}
                    )
        
        # Analyze episodic memories for temporal relationships
        self._analyze_episodic_memories(schema, memories, important_concepts)
        
        return schema
    
    def _analyze_episodic_memories(self, schema: SchemaGraph, 
                                  memories: List[Any],
                                  concepts: Dict[str, Any]) -> None:
        """Analyze episodic memories to find temporal relationships between concepts."""
        episodic_memories = [m for m in memories if m.memory_type == "episodic"]
        episodic_memories.sort(key=lambda m: m.timestamp)
        
        # Extract concept mentions in episodic memories
        concept_mentions = {}
        for memory in episodic_memories:
            content_str = str(memory.content).lower()
            mentioned_concepts = []
            
            for concept_name in concepts:
                if concept_name.lower() in content_str:
                    mentioned_concepts.append(concept_name)
            
            if mentioned_concepts:
                concept_mentions[memory.id] = {
                    "concepts": mentioned_concepts,
                    "timestamp": memory.timestamp
                }
        
        # Find temporal relationships (concepts that appear in sequence)
        timestamps = sorted([m["timestamp"] for m in concept_mentions.values()])
        if not timestamps:
            return
            
        # Establish a time window (using average distance between timestamps)
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                         for i in range(len(timestamps)-1)]
            avg_diff = sum(time_diffs) / len(time_diffs)
            time_window = avg_diff * 2  # Double the average time difference
        else:
            time_window = 3600  # Default: 1 hour
        
        # Analyze temporal co-occurrence
        concept_pairs = {}
        sorted_memories = sorted(concept_mentions.values(), key=lambda x: x["timestamp"])
        
        for i, memory_data in enumerate(sorted_memories):
            for concept1 in memory_data["concepts"]:
                # Look ahead for temporally related concepts
                j = i + 1
                while (j < len(sorted_memories) and 
                      (sorted_memories[j]["timestamp"] - memory_data["timestamp"]).total_seconds() <= time_window):
                    
                    for concept2 in sorted_memories[j]["concepts"]:
                        if concept1 != concept2:
                            pair_key = (concept1, concept2)
                            if pair_key not in concept_pairs:
                                concept_pairs[pair_key] = 0
                            concept_pairs[pair_key] += 1
                    j += 1
        
        # Add temporal relationships to schema
        for (concept1, concept2), count in concept_pairs.items():
            if concept1 in schema.graph.nodes and concept2 in schema.graph.nodes:
                schema.add_relationship(
                    concept1, concept2,
                    "temporal_sequence",
                    min(1.0, count / 5),  # Normalize weight, max at 1.0
                    {"sequence_count": count}
                )
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
            
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def add_schema_snapshot(self, schema: SchemaGraph) -> str:
        """
        Save a snapshot of the current schema graph.
        
        Args:
            schema: The schema graph to snapshot
            
        Returns:
            str: ID of the saved snapshot
        """
        snapshot_id = str(uuid.uuid4())
        snapshot_data = {
            "id": snapshot_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "schema": schema.to_dict()
        }
        
        # Save to file
        snapshot_path = os.path.join(self.snapshots_dir, f"{snapshot_id}.json")
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot_data, f)
        
        # Store in memory
        self.schema_snapshots[snapshot_id] = snapshot_data
        
        return snapshot_id
    
    def get_schema_snapshots(self) -> List[Dict[str, Any]]:
        """
        Get a list of all schema snapshots.
        
        Returns:
            List of snapshot metadata (id, timestamp)
        """
        # Load any snapshots from disk that aren't in memory
        for filename in os.listdir(self.snapshots_dir):
            if filename.endswith('.json'):
                snapshot_id = filename.replace('.json', '')
                if snapshot_id not in self.schema_snapshots:
                    snapshot_path = os.path.join(self.snapshots_dir, filename)
                    try:
                        with open(snapshot_path, 'r') as f:
                            self.schema_snapshots[snapshot_id] = json.load(f)
                    except:
                        pass
        
        # Return metadata only
        return [
            {
                "id": snapshot_id,
                "timestamp": snapshot_data["timestamp"],
                "num_concepts": snapshot_data["schema"]["metadata"]["num_concepts"],
                "num_relationships": snapshot_data["schema"]["metadata"]["num_relationships"]
            }
            for snapshot_id, snapshot_data in self.schema_snapshots.items()
        ]
    
    def get_schema_snapshot(self, snapshot_id: str) -> Optional[SchemaGraph]:
        """
        Get a schema snapshot by ID.
        
        Args:
            snapshot_id: ID of the snapshot to retrieve
            
        Returns:
            SchemaGraph or None if not found
        """
        if snapshot_id in self.schema_snapshots:
            schema_dict = self.schema_snapshots[snapshot_id]["schema"]
            return SchemaGraph.from_dict(schema_dict)
        
        # Try loading from disk
        snapshot_path = os.path.join(self.snapshots_dir, f"{snapshot_id}.json")
        if os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, 'r') as f:
                    snapshot_data = json.load(f)
                    self.schema_snapshots[snapshot_id] = snapshot_data
                    return SchemaGraph.from_dict(snapshot_data["schema"])
            except:
                pass
        
        return None
    
    def compare_schemas(self, snapshot_id1: str, snapshot_id2: str) -> Dict[str, Any]:
        """
        Compare two schema snapshots.
        
        Args:
            snapshot_id1: ID of the first snapshot
            snapshot_id2: ID of the second snapshot
            
        Returns:
            Dict with comparison results
        """
        schema1 = self.get_schema_snapshot(snapshot_id1)
        schema2 = self.get_schema_snapshot(snapshot_id2)
        
        if not schema1 or not schema2:
            return {"error": "One or both snapshots not found"}
        
        # Get sets of concepts
        concepts1 = set(schema1.graph.nodes)
        concepts2 = set(schema2.graph.nodes)
        
        # Find differences
        added_concepts = concepts2 - concepts1
        removed_concepts = concepts1 - concepts2
        common_concepts = concepts1 & concepts2
        
        # Compare relationships
        edges1 = set((u, v) for u, v in schema1.graph.edges)
        edges2 = set((u, v) for u, v in schema2.graph.edges)
        
        added_relationships = edges2 - edges1
        removed_relationships = edges1 - edges2
        
        # Compare concept attributes for common concepts
        changed_concepts = []
        for concept in common_concepts:
            attrs1 = schema1.graph.nodes[concept]
            attrs2 = schema2.graph.nodes[concept]
            
            # Compare importance
            if attrs1.get("importance", 0) != attrs2.get("importance", 0):
                changed_concepts.append({
                    "concept": concept,
                    "changes": {
                        "importance": {
                            "old": attrs1.get("importance", 0),
                            "new": attrs2.get("importance", 0)
                        }
                    }
                })
        
        return {
            "snapshot1": {
                "id": snapshot_id1,
                "timestamp": self.schema_snapshots[snapshot_id1]["timestamp"],
                "num_concepts": len(concepts1),
                "num_relationships": len(edges1)
            },
            "snapshot2": {
                "id": snapshot_id2,
                "timestamp": self.schema_snapshots[snapshot_id2]["timestamp"],
                "num_concepts": len(concepts2),
                "num_relationships": len(edges2)
            },
            "added_concepts": list(added_concepts),
            "removed_concepts": list(removed_concepts),
            "changed_concepts": changed_concepts,
            "added_relationships": [{"source": u, "target": v} for u, v in added_relationships],
            "removed_relationships": [{"source": u, "target": v} for u, v in removed_relationships],
            "concept_overlap_percentage": 
                len(common_concepts) / max(1, len(concepts1.union(concepts2))) * 100
        }
    
    def analyze_concept_drift(self, snapshot_id1: str, snapshot_id2: str) -> Dict[str, Any]:
        """
        Analyze concept drift between two schema snapshots.
        
        Args:
            snapshot_id1: ID of the first snapshot
            snapshot_id2: ID of the second snapshot
            
        Returns:
            Dict with drift analysis results
        """
        schema1 = self.get_schema_snapshot(snapshot_id1)
        schema2 = self.get_schema_snapshot(snapshot_id2)
        
        if not schema1 or not schema2:
            return {"error": "One or both snapshots not found"}
        
        # Get common concepts
        common_concepts = set(schema1.graph.nodes).intersection(set(schema2.graph.nodes))
        
        # Compute centrality for each schema
        centrality1 = nx.eigenvector_centrality(schema1.graph, weight='weight', max_iter=1000)
        centrality2 = nx.eigenvector_centrality(schema2.graph, weight='weight', max_iter=1000)
        
        # Identify concepts with significant centrality changes
        centrality_changes = {}
        for concept in common_concepts:
            value1 = centrality1.get(concept, 0)
            value2 = centrality2.get(concept, 0)
            change = value2 - value1
            
            if abs(change) > 0.05:  # Only track significant changes
                centrality_changes[concept] = change
        
        # Compute community structures
        communities1 = nx.community.louvain_communities(
            schema1.graph.to_undirected(), weight='weight'
        )
        communities2 = nx.community.louvain_communities(
            schema2.graph.to_undirected(), weight='weight'
        )
        
        # Track concept community membership
        community_mapping1 = {}
        for i, community in enumerate(communities1):
            for concept in community:
                community_mapping1[concept] = i
                
        community_mapping2 = {}
        for i, community in enumerate(communities2):
            for concept in community:
                community_mapping2[concept] = i
        
        # Identify concepts that changed communities
        community_changes = {}
        for concept in common_concepts:
            if concept in community_mapping1 and concept in community_mapping2:
                if community_mapping1[concept] != community_mapping2[concept]:
                    community_changes[concept] = {
                        "from": community_mapping1[concept],
                        "to": community_mapping2[concept]
                    }
        
        return {
            "snapshot1": {
                "id": snapshot_id1,
                "timestamp": self.schema_snapshots[snapshot_id1]["timestamp"]
            },
            "snapshot2": {
                "id": snapshot_id2,
                "timestamp": self.schema_snapshots[snapshot_id2]["timestamp"]
            },
            "common_concepts_count": len(common_concepts),
            "centrality_changes": [
                {"concept": concept, "change": change} 
                for concept, change in sorted(
                    centrality_changes.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
            ],
            "community_changes": [
                {
                    "concept": concept,
                    "from_community": data["from"],
                    "to_community": data["to"]
                }
                for concept, data in community_changes.items()
            ],
            "drift_magnitude": sum(abs(change) for change in centrality_changes.values())
        }
    
    def identify_pivot_concepts(self, snapshot_id: str) -> List[Dict[str, Any]]:
        """
        Identify pivot concepts in a schema - concepts that connect different clusters.
        
        Args:
            snapshot_id: ID of the schema snapshot
            
        Returns:
            List of pivot concepts with metrics
        """
        schema = self.get_schema_snapshot(snapshot_id)
        if not schema:
            return {"error": "Snapshot not found"}
        
        # Get community structure
        communities = nx.community.louvain_communities(
            schema.graph.to_undirected(), weight='weight'
        )
        
        # Map nodes to communities
        node_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_community[node] = i
        
        # Compute betweenness centrality
        betweenness = nx.betweenness_centrality(schema.graph, weight='weight')
        
        # Find boundary nodes (connecting different communities)
        pivot_concepts = []
        for node in schema.graph.nodes:
            neighbors = list(schema.graph.neighbors(node))
            if not neighbors:
                continue
                
            node_comm = node_community.get(node)
            neighbor_comms = [node_community.get(neigh) for neigh in neighbors]
            
            # Count unique communities among neighbors
            unique_comms = set(neighbor_comms)
            if node_comm in unique_comms:
                unique_comms.remove(node_comm)
                
            if len(unique_comms) > 0:
                pivot_concepts.append({
                    "concept": node,
                    "betweenness": betweenness[node],
                    "connects_communities": list(unique_comms),
                    "degree": schema.graph.degree[node]
                })
        
        # Sort by betweenness centrality
        pivot_concepts.sort(key=lambda x: x["betweenness"], reverse=True)
        
        return pivot_concepts[:10]  # Return top 10 pivot concepts
    
    def generate_schema_summary(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Generate a summary of the schema snapshot.
        
        Args:
            snapshot_id: ID of the schema snapshot
            
        Returns:
            Dict with schema summary
        """
        schema = self.get_schema_snapshot(snapshot_id)
        if not schema:
            return {"error": "Snapshot not found"}
        
        # Get basic metadata
        num_concepts = len(schema.graph.nodes)
        num_relationships = len(schema.graph.edges)
        
        # Get central concepts
        central_concepts = schema.get_central_concepts(top_n=5)
        
        # Get concept clusters
        clusters = schema.get_concept_clusters()
        
        # Compute density
        if num_concepts > 1:
            max_possible_edges = num_concepts * (num_concepts - 1)
            density = num_relationships / max_possible_edges
        else:
            density = 0
        
        # Identify most connected concepts
        degree_centrality = nx.degree_centrality(schema.graph)
        most_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "snapshot_id": snapshot_id,
            "timestamp": self.schema_snapshots[snapshot_id]["timestamp"],
            "num_concepts": num_concepts,
            "num_relationships": num_relationships,
            "density": density,
            "central_concepts": [
                {"concept": concept, "centrality": centrality}
                for concept, centrality in central_concepts
            ],
            "most_connected_concepts": [
                {"concept": concept, "connectivity": centrality}
                for concept, centrality in most_connected
            ],
            "clusters": [
                {"id": cluster_id, "size": len(concepts), "concepts": concepts[:5] + (["..."] if len(concepts) > 5 else [])}
                for cluster_id, concepts in clusters.items()
            ]
        }


# Demo function for testing
def schema_integration_demo():
    """
    Demonstrate the use of SchemaIntegration
    """
    # Create memory sandbox
    memory_sandbox = MemorySandbox()
    
    # Create some synthetic memories
    memory_sandbox.create_synthetic_memory(
        "I learned about neural networks yesterday",
        memory_type="episodic",
        metadata={"importance": 0.8}
    )
    memory_sandbox.create_synthetic_memory(
        "Neural networks are a type of machine learning model",
        memory_type="semantic",
        metadata={"importance": 0.9}
    )
    memory_sandbox.create_synthetic_memory(
        "I feel excited about learning machine learning",
        memory_type="emotional",
        metadata={"importance": 0.7}
    )
    memory_sandbox.create_synthetic_memory(
        "First you initialize the weights of the neural network",
        memory_type="procedural",
        metadata={"importance": 0.6}
    )
    
    # Create schema integration
    schema_integration = SchemaIntegration(memory_sandbox)
    
    # Build schema from memory store
    schema = schema_integration.build_schema_from_memory_store()
    
    # Print schema summary
    summary = schema_integration.generate_schema_summary()
    print("\nSchema Summary:")
    print(f"Node Count: {summary['node_count']}")
    print(f"Edge Count: {summary['edge_count']}")
    print(f"Concept Count: {summary['concept_count']}")
    print(f"Memory Count: {summary['memory_count']}")
    print("\nMemory Types:")
    for memory_type, count in summary["memory_types"].items():
        print(f"- {memory_type}: {count}")
    
    print("\nCentral Concepts:")
    for i, concept in enumerate(summary["central_concepts"], 1):
        print(f"{i}. {concept['name']} (centrality: {concept['centrality']:.4f})")
    
    # Visualize schema graph
    schema_integration.visualize_schema_graph(output_path="schema_graph.png")
    
    # Add more memories to simulate evolution
    memory_sandbox.create_synthetic_memory(
        "Gradient descent is used to optimize neural networks",
        memory_type="semantic",
        metadata={"importance": 0.8}
    )
    memory_sandbox.create_synthetic_memory(
        "Today I practiced implementing a neural network",
        memory_type="episodic",
        metadata={"importance": 0.7}
    )
    
    # Build new schema
    schema_integration.build_schema_from_memory_store()
    
    # Compare schemas
    comparison = schema_integration.compare_schemas()
    print("\nSchema Comparison:")
    print(f"New Nodes: {comparison['summary']['unique_to_graph2']}")
    print(f"New Concepts: {len(comparison['concept_changes']['unique_to_graph2'])}")
    
    # Analyze concept drift
    drift = schema_integration.analyze_concept_drift()
    print("\nConcept Drift:")
    print(f"New Concepts: {drift['summary']['new_concepts']}")
    print(f"Evolved Concepts: {drift['summary']['evolved_concepts']}")
    
    # Visualize schema evolution
    schema_integration.visualize_schema_evolution(
        output_path="schema_evolution.png"
    )
    
    # Identify pivot concepts
    pivot_concepts = schema_integration.identify_pivot_concepts(top_n=3)
    print("\nPivot Concepts:")
    for i, concept in enumerate(pivot_concepts, 1):
        print(f"{i}. {concept['name']} (pivot score: {concept['pivot_score']:.4f})")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    schema_integration_demo() 