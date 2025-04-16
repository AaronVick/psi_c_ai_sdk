"""
Schema Graph Builder for ΨC-AI SDK

This module implements a graph-based schema representation of memories,
allowing the system to model relationships between memories and concepts.
The schema graph captures coherence relationships between memories,
forming a structured knowledge representation that facilitates reasoning
and reflection.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from uuid import UUID
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from dataclasses import dataclass, field

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SchemaNode:
    """A node in the schema graph representing a memory or concept."""
    
    id: str
    memory_id: Optional[str] = None
    label: str = ""
    node_type: str = "memory"
    importance: float = 0.5
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "label": self.label,
            "node_type": self.node_type,
            "importance": self.importance,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_memory(cls, memory: Memory) -> 'SchemaNode':
        """Create a schema node from a memory object."""
        return cls(
            id=f"memory_{memory.uuid}",
            memory_id=str(memory.uuid),
            label=memory.content[:50] + ("..." if len(memory.content) > 50 else ""),
            node_type="memory",
            importance=memory.importance,
            embedding=memory.embedding,
            tags=memory.tags or [],
            metadata={
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "memory_type": memory.memory_type.value if memory.memory_type else "unknown",
                "source": memory.source
            }
        )


@dataclass
class SchemaEdge:
    """An edge in the schema graph representing a relationship between nodes."""
    
    source_id: str
    target_id: str
    weight: float = 0.0
    edge_type: str = "coherence"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source_id,
            "target": self.target_id,
            "weight": self.weight,
            "edge_type": self.edge_type,
            "metadata": self.metadata
        }


class SchemaGraph:
    """
    A graph-based representation of memory schema.
    
    The schema graph captures relationships between memories and concepts,
    forming a structured knowledge representation that facilitates 
    reasoning and context management.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        coherence_scorer: CoherenceScorer,
        min_edge_weight: float = 0.3,
        auto_prune: bool = True,
        max_nodes: int = 500
    ):
        """
        Initialize the schema graph.
        
        Args:
            memory_store: Memory store to track
            coherence_scorer: Coherence scorer for calculating relationships
            min_edge_weight: Minimum coherence to create an edge between nodes
            auto_prune: Whether to automatically prune low-importance nodes
            max_nodes: Maximum number of nodes to maintain in the graph
        """
        self.memory_store = memory_store
        self.coherence_scorer = coherence_scorer
        self.min_edge_weight = min_edge_weight
        self.auto_prune = auto_prune
        self.max_nodes = max_nodes
        
        # Initialize NetworkX graph for schema representation
        self.graph = nx.Graph()
        
        # Track node and edge counts
        self.node_count = 0
        self.edge_count = 0
        
        # Track memory IDs that have been added to the schema
        self.tracked_memory_ids: Set[str] = set()
        
        # Performance metrics
        self.last_update_time = 0
        self.last_update_duration = 0
    
    def add_memory_node(self, memory: Memory) -> str:
        """
        Add a memory to the schema graph.
        
        Args:
            memory: Memory object to add
            
        Returns:
            ID of the created node
        """
        # Check if memory is already tracked
        if str(memory.uuid) in self.tracked_memory_ids:
            # Update the existing node instead
            return self.update_memory_node(memory)
        
        # Create a schema node from the memory
        node = SchemaNode.from_memory(memory)
        
        # Add to NetworkX graph
        self.graph.add_node(
            node.id,
            memory_id=node.memory_id,
            label=node.label,
            node_type=node.node_type,
            importance=node.importance,
            tags=node.tags,
            metadata=node.metadata
        )
        
        # Track this memory
        self.tracked_memory_ids.add(str(memory.uuid))
        self.node_count += 1
        
        # Return the node ID
        return node.id
    
    def update_memory_node(self, memory: Memory) -> str:
        """
        Update an existing memory node in the schema.
        
        Args:
            memory: Updated memory object
            
        Returns:
            ID of the updated node
        """
        node_id = f"memory_{memory.uuid}"
        
        if node_id in self.graph:
            # Update node attributes
            self.graph.nodes[node_id]["label"] = memory.content[:50] + ("..." if len(memory.content) > 50 else "")
            self.graph.nodes[node_id]["importance"] = memory.importance
            self.graph.nodes[node_id]["tags"] = memory.tags or []
            
            # Update metadata
            metadata = self.graph.nodes[node_id].get("metadata", {})
            metadata.update({
                "last_updated": time.time(),
                "memory_type": memory.memory_type.value if memory.memory_type else "unknown",
            })
            self.graph.nodes[node_id]["metadata"] = metadata
            
            return node_id
        else:
            # Node doesn't exist, so add it
            return self.add_memory_node(memory)
    
    def add_concept_node(
        self,
        label: str,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a concept node to the schema that doesn't directly represent a memory.
        
        Args:
            label: Label for the concept
            embedding: Optional embedding vector
            importance: Importance score (0-1)
            tags: Optional tags for categorization
            metadata: Optional metadata
            
        Returns:
            ID of the created node
        """
        # Generate a unique ID for this concept
        node_id = f"concept_{int(time.time())}_{len(self.graph)}"
        
        # Add to NetworkX graph
        self.graph.add_node(
            node_id,
            memory_id=None,
            label=label,
            node_type="concept",
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            embedding=embedding
        )
        
        self.node_count += 1
        return node_id
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float,
        edge_type: str = "coherence",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an edge between two nodes in the schema.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            weight: Edge weight (e.g., coherence score)
            edge_type: Type of relationship
            metadata: Optional metadata
            
        Returns:
            True if edge was added, False otherwise
        """
        # Validate that nodes exist
        if source_id not in self.graph or target_id not in self.graph:
            return False
        
        # Don't add edges below the minimum weight
        if weight < self.min_edge_weight:
            return False
        
        # Add the edge to the graph
        self.graph.add_edge(
            source_id,
            target_id,
            weight=weight,
            edge_type=edge_type,
            metadata=metadata or {}
        )
        
        self.edge_count += 1
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the schema.
        
        Args:
            node_id: ID of the node to remove
            
        Returns:
            True if the node was removed, False otherwise
        """
        if node_id in self.graph:
            # If this is a memory node, remove from tracked memories
            if node_id.startswith("memory_"):
                memory_id = self.graph.nodes[node_id].get("memory_id")
                if memory_id and memory_id in self.tracked_memory_ids:
                    self.tracked_memory_ids.remove(memory_id)
            
            # Remove from graph
            self.graph.remove_node(node_id)
            self.node_count -= 1
            return True
        return False
    
    def calculate_memory_relationships(self, memory: Memory) -> List[Tuple[str, float]]:
        """
        Calculate relationships between a memory and existing nodes.
        
        Args:
            memory: Memory to calculate relationships for
            
        Returns:
            List of tuples (node_id, coherence_score)
        """
        if not memory.embedding:
            return []
        
        relationships = []
        
        # Check against all nodes with embeddings
        for node_id in self.graph:
            node_data = self.graph.nodes[node_id]
            
            # Skip if it's the same memory or node has no embedding
            if node_data.get("memory_id") == str(memory.uuid):
                continue
                
            # Get node embedding
            node_embedding = None
            if node_id.startswith("memory_") and node_data.get("memory_id"):
                memory_id = node_data.get("memory_id")
                node_memory = self.memory_store.get_memory(UUID(memory_id))
                if node_memory and node_memory.embedding:
                    node_embedding = node_memory.embedding
            elif "embedding" in node_data:
                node_embedding = node_data["embedding"]
                
            if not node_embedding:
                continue
                
            # Calculate coherence
            coherence = self.coherence_scorer.calculate_similarity(memory.embedding, node_embedding)
            
            # Only include significant relationships
            if coherence >= self.min_edge_weight:
                relationships.append((node_id, coherence))
                
        # Sort by coherence (highest first)
        relationships.sort(key=lambda x: x[1], reverse=True)
        return relationships
    
    def update_schema(self, max_memories: int = 100) -> Dict[str, int]:
        """
        Update the schema graph based on the current memory store.
        
        Args:
            max_memories: Maximum number of memories to process per update
            
        Returns:
            Dictionary with update statistics
        """
        start_time = time.time()
        
        # Statistics to track
        stats = {
            "nodes_added": 0,
            "nodes_updated": 0,
            "edges_added": 0,
            "nodes_pruned": 0
        }
        
        # Get memories not yet in the schema
        memories = self.memory_store.get_all_memories()
        new_memories = [
            m for m in memories 
            if str(m.uuid) not in self.tracked_memory_ids
        ]
        
        # Also get high-importance memories for updating
        high_importance = [
            m for m in memories
            if str(m.uuid) in self.tracked_memory_ids and m.importance > 0.7
        ]
        
        # Process new memories (up to the limit)
        for memory in new_memories[:max_memories]:
            # Add the memory node
            node_id = self.add_memory_node(memory)
            stats["nodes_added"] += 1
            
            # Calculate and add relationships
            relationships = self.calculate_memory_relationships(memory)
            for related_id, score in relationships:
                if self.add_edge(node_id, related_id, score):
                    stats["edges_added"] += 1
        
        # Update high-importance memory nodes
        for memory in high_importance[:max(10, max_memories // 10)]:
            self.update_memory_node(memory)
            stats["nodes_updated"] += 1
        
        # Prune low-importance nodes if needed
        if self.auto_prune and len(self.graph) > self.max_nodes:
            pruned = self._prune_graph()
            stats["nodes_pruned"] = pruned
        
        # Update metrics
        self.last_update_duration = time.time() - start_time
        self.last_update_time = time.time()
        
        return stats
    
    def _prune_graph(self) -> int:
        """
        Prune low-importance nodes to maintain graph size.
        
        Returns:
            Number of nodes pruned
        """
        # If we're under the limit, no pruning needed
        if len(self.graph) <= self.max_nodes:
            return 0
        
        # Calculate how many nodes to remove
        to_remove = len(self.graph) - self.max_nodes
        
        # Get list of nodes sorted by importance (lowest first)
        nodes = [(node, data.get("importance", 0)) for node, data in self.graph.nodes(data=True)]
        nodes.sort(key=lambda x: x[1])
        
        # Remove the least important nodes
        pruned = 0
        for node_id, _ in nodes[:to_remove]:
            # Don't prune concept nodes that are highly connected
            if self.graph.nodes[node_id].get("node_type") == "concept" and self.graph.degree(node_id) > 5:
                continue
                
            if self.remove_node(node_id):
                pruned += 1
                
        return pruned
    
    def get_subgraph(self, center_id: str, max_distance: int = 2) -> nx.Graph:
        """
        Get a subgraph centered around a node.
        
        Args:
            center_id: ID of the central node
            max_distance: Maximum distance from center
            
        Returns:
            NetworkX subgraph
        """
        if center_id not in self.graph:
            return nx.Graph()
            
        # Get nodes within distance
        nodes = set([center_id])
        frontier = set([center_id])
        
        for _ in range(max_distance):
            new_frontier = set()
            for node in frontier:
                neighbors = set(self.graph.neighbors(node))
                new_frontier.update(neighbors - nodes)
            nodes.update(new_frontier)
            frontier = new_frontier
            
        # Return the subgraph
        return self.graph.subgraph(nodes)
    
    def get_similar_nodes(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find nodes with similar embeddings to the query.
        
        Args:
            query_embedding: Embedding vector to compare against
            top_k: Maximum number of results
            
        Returns:
            List of tuples (node_id, similarity_score)
        """
        similarities = []
        
        for node_id, data in self.graph.nodes(data=True):
            # Get node embedding
            node_embedding = None
            if node_id.startswith("memory_") and data.get("memory_id"):
                memory_id = data.get("memory_id")
                node_memory = self.memory_store.get_memory(UUID(memory_id))
                if node_memory and node_memory.embedding:
                    node_embedding = node_memory.embedding
            elif "embedding" in data:
                node_embedding = data["embedding"]
                
            if not node_embedding:
                continue
                
            # Calculate similarity
            similarity = self.coherence_scorer.calculate_similarity(query_embedding, node_embedding)
            similarities.append((node_id, similarity))
            
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def visualize(
        self,
        filename: Optional[str] = None,
        highlight_nodes: Optional[List[str]] = None,
        max_nodes: int = 100
    ) -> None:
        """
        Visualize the schema graph.
        
        Args:
            filename: Optional file path to save the visualization
            highlight_nodes: Optional list of node IDs to highlight
            max_nodes: Maximum nodes to include in visualization
        """
        # Use a subset if graph is too large
        if len(self.graph) > max_nodes:
            # Prioritize high-importance and highlighted nodes
            importance_dict = nx.get_node_attributes(self.graph, "importance")
            nodes_by_importance = sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Get top nodes by importance
            top_nodes = [node for node, _ in nodes_by_importance[:max_nodes]]
            
            # Add highlighted nodes if specified
            if highlight_nodes:
                for node in highlight_nodes:
                    if node in self.graph and node not in top_nodes:
                        top_nodes.append(node)
                        if len(top_nodes) >= max_nodes:
                            break
            
            # Create subgraph
            graph = self.graph.subgraph(top_nodes)
        else:
            graph = self.graph
            
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Get node attributes
        node_importance = [
            data.get("importance", 0.5) * 500 
            for _, data in graph.nodes(data=True)
        ]
        
        node_colors = []
        for node, data in graph.nodes(data=True):
            if highlight_nodes and node in highlight_nodes:
                node_colors.append("red")
            elif data.get("node_type") == "concept":
                node_colors.append("blue")
            else:
                node_colors.append("green")
                
        # Get edge weights
        edge_weights = [
            data.get("weight", 0.1) * 2
            for _, _, data in graph.edges(data=True)
        ]
        
        # Create layout
        layout = nx.spring_layout(graph, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph,
            layout,
            node_size=node_importance,
            node_color=node_colors,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph,
            layout,
            width=edge_weights,
            alpha=0.5,
            edge_color="gray"
        )
        
        # Draw labels
        label_dict = {}
        for node, data in graph.nodes(data=True):
            label = data.get("label", node)
            if len(label) > 20:
                label = label[:17] + "..."
            label_dict[node] = label
            
        nx.draw_networkx_labels(
            graph,
            layout,
            labels=label_dict,
            font_size=8
        )
        
        plt.title(f"Memory Schema Graph ({len(graph)} nodes, {graph.number_of_edges()} edges)")
        plt.axis("off")
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            logger.info(f"Schema visualization saved to {filename}")
        else:
            plt.show()
        
        plt.close()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the schema graph to a dictionary for serialization.
        
        Returns:
            Dict representation of the schema
        """
        nodes = []
        for node, data in self.graph.nodes(data=True):
            node_dict = {
                "id": node,
                "label": data.get("label", ""),
                "node_type": data.get("node_type", "memory"),
                "importance": data.get("importance", 0.5),
                "memory_id": data.get("memory_id"),
                "tags": data.get("tags", []),
                "metadata": data.get("metadata", {})
            }
            nodes.append(node_dict)
            
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edge_dict = {
                "source": source,
                "target": target,
                "weight": data.get("weight", 0.0),
                "edge_type": data.get("edge_type", "coherence"),
                "metadata": data.get("metadata", {})
            }
            edges.append(edge_dict)
            
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": self.node_count,
                "edge_count": self.edge_count,
                "tracked_memories": len(self.tracked_memory_ids),
                "last_update": self.last_update_time
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the schema graph.
        
        Returns:
            Dictionary with schema statistics
        """
        # Calculate node type distribution
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        # Calculate average degree
        if len(self.graph) > 0:
            avg_degree = sum(dict(self.graph.degree()).values()) / len(self.graph)
        else:
            avg_degree = 0
            
        # Calculate clustering coefficient
        try:
            clustering = nx.average_clustering(self.graph)
        except:
            clustering = 0
            
        # Get connected components
        components = list(nx.connected_components(self.graph))
        
        return {
            "node_count": len(self.graph),
            "edge_count": self.graph.number_of_edges(),
            "node_types": node_types,
            "avg_degree": avg_degree,
            "clustering_coefficient": clustering,
            "connected_components": len(components),
            "largest_component_size": len(max(components, key=len)) if components else 0,
            "tracked_memories": len(self.tracked_memory_ids),
            "last_update_time": self.last_update_time,
            "last_update_duration": self.last_update_duration
        } 