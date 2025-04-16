"""
Schema Mutation System for ΨC-AI SDK

This module implements schema evolution capabilities that allow the schema graph
to adapt and restructure when coherence drops or contradictions are detected.
The mutation system provides operations for merging similar nodes, splitting complex nodes,
pruning irrelevant connections, and logging mutation events for transparency.

The schema mutation process is guided by a simulated annealing approach where the
mutation rate decreases over time, promoting stability as the schema matures.
"""

import logging
import time
import random
import uuid
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from datetime import datetime
import networkx as nx
import numpy as np
from dataclasses import dataclass, field

from psi_c_ai_sdk.schema.schema import SchemaNode, SchemaEdge, SchemaGraph
from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of schema mutations that can occur."""
    
    MERGE = "merge"
    SPLIT = "split"
    PRUNE = "prune"
    ADD_CONCEPT = "add_concept"
    CONSOLIDATE = "consolidate"
    RESTRUCTURE = "restructure"


@dataclass
class MutationEvent:
    """Record of a schema mutation event."""
    
    id: str
    mutation_type: MutationType
    timestamp: float
    trigger: str
    coherence_before: float
    coherence_after: float
    affected_nodes: List[str] = field(default_factory=list)
    affected_edges: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "mutation_type": self.mutation_type.value,
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "coherence_before": self.coherence_before,
            "coherence_after": self.coherence_after,
            "affected_nodes": self.affected_nodes,
            "affected_edges": [{"source": src, "target": tgt} for src, tgt in self.affected_edges],
            "metadata": self.metadata
        }


class SchemaMutationSystem:
    """
    System for evolving and adapting the schema graph structure.
    
    This system implements various mutation operations that can be applied to
    a schema graph, including merging similar nodes, splitting complex nodes,
    pruning low-importance connections, and creating new concept nodes.
    
    Mutations are guided by coherence metrics and can be triggered automatically
    when coherence drops below a threshold.
    """
    
    def __init__(
        self,
        schema_graph: SchemaGraph,
        coherence_scorer: CoherenceScorer,
        memory_store: MemoryStore,
        mutation_threshold: float = 0.5,
        max_mutation_history: int = 50,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.01
    ):
        """
        Initialize the schema mutation system.
        
        Args:
            schema_graph: The schema graph to mutate
            coherence_scorer: Coherence scorer for measuring impacts
            memory_store: Memory store to access memories
            mutation_threshold: Coherence threshold below which mutation is triggered
            max_mutation_history: Maximum mutation events to keep in history
            initial_temperature: Starting temperature for annealing (higher = more mutation)
            cooling_rate: Rate at which temperature decreases per mutation
        """
        self.schema_graph = schema_graph
        self.coherence_scorer = coherence_scorer
        self.memory_store = memory_store
        self.mutation_threshold = mutation_threshold
        self.max_mutation_history = max_mutation_history
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        
        self.mutation_history: List[MutationEvent] = []
        self.mutation_count = 0
        self.successful_mutations = 0
        self.last_mutation_time = 0
    
    def check_mutation_needed(self) -> Tuple[bool, str]:
        """
        Check if schema mutation is needed based on coherence.
        
        Returns:
            Tuple of (should_mutate, trigger_reason)
        """
        coherence = self.coherence_scorer.calculate_global_coherence(self.memory_store)
        
        # Check coherence threshold
        if coherence < self.mutation_threshold:
            return True, f"low_coherence ({coherence:.3f} < {self.mutation_threshold:.3f})"
        
        # Check contradiction density
        contradictions = self.coherence_scorer.detect_contradictions(self.memory_store)
        if len(contradictions) > 5:
            return True, f"high_contradictions ({len(contradictions)})"
        
        # Check graph density and complexity
        num_nodes = len(self.schema_graph.graph.nodes)
        num_edges = len(self.schema_graph.graph.edges)
        
        if num_nodes > 0 and num_edges / num_nodes > 10:
            return True, "high_edge_density"
            
        return False, ""
    
    def mutate_schema(self, trigger: str = "manual") -> Optional[MutationEvent]:
        """
        Perform a schema mutation based on current state.
        
        Args:
            trigger: Reason for the mutation
            
        Returns:
            MutationEvent record if mutation occurred, None otherwise
        """
        # Measure coherence before mutation
        coherence_before = self.coherence_scorer.calculate_global_coherence(self.memory_store)
        
        # Choose mutation type based on current state and temperature
        mutation_choices = self._select_mutation_types()
        if not mutation_choices:
            logger.info("No suitable mutation types available")
            return None
            
        mutation_type = random.choices(
            population=list(mutation_choices.keys()),
            weights=list(mutation_choices.values()),
            k=1
        )[0]
        
        # Create mutation event record
        mutation_id = f"mutation_{self.mutation_count}_{int(time.time())}"
        mutation_event = MutationEvent(
            id=mutation_id,
            mutation_type=mutation_type,
            timestamp=time.time(),
            trigger=trigger,
            coherence_before=coherence_before,
            coherence_after=0.0  # Will update after mutation
        )
        
        logger.info(f"Performing schema mutation: {mutation_type.value} (trigger: {trigger})")
        
        # Perform the selected mutation
        success = False
        if mutation_type == MutationType.MERGE:
            success = self._merge_similar_nodes(mutation_event)
        elif mutation_type == MutationType.SPLIT:
            success = self._split_complex_node(mutation_event)
        elif mutation_type == MutationType.PRUNE:
            success = self._prune_weak_connections(mutation_event)
        elif mutation_type == MutationType.ADD_CONCEPT:
            success = self._add_concept_node(mutation_event)
        elif mutation_type == MutationType.CONSOLIDATE:
            success = self._consolidate_nodes(mutation_event)
        elif mutation_type == MutationType.RESTRUCTURE:
            success = self._restructure_subgraph(mutation_event)
        
        if not success:
            logger.warning(f"Mutation {mutation_type.value} failed")
            return None
            
        # Measure coherence after mutation
        coherence_after = self.coherence_scorer.calculate_global_coherence(self.memory_store)
        mutation_event.coherence_after = coherence_after
        
        # Update system state
        self.mutation_count += 1
        self.last_mutation_time = time.time()
        
        # Check if mutation improved coherence
        if coherence_after > coherence_before:
            self.successful_mutations += 1
            mutation_event.metadata["improved_coherence"] = True
            logger.info(f"Mutation improved coherence: {coherence_before:.3f} -> {coherence_after:.3f}")
        else:
            mutation_event.metadata["improved_coherence"] = False
            logger.info(f"Mutation did not improve coherence: {coherence_before:.3f} -> {coherence_after:.3f}")
        
        # Cool down temperature
        self.temperature = max(0.1, self.temperature * (1 - self.cooling_rate))
        mutation_event.metadata["temperature"] = self.temperature
        
        # Add to history
        self.mutation_history.append(mutation_event)
        if len(self.mutation_history) > self.max_mutation_history:
            self.mutation_history.pop(0)
            
        return mutation_event
    
    def _select_mutation_types(self) -> Dict[MutationType, float]:
        """
        Select appropriate mutation types based on current schema state.
        
        Returns:
            Dictionary of mutation types and their weights
        """
        mutation_weights = {}
        
        # Get graph properties
        num_nodes = len(self.schema_graph.graph.nodes)
        num_edges = len(self.schema_graph.graph.edges)
        
        # Need at least 2 nodes for merge operations
        if num_nodes >= 2:
            mutation_weights[MutationType.MERGE] = 0.3 * self.temperature
            
        # Need at least 1 node with multiple connections for split
        if any(len(list(self.schema_graph.graph.neighbors(n))) >= 3 for n in self.schema_graph.graph.nodes):
            mutation_weights[MutationType.SPLIT] = 0.2 * self.temperature
            
        # Need edges to prune
        if num_edges > 0:
            mutation_weights[MutationType.PRUNE] = 0.2 * self.temperature
            
        # Always allow adding concepts
        mutation_weights[MutationType.ADD_CONCEPT] = 0.1
        
        # Need clusters for consolidation
        if num_nodes >= 5:
            mutation_weights[MutationType.CONSOLIDATE] = 0.1 * self.temperature
            
        # Need enough nodes for restructuring
        if num_nodes >= 10:
            mutation_weights[MutationType.RESTRUCTURE] = 0.1 * self.temperature
            
        return mutation_weights
    
    def _merge_similar_nodes(self, mutation_event: MutationEvent) -> bool:
        """
        Merge similar nodes in the schema graph.
        
        Args:
            mutation_event: Event record to update
            
        Returns:
            True if merge was successful, False otherwise
        """
        # Find nodes with similar embeddings
        node_pairs = []
        
        for node1 in list(self.schema_graph.graph.nodes):
            for node2 in list(self.schema_graph.graph.nodes):
                if node1 == node2:
                    continue
                    
                # Get node embeddings
                embedding1 = self._get_node_embedding(node1)
                embedding2 = self._get_node_embedding(node2)
                
                if embedding1 is None or embedding2 is None:
                    continue
                    
                # Calculate similarity
                similarity = self.coherence_scorer.calculate_similarity(embedding1, embedding2)
                
                # Check if highly similar
                if similarity > 0.85:
                    node_pairs.append((node1, node2, similarity))
        
        if not node_pairs:
            return False
            
        # Sort by similarity (highest first)
        node_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Take top pair
        node1, node2, similarity = node_pairs[0]
        
        # Create merged node
        node1_data = self.schema_graph.graph.nodes[node1]
        node2_data = self.schema_graph.graph.nodes[node2]
        
        merged_id = f"merged_{uuid.uuid4().hex[:8]}"
        
        # Combine tags
        merged_tags = list(set(node1_data.get("tags", []) + node2_data.get("tags", [])))
        
        # Combine importance (weighted average)
        merged_importance = (
            node1_data.get("importance", 0.5) + node2_data.get("importance", 0.5)
        ) / 2
        
        # Create metadata
        merged_metadata = {
            "merged_from": [node1, node2],
            "merged_similarity": similarity,
            "merged_at": time.time()
        }
        
        # Add merged node
        self.schema_graph.graph.add_node(
            merged_id,
            label=f"Merged: {node1_data.get('label', node1)[:20]} + {node2_data.get('label', node2)[:20]}",
            node_type="concept",
            importance=merged_importance,
            tags=merged_tags,
            metadata=merged_metadata
        )
        
        # Connect to all neighbors of original nodes
        for neighbor in set(self.schema_graph.graph.neighbors(node1)):
            if neighbor != node2:  # Avoid self-loops
                weight = self.schema_graph.graph[node1][neighbor].get("weight", 0.5)
                self.schema_graph.graph.add_edge(merged_id, neighbor, weight=weight)
                mutation_event.affected_edges.append((merged_id, neighbor))
                
        for neighbor in set(self.schema_graph.graph.neighbors(node2)):
            if neighbor != node1:  # Avoid self-loops
                weight = self.schema_graph.graph[node2][neighbor].get("weight", 0.5)
                self.schema_graph.graph.add_edge(merged_id, neighbor, weight=weight)
                mutation_event.affected_edges.append((merged_id, neighbor))
        
        # Record affected nodes
        mutation_event.affected_nodes = [node1, node2, merged_id]
        
        # Remove original nodes
        self.schema_graph.graph.remove_node(node1)
        self.schema_graph.graph.remove_node(node2)
        
        logger.info(f"Merged nodes {node1} and {node2} into {merged_id} (similarity: {similarity:.3f})")
        return True
    
    def _split_complex_node(self, mutation_event: MutationEvent) -> bool:
        """
        Split a complex node with many connections into multiple nodes.
        
        Args:
            mutation_event: Event record to update
            
        Returns:
            True if split was successful, False otherwise
        """
        # Find nodes with many connections
        complex_nodes = []
        
        for node in self.schema_graph.graph.nodes:
            neighbors = list(self.schema_graph.graph.neighbors(node))
            if len(neighbors) >= 5:
                complex_nodes.append((node, len(neighbors)))
        
        if not complex_nodes:
            return False
            
        # Sort by number of connections (highest first)
        complex_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Take top node
        node_to_split, num_connections = complex_nodes[0]
        node_data = self.schema_graph.graph.nodes[node_to_split]
        
        # Get all neighbors
        neighbors = list(self.schema_graph.graph.neighbors(node_to_split))
        
        # Split into approximately 2 groups using neighbor embeddings
        neighbor_embeddings = []
        valid_neighbors = []
        
        for neighbor in neighbors:
            emb = self._get_node_embedding(neighbor)
            if emb is not None:
                neighbor_embeddings.append(emb)
                valid_neighbors.append(neighbor)
        
        if len(valid_neighbors) < 4:
            return False
            
        # Use K-means to split
        from sklearn.cluster import KMeans
        
        k = 2  # Split into 2 groups
        kmeans = KMeans(n_clusters=k, random_state=42).fit(neighbor_embeddings)
        clusters = kmeans.labels_
        
        # Create split nodes
        split_nodes = []
        for i in range(k):
            split_id = f"split_{node_to_split}_{i}_{uuid.uuid4().hex[:6]}"
            
            split_label = f"Split {i+1} of {node_data.get('label', node_to_split)}"
            if len(split_label) > 30:
                split_label = split_label[:27] + "..."
                
            self.schema_graph.graph.add_node(
                split_id,
                label=split_label,
                node_type=node_data.get("node_type", "concept"),
                importance=node_data.get("importance", 0.5) * 0.9,  # Slightly lower
                tags=node_data.get("tags", []),
                metadata={
                    "split_from": node_to_split,
                    "split_group": i,
                    "split_at": time.time()
                }
            )
            split_nodes.append(split_id)
            mutation_event.affected_nodes.append(split_id)
        
        # Connect neighbors to appropriate split nodes
        for i, neighbor in enumerate(valid_neighbors):
            cluster = clusters[i]
            split_node = split_nodes[cluster]
            
            weight = self.schema_graph.graph[node_to_split][neighbor].get("weight", 0.5)
            self.schema_graph.graph.add_edge(split_node, neighbor, weight=weight)
            mutation_event.affected_edges.append((split_node, neighbor))
        
        # Connect split nodes to each other
        for i in range(len(split_nodes)):
            for j in range(i+1, len(split_nodes)):
                self.schema_graph.graph.add_edge(
                    split_nodes[i], split_nodes[j], 
                    weight=0.7,  # Strong connection between splits
                    edge_type="split_relation"
                )
                mutation_event.affected_edges.append((split_nodes[i], split_nodes[j]))
        
        # Remove original node
        mutation_event.affected_nodes.append(node_to_split)
        self.schema_graph.graph.remove_node(node_to_split)
        
        logger.info(f"Split node {node_to_split} with {num_connections} connections into {k} new nodes")
        return True
    
    def _prune_weak_connections(self, mutation_event: MutationEvent) -> bool:
        """
        Prune weak connections from the schema graph.
        
        Args:
            mutation_event: Event record to update
            
        Returns:
            True if pruning was successful, False otherwise
        """
        # Find edges with low weights
        weak_edges = []
        
        for u, v, data in self.schema_graph.graph.edges(data=True):
            weight = data.get("weight", 0.5)
            if weight < 0.2:
                weak_edges.append((u, v, weight))
        
        if not weak_edges:
            return False
            
        # Sort by weight (lowest first)
        weak_edges.sort(key=lambda x: x[2])
        
        # Remove up to 1/3 of the weakest edges
        edges_to_remove = weak_edges[:max(1, len(weak_edges) // 3)]
        
        for u, v, _ in edges_to_remove:
            self.schema_graph.graph.remove_edge(u, v)
            mutation_event.affected_edges.append((u, v))
        
        # Record affected nodes
        affected_nodes = set()
        for u, v, _ in edges_to_remove:
            affected_nodes.add(u)
            affected_nodes.add(v)
        
        mutation_event.affected_nodes = list(affected_nodes)
        
        logger.info(f"Pruned {len(edges_to_remove)} weak connections below weight 0.2")
        return True
    
    def _add_concept_node(self, mutation_event: MutationEvent) -> bool:
        """
        Add a new concept node based on connected memory clusters.
        
        Args:
            mutation_event: Event record to update
            
        Returns:
            True if concept addition was successful, False otherwise
        """
        # Get memory nodes
        memory_nodes = [
            node for node in self.schema_graph.graph.nodes 
            if self.schema_graph.graph.nodes[node].get("node_type") == "memory"
        ]
        
        if len(memory_nodes) < 5:
            return False
            
        # Use community detection to find clusters
        try:
            from networkx.algorithms import community
            
            # Create subgraph of memory nodes
            memory_subgraph = self.schema_graph.graph.subgraph(memory_nodes)
            
            # Find communities
            communities = list(community.greedy_modularity_communities(memory_subgraph))
            
            # Filter to communities with at least 3 nodes
            valid_communities = [c for c in communities if len(c) >= 3]
            
            if not valid_communities:
                return False
                
            # Select a random community
            selected_community = random.choice(valid_communities)
            selected_community = list(selected_community)
            
            # Get memory content for these nodes
            memory_contents = []
            for node in selected_community:
                memory_id = self.schema_graph.graph.nodes[node].get("memory_id")
                if memory_id:
                    memory = self.memory_store.get_memory(uuid.UUID(memory_id))
                    if memory:
                        memory_contents.append(memory.content)
            
            if not memory_contents:
                return False
                
            # Create concept node
            concept_id = f"concept_{uuid.uuid4().hex[:8]}"
            
            # Generate a label from common tags or use generic label
            common_tags = self._find_common_tags(selected_community)
            
            if common_tags:
                concept_label = f"Concept: {', '.join(common_tags[:2])}"
            else:
                concept_label = f"Concept {len(memory_contents)} memories"
            
            # Create the concept node
            self.schema_graph.graph.add_node(
                concept_id,
                label=concept_label,
                node_type="concept",
                importance=0.6,
                tags=common_tags,
                metadata={
                    "created_from": list(selected_community),
                    "created_at": time.time(),
                    "community_size": len(selected_community)
                }
            )
            
            # Connect to all nodes in the community
            for node in selected_community:
                self.schema_graph.graph.add_edge(
                    concept_id, node, 
                    weight=0.6,
                    edge_type="concept_relation"
                )
                mutation_event.affected_edges.append((concept_id, node))
            
            # Record affected nodes
            mutation_event.affected_nodes = [concept_id] + list(selected_community)
            
            logger.info(f"Added concept node {concept_id} connecting {len(selected_community)} related memories")
            return True
            
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            return False
    
    def _consolidate_nodes(self, mutation_event: MutationEvent) -> bool:
        """
        Consolidate related concept nodes.
        
        Args:
            mutation_event: Event record to update
            
        Returns:
            True if consolidation was successful, False otherwise
        """
        # Get concept nodes
        concept_nodes = [
            node for node in self.schema_graph.graph.nodes 
            if self.schema_graph.graph.nodes[node].get("node_type") == "concept"
        ]
        
        if len(concept_nodes) < 3:
            return False
            
        # Find concepts with overlapping neighbors
        concept_pairs = []
        
        for i, concept1 in enumerate(concept_nodes):
            for concept2 in concept_nodes[i+1:]:
                neighbors1 = set(self.schema_graph.graph.neighbors(concept1))
                neighbors2 = set(self.schema_graph.graph.neighbors(concept2))
                
                overlap = neighbors1.intersection(neighbors2)
                
                if len(overlap) >= 2:
                    jaccard = len(overlap) / len(neighbors1.union(neighbors2))
                    concept_pairs.append((concept1, concept2, jaccard))
        
        if not concept_pairs:
            return False
            
        # Sort by overlap (highest first)
        concept_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Take top pair
        concept1, concept2, overlap = concept_pairs[0]
        
        # Consolidate by creating a parent concept
        parent_id = f"parent_{uuid.uuid4().hex[:8]}"
        
        # Combine metadata
        c1_data = self.schema_graph.graph.nodes[concept1]
        c2_data = self.schema_graph.graph.nodes[concept2]
        
        # Create the parent node
        self.schema_graph.graph.add_node(
            parent_id,
            label=f"Parent: {c1_data.get('label', '')[:10]} + {c2_data.get('label', '')[:10]}",
            node_type="concept",
            importance=max(c1_data.get("importance", 0.5), c2_data.get("importance", 0.5)) + 0.1,
            tags=list(set(c1_data.get("tags", []) + c2_data.get("tags", []))),
            metadata={
                "consolidated_from": [concept1, concept2],
                "overlap_score": overlap,
                "created_at": time.time()
            }
        )
        
        # Connect parent to children
        self.schema_graph.graph.add_edge(parent_id, concept1, weight=0.8, edge_type="parent_child")
        self.schema_graph.graph.add_edge(parent_id, concept2, weight=0.8, edge_type="parent_child")
        
        mutation_event.affected_edges.append((parent_id, concept1))
        mutation_event.affected_edges.append((parent_id, concept2))
        
        # Get shared neighbors
        common_neighbors = set(self.schema_graph.graph.neighbors(concept1)) & set(self.schema_graph.graph.neighbors(concept2))
        
        # Connect parent to common neighbors with stronger weight
        for neighbor in common_neighbors:
            weight1 = self.schema_graph.graph[concept1][neighbor].get("weight", 0.5)
            weight2 = self.schema_graph.graph[concept2][neighbor].get("weight", 0.5)
            
            # Average weight, but higher for common neighbors
            avg_weight = (weight1 + weight2) / 2 + 0.1
            
            self.schema_graph.graph.add_edge(parent_id, neighbor, weight=avg_weight)
            mutation_event.affected_edges.append((parent_id, neighbor))
        
        # Record affected nodes
        mutation_event.affected_nodes = [parent_id, concept1, concept2]
        
        logger.info(f"Created parent concept {parent_id} consolidating {concept1} and {concept2}")
        return True
    
    def _restructure_subgraph(self, mutation_event: MutationEvent) -> bool:
        """
        Restructure a portion of the schema graph to improve overall coherence.
        
        Args:
            mutation_event: Event record to update
            
        Returns:
            True if restructuring was successful, False otherwise
        """
        # Find a portion of the graph with low internal coherence
        # We'll use a simple approach - look for nodes with many contradicting neighbors
        
        problematic_nodes = []
        
        for node in self.schema_graph.graph.nodes:
            # Get node embedding
            node_emb = self._get_node_embedding(node)
            if node_emb is None:
                continue
                
            # Get neighbors
            neighbors = list(self.schema_graph.graph.neighbors(node))
            if len(neighbors) < 3:
                continue
                
            # Check for contradictions among neighbors
            contradiction_count = 0
            
            for i, n1 in enumerate(neighbors):
                n1_emb = self._get_node_embedding(n1)
                if n1_emb is None:
                    continue
                    
                for n2 in neighbors[i+1:]:
                    n2_emb = self._get_node_embedding(n2)
                    if n2_emb is None:
                        continue
                        
                    # Calculate similarity
                    similarity = self.coherence_scorer.calculate_similarity(n1_emb, n2_emb)
                    
                    # Check if potentially contradictory
                    if similarity < 0.2:
                        contradiction_count += 1
            
            if contradiction_count >= 2:
                problematic_nodes.append((node, contradiction_count))
        
        if not problematic_nodes:
            return False
            
        # Sort by contradiction count (highest first)
        problematic_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Take top node
        focus_node, contradictions = problematic_nodes[0]
        
        # Get neighbors and their embeddings
        neighbors = list(self.schema_graph.graph.neighbors(focus_node))
        valid_neighbors = []
        embeddings = []
        
        for neighbor in neighbors:
            emb = self._get_node_embedding(neighbor)
            if emb is not None:
                valid_neighbors.append(neighbor)
                embeddings.append(emb)
        
        if len(valid_neighbors) < 3:
            return False
            
        # Use clustering to find coherent groups
        from sklearn.cluster import AgglomerativeClustering
        
        # Determine number of clusters based on data
        max_clusters = min(4, len(valid_neighbors) // 2)
        if max_clusters < 2:
            max_clusters = 2
            
        clustering = AgglomerativeClustering(
            n_clusters=max_clusters,
            metric='euclidean',
            linkage='ward'
        ).fit(embeddings)
        
        clusters = clustering.labels_
        
        # Create intermediate nodes for each cluster
        new_nodes = []
        for i in range(max_clusters):
            # Get nodes in this cluster
            cluster_nodes = [valid_neighbors[j] for j in range(len(valid_neighbors)) if clusters[j] == i]
            
            if not cluster_nodes:
                continue
                
            # Create intermediate node
            new_id = f"cluster_{uuid.uuid4().hex[:8]}"
            
            # Find common tags
            common_tags = self._find_common_tags(cluster_nodes)
            
            # Create the new node
            self.schema_graph.graph.add_node(
                new_id,
                label=f"Cluster: {', '.join(common_tags[:2]) if common_tags else f'Group {i+1}'}",
                node_type="concept",
                importance=0.5,
                tags=common_tags,
                metadata={
                    "restructured_from": list(cluster_nodes),
                    "cluster_id": i,
                    "created_at": time.time()
                }
            )
            
            # Connect to all nodes in the cluster
            for node in cluster_nodes:
                old_weight = self.schema_graph.graph[focus_node][node].get("weight", 0.5)
                
                # Stronger connection
                self.schema_graph.graph.add_edge(new_id, node, weight=max(old_weight, 0.6))
                mutation_event.affected_edges.append((new_id, node))
                
                # Weaken original connection but don't remove
                self.schema_graph.graph[focus_node][node]["weight"] = old_weight * 0.5
                
            # Connect to focus node
            self.schema_graph.graph.add_edge(
                focus_node, new_id, 
                weight=0.7,
                edge_type="restructured"
            )
            mutation_event.affected_edges.append((focus_node, new_id))
            
            new_nodes.append(new_id)
        
        # Record affected nodes
        mutation_event.affected_nodes = [focus_node] + new_nodes + valid_neighbors
        
        logger.info(f"Restructured subgraph around {focus_node} with {len(valid_neighbors)} neighbors into {len(new_nodes)} clusters")
        return True
    
    def _get_node_embedding(self, node_id: str) -> Optional[List[float]]:
        """Get embedding for a node if available."""
        if node_id not in self.schema_graph.graph:
            return None
            
        node_data = self.schema_graph.graph.nodes[node_id]
        
        # If node has memory_id, get embedding from memory
        if node_data.get("memory_id"):
            memory_id = node_data.get("memory_id")
            memory = self.memory_store.get_memory(uuid.UUID(memory_id))
            if memory and memory.embedding:
                return memory.embedding
                
        # Check if node has its own embedding
        if "embedding" in node_data:
            return node_data["embedding"]
            
        return None
    
    def _find_common_tags(self, node_ids: List[str]) -> List[str]:
        """Find common tags among a group of nodes."""
        tag_counts = {}
        
        for node_id in node_ids:
            if node_id in self.schema_graph.graph:
                tags = self.schema_graph.graph.nodes[node_id].get("tags", [])
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Get tags present in at least 1/3 of the nodes
        threshold = max(1, len(node_ids) // 3)
        common_tags = [tag for tag, count in tag_counts.items() if count >= threshold]
        
        return common_tags
    
    def get_mutation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about schema mutations.
        
        Returns:
            Dictionary with mutation statistics
        """
        return {
            "total_mutations": self.mutation_count,
            "successful_mutations": self.successful_mutations,
            "success_rate": self.successful_mutations / max(1, self.mutation_count),
            "last_mutation_time": self.last_mutation_time,
            "temperature": self.temperature,
            "mutation_history_size": len(self.mutation_history)
        }
    
    def get_mutation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get mutation history.
        
        Args:
            limit: Maximum events to return (None = all)
            
        Returns:
            List of mutation events as dictionaries
        """
        if limit is not None:
            events = self.mutation_history[-limit:]
        else:
            events = self.mutation_history.copy()
            
        return [event.to_dict() for event in events]
    
    def force_mutation(self, mutation_type: Optional[MutationType] = None) -> Optional[MutationEvent]:
        """
        Force a schema mutation of a specific type.
        
        Args:
            mutation_type: Type of mutation to perform (None = auto-select)
            
        Returns:
            MutationEvent if successful, None otherwise
        """
        if mutation_type is None:
            return self.mutate_schema(trigger="manual_force")
        
        # Create mutation event
        mutation_id = f"mutation_{self.mutation_count}_{int(time.time())}"
        
        mutation_event = MutationEvent(
            id=mutation_id,
            mutation_type=mutation_type,
            timestamp=time.time(),
            trigger="manual_force",
            coherence_before=self.coherence_scorer.calculate_global_coherence(self.memory_store),
            coherence_after=0.0  # Will update after
        )
        
        # Attempt the mutation
        success = False
        
        if mutation_type == MutationType.MERGE:
            success = self._merge_similar_nodes(mutation_event)
        elif mutation_type == MutationType.SPLIT:
            success = self._split_complex_node(mutation_event)
        elif mutation_type == MutationType.PRUNE:
            success = self._prune_weak_connections(mutation_event)
        elif mutation_type == MutationType.ADD_CONCEPT:
            success = self._add_concept_node(mutation_event)
        elif mutation_type == MutationType.CONSOLIDATE:
            success = self._consolidate_nodes(mutation_event)
        elif mutation_type == MutationType.RESTRUCTURE:
            success = self._restructure_subgraph(mutation_event)
            
        if not success:
            logger.warning(f"Forced mutation {mutation_type.value} failed")
            return None
            
        # Measure coherence after
        mutation_event.coherence_after = self.coherence_scorer.calculate_global_coherence(self.memory_store)
        
        # Update state
        self.mutation_count += 1
        self.last_mutation_time = time.time()
        
        if mutation_event.coherence_after > mutation_event.coherence_before:
            self.successful_mutations += 1
            
        # Add to history
        self.mutation_history.append(mutation_event)
        if len(self.mutation_history) > self.max_mutation_history:
            self.mutation_history.pop(0)
            
        logger.info(f"Forced mutation {mutation_type.value} completed")
        return mutation_event 