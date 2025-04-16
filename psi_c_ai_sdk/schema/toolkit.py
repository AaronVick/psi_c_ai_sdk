"""
Schema Toolkit: A unified interface for working with memory schema graphs.

This module provides a high-level toolkit for creating, updating, mutating,
and visualizing schema graphs that represent relationships between memories.
The toolkit integrates schema graph management, mutation operations,
and visualization capabilities into a cohesive system.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime
import os
import networkx as nx
import matplotlib.pyplot as plt
from uuid import UUID

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer
from psi_c_ai_sdk.schema.schema import SchemaGraph, SchemaNode
from psi_c_ai_sdk.schema.mutation import SchemaMutationSystem, MutationType, MutationEvent
from psi_c_ai_sdk.schema.fingerprint import SchemaFingerprint
from psi_c_ai_sdk.schema.fingerprint import SchemaDiffCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchemaToolkit:
    """
    Unified toolkit for working with memory schema graphs.
    
    The SchemaToolkit provides a simplified interface for managing schema graphs,
    including creation, update, mutation, and visualization. It integrates
    the core schema functionality into a cohesive system that can be easily
    used in applications.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        coherence_scorer: CoherenceScorer,
        min_edge_weight: float = 0.3,
        mutation_threshold: float = 0.5,
        auto_prune: bool = True,
        max_nodes: int = 500,
        visualize_dir: Optional[str] = None
    ):
        """
        Initialize the schema toolkit.
        
        Args:
            memory_store: Memory store containing memories
            coherence_scorer: Coherence scorer for calculating relationships
            min_edge_weight: Minimum coherence score to create an edge (default: 0.3)
            mutation_threshold: Coherence threshold for triggering mutations (default: 0.5)
            auto_prune: Whether to automatically prune low-importance nodes (default: True)
            max_nodes: Maximum number of nodes in the schema graph (default: 500)
            visualize_dir: Directory to save visualizations (optional)
        """
        self.memory_store = memory_store
        self.coherence_scorer = coherence_scorer
        
        # Create schema graph
        self.schema_graph = SchemaGraph(
            memory_store=memory_store,
            coherence_scorer=coherence_scorer,
            min_edge_weight=min_edge_weight,
            auto_prune=auto_prune,
            max_nodes=max_nodes
        )
        
        # Create mutation system
        self.mutation_system = SchemaMutationSystem(
            schema_graph=self.schema_graph,
            coherence_scorer=coherence_scorer,
            memory_store=memory_store,
            mutation_threshold=mutation_threshold
        )
        
        # Create fingerprint calculator
        self.fingerprint = SchemaFingerprint(schema_graph=self.schema_graph)
        
        # Previous schema snapshot for diff calculation
        self.previous_schema = None
        
        # Set visualization directory
        self.visualize_dir = visualize_dir
        if visualize_dir and not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
            
        # Statistics tracking
        self.stats = {
            "updates": 0,
            "mutations": 0,
            "nodes_added": 0,
            "nodes_pruned": 0,
            "edges_added": 0,
            "visualization_count": 0,
            "last_update": None,
            "last_mutation": None,
            "mutation_types": {}
        }
    
    def add_memory(
        self,
        memory: Memory,
        update_schema: bool = True,
        check_mutation: bool = True
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Add a memory to the schema graph.
        
        Args:
            memory: Memory object to add
            update_schema: Whether to update the full schema after adding
            check_mutation: Whether to check if mutation is needed after adding
            
        Returns:
            Tuple of (node_id, mutation_info)
        """
        # Add the memory node
        node_id = self.schema_graph.add_memory_node(memory)
        self.stats["nodes_added"] += 1
        
        # Calculate relationships
        relationships = self.schema_graph.calculate_memory_relationships(memory)
        
        # Add edges
        for related_id, score in relationships:
            if self.schema_graph.add_edge(node_id, related_id, score):
                self.stats["edges_added"] += 1
        
        # Update schema if requested
        update_info = None
        if update_schema:
            update_info = self.update_schema(max_memories=10)
        
        # Check for mutations if requested
        mutation_info = None
        if check_mutation:
            should_mutate, reason = self.mutation_system.check_mutation_needed()
            if should_mutate:
                mutation_event = self.mutation_system.mutate_schema(trigger=f"memory_added_{reason}")
                if mutation_event:
                    mutation_info = mutation_event.to_dict()
                    self._record_mutation(mutation_event)
        
        return node_id, mutation_info
    
    def update_schema(self, max_memories: int = 100) -> Dict[str, int]:
        """
        Update the schema graph based on the current memory store.
        
        Args:
            max_memories: Maximum number of memories to process per update
            
        Returns:
            Dictionary with update statistics
        """
        # Take snapshot of current schema for diff calculation
        self.previous_schema = self._clone_schema_graph()
        
        # Update schema
        update_stats = self.schema_graph.update_schema(max_memories=max_memories)
        
        # Update toolkit stats
        self.stats["updates"] += 1
        self.stats["nodes_added"] += update_stats.get("nodes_added", 0)
        self.stats["edges_added"] += update_stats.get("edges_added", 0)
        self.stats["nodes_pruned"] += update_stats.get("nodes_pruned", 0)
        self.stats["last_update"] = datetime.now().isoformat()
        
        return update_stats
    
    def mutate_schema(self, mutation_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Perform a schema mutation.
        
        Args:
            mutation_type: Optional specific mutation type to perform
            
        Returns:
            Mutation event information if successful, None otherwise
        """
        # Convert string mutation type to enum if provided
        mutation_type_enum = None
        if mutation_type:
            try:
                mutation_type_enum = MutationType(mutation_type)
            except ValueError:
                logger.warning(f"Invalid mutation type: {mutation_type}")
                return None
        
        # Perform mutation
        if mutation_type_enum:
            mutation_event = self.mutation_system.force_mutation(mutation_type_enum)
        else:
            mutation_event = self.mutation_system.mutate_schema(trigger="manual")
            
        # Record mutation if successful
        if mutation_event:
            self._record_mutation(mutation_event)
            return mutation_event.to_dict()
            
        return None
    
    def check_mutation_needed(self) -> Tuple[bool, str]:
        """
        Check if schema mutation is needed based on coherence.
        
        Returns:
            Tuple of (should_mutate, reason)
        """
        return self.mutation_system.check_mutation_needed()
    
    def add_concept(
        self,
        label: str,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        related_memories: Optional[List[Union[str, UUID, Memory]]] = None,
        relationship_weight: float = 0.6
    ) -> str:
        """
        Add a concept node to the schema.
        
        Args:
            label: Label for the concept
            embedding: Optional embedding vector
            importance: Importance score (0-1)
            tags: Optional tags for categorization
            metadata: Optional metadata
            related_memories: Optional list of memories to connect to this concept
            relationship_weight: Weight for edges to related memories
            
        Returns:
            ID of the created concept node
        """
        # Add the concept node
        concept_id = self.schema_graph.add_concept_node(
            label=label,
            embedding=embedding,
            importance=importance,
            tags=tags,
            metadata=metadata
        )
        
        self.stats["nodes_added"] += 1
        
        # Connect to related memories if provided
        if related_memories:
            for memory_ref in related_memories:
                # Handle different types of memory references
                memory_id = None
                
                if isinstance(memory_ref, Memory):
                    memory_id = str(memory_ref.uuid)
                elif isinstance(memory_ref, UUID):
                    memory_id = str(memory_ref)
                elif isinstance(memory_ref, str):
                    memory_id = memory_ref
                
                if memory_id:
                    # Find the node for this memory
                    memory_node_id = f"memory_{memory_id}"
                    
                    # Add edge if the memory node exists
                    if memory_node_id in self.schema_graph.graph:
                        if self.schema_graph.add_edge(
                            concept_id, memory_node_id, 
                            relationship_weight,
                            edge_type="concept_memory"
                        ):
                            self.stats["edges_added"] += 1
        
        return concept_id
    
    def visualize_schema(
        self,
        filename: Optional[str] = None,
        highlight_nodes: Optional[List[str]] = None,
        max_nodes: int = 100,
        title: Optional[str] = None
    ) -> Optional[str]:
        """
        Visualize the schema graph.
        
        Args:
            filename: Optional filename to save visualization
            highlight_nodes: Optional list of node IDs to highlight
            max_nodes: Maximum number of nodes to include in visualization
            title: Optional title for the visualization
            
        Returns:
            Path to saved visualization file if successful, None otherwise
        """
        # Generate filename if not provided
        if not filename and self.visualize_dir:
            timestamp = int(time.time())
            filename = os.path.join(
                self.visualize_dir, 
                f"schema_{timestamp}.png"
            )
        
        # Generate visualization
        self.schema_graph.visualize(
            filename=filename,
            highlight_nodes=highlight_nodes,
            max_nodes=max_nodes
        )
        
        self.stats["visualization_count"] += 1
        
        return filename if os.path.exists(filename) else None
    
    def get_related_memories(
        self,
        memory: Union[Memory, str, UUID],
        min_coherence: float = 0.0,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get memories related to a given memory based on schema graph connections.
        
        Args:
            memory: Memory object, ID, or UUID to find related memories for
            min_coherence: Minimum coherence score for relationships
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries with related memory information
        """
        # Get memory ID
        memory_id = None
        if isinstance(memory, Memory):
            memory_id = str(memory.uuid)
        elif isinstance(memory, UUID):
            memory_id = str(memory)
        elif isinstance(memory, str):
            memory_id = memory
        
        if not memory_id:
            return []
            
        # Find the node for this memory
        memory_node_id = f"memory_{memory_id}"
        
        if memory_node_id not in self.schema_graph.graph:
            return []
            
        # Get neighbors
        neighbors = list(self.schema_graph.graph.neighbors(memory_node_id))
        
        related_memories = []
        for neighbor in neighbors:
            # Only include memory nodes
            if not neighbor.startswith("memory_"):
                continue
                
            # Get edge weight
            weight = self.schema_graph.graph[memory_node_id][neighbor].get("weight", 0.0)
            
            # Skip if below minimum coherence
            if weight < min_coherence:
                continue
                
            # Get memory information
            related_id = self.schema_graph.graph.nodes[neighbor].get("memory_id")
            if related_id:
                memory_obj = self.memory_store.get_memory(UUID(related_id))
                if memory_obj:
                    related_memories.append({
                        "memory_id": related_id,
                        "content": memory_obj.content,
                        "coherence": weight,
                        "importance": memory_obj.importance,
                        "created_at": memory_obj.created_at.isoformat() if memory_obj.created_at else None
                    })
        
        # Sort by coherence (highest first) and limit results
        related_memories.sort(key=lambda x: x["coherence"], reverse=True)
        return related_memories[:max_results]
    
    def get_concept_hierarchy(self) -> Dict[str, Any]:
        """
        Get the concept hierarchy from the schema graph.
        
        Returns:
            Dictionary representing the concept hierarchy
        """
        # Get all concept nodes
        concept_nodes = [
            node for node in self.schema_graph.graph.nodes
            if self.schema_graph.graph.nodes[node].get("node_type") == "concept"
        ]
        
        # Build hierarchy
        hierarchy = {"concepts": []}
        
        for node_id in concept_nodes:
            node_data = self.schema_graph.graph.nodes[node_id]
            
            # Get parent relationships
            parents = []
            children = []
            
            for neighbor in self.schema_graph.graph.neighbors(node_id):
                edge_data = self.schema_graph.graph[node_id][neighbor]
                
                if edge_data.get("edge_type") == "parent_child":
                    if node_id < neighbor:  # This node is the parent
                        children.append(neighbor)
                    else:  # This node is the child
                        parents.append(neighbor)
            
            # Get connected memories
            memory_connections = []
            for neighbor in self.schema_graph.graph.neighbors(node_id):
                if neighbor.startswith("memory_"):
                    memory_id = self.schema_graph.graph.nodes[neighbor].get("memory_id")
                    if memory_id:
                        memory_connections.append(memory_id)
            
            concept_info = {
                "id": node_id,
                "label": node_data.get("label", ""),
                "importance": node_data.get("importance", 0.5),
                "tags": node_data.get("tags", []),
                "parents": parents,
                "children": children,
                "memories": memory_connections,
                "created_at": node_data.get("metadata", {}).get("created_at")
            }
            
            hierarchy["concepts"].append(concept_info)
        
        return hierarchy
    
    def calculate_schema_diff(self) -> Dict[str, Any]:
        """
        Calculate differences between current and previous schema.
        
        Returns:
            Dictionary with diff information
        """
        if not self.previous_schema:
            return {"error": "No previous schema available"}
            
        diff_calculator = SchemaDiffCalculator(
            current_schema=self.schema_graph,
            previous_schema=self.previous_schema
        )
        
        return diff_calculator.calculate_diff()
    
    def get_schema_fingerprint(self) -> str:
        """
        Get a unique fingerprint of the current schema.
        
        Returns:
            SHA-256 hash representing the schema state
        """
        return self.fingerprint.compute_fingerprint()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the schema toolkit.
        
        Returns:
            Dictionary with schema statistics
        """
        # Get schema graph stats
        graph_stats = self.schema_graph.get_stats()
        
        # Get mutation stats
        mutation_stats = self.mutation_system.get_mutation_stats()
        
        # Combine stats
        combined_stats = {
            **self.stats,
            "graph": graph_stats,
            "mutation": mutation_stats,
            "fingerprint": self.get_schema_fingerprint()
        }
        
        return combined_stats
    
    def get_mutation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of schema mutations.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of mutation event records
        """
        return self.mutation_system.get_mutation_history(limit)
    
    def _record_mutation(self, mutation_event: MutationEvent) -> None:
        """
        Record a mutation event in statistics.
        
        Args:
            mutation_event: Mutation event to record
        """
        self.stats["mutations"] += 1
        self.stats["last_mutation"] = datetime.now().isoformat()
        
        # Record mutation type
        mutation_type = mutation_event.mutation_type.value
        self.stats["mutation_types"][mutation_type] = self.stats["mutation_types"].get(mutation_type, 0) + 1
    
    def _clone_schema_graph(self) -> SchemaGraph:
        """
        Create a deep copy of the current schema graph.
        
        Returns:
            Cloned schema graph
        """
        # Create new schema graph with same parameters
        cloned = SchemaGraph(
            memory_store=self.memory_store,
            coherence_scorer=self.coherence_scorer,
            min_edge_weight=self.schema_graph.min_edge_weight,
            auto_prune=self.schema_graph.auto_prune,
            max_nodes=self.schema_graph.max_nodes
        )
        
        # Copy nodes
        for node, data in self.schema_graph.graph.nodes(data=True):
            cloned.graph.add_node(node, **data)
            
        # Copy edges
        for u, v, data in self.schema_graph.graph.edges(data=True):
            cloned.graph.add_edge(u, v, **data)
            
        # Copy tracked memories
        cloned.tracked_memory_ids = set(self.schema_graph.tracked_memory_ids)
        
        # Copy metrics
        cloned.node_count = self.schema_graph.node_count
        cloned.edge_count = self.schema_graph.edge_count
        cloned.last_update_time = self.schema_graph.last_update_time
        
        return cloned 