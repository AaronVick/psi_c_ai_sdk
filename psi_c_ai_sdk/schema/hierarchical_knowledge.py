#!/usr/bin/env python3
"""
Hierarchical Knowledge Management for ΨC-AI SDK

This module implements a comprehensive system for managing hierarchical knowledge structures,
including concept taxonomies, knowledge categorization, and hierarchical relationships
between memories and beliefs. It integrates with the schema system to provide
advanced knowledge organization capabilities.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from uuid import UUID

import networkx as nx
import numpy as np
from dataclasses import dataclass, field

from psi_c_ai_sdk.schema.schema import SchemaGraph, SchemaNode, SchemaEdge
from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer
from psi_c_ai_sdk.embedding.embedding import EmbeddingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeCategory:
    """A category in the knowledge hierarchy."""
    
    id: str
    name: str
    description: str = ""
    parent_id: Optional[str] = None
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "importance": self.importance,
            "metadata": self.metadata
        }


@dataclass
class HierarchicalRelationship:
    """A hierarchical relationship between knowledge entities."""
    
    source_id: str
    target_id: str
    relationship_type: str  # e.g., "is_a", "part_of", "subclass_of", "instance_of"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
            "metadata": self.metadata
        }


class HierarchicalKnowledgeManager:
    """
    Manages hierarchical knowledge structures in the system.
    
    The HierarchicalKnowledgeManager provides capabilities for organizing knowledge
    in hierarchical structures, managing taxonomies, categorizing information, and
    maintaining structured relationships between concepts and memories.
    """
    
    def __init__(
        self,
        schema_graph: SchemaGraph,
        memory_store: Optional[MemoryStore] = None,
        coherence_scorer: Optional[CoherenceScorer] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        max_hierarchy_depth: int = 10,
        min_relationship_weight: float = 0.3
    ):
        """
        Initialize the hierarchical knowledge manager.
        
        Args:
            schema_graph: Schema graph to use for knowledge representation
            memory_store: Optional memory store for memory integration
            coherence_scorer: Optional coherence scorer for relationship weighting
            embedding_engine: Optional embedding engine for semantic search
            max_hierarchy_depth: Maximum depth for hierarchy traversal
            min_relationship_weight: Minimum weight for hierarchical relationships
        """
        self.schema_graph = schema_graph
        self.memory_store = memory_store
        self.coherence_scorer = coherence_scorer
        self.embedding_engine = embedding_engine
        self.max_hierarchy_depth = max_hierarchy_depth
        self.min_relationship_weight = min_relationship_weight
        
        # Caches for efficient retrieval
        self.categories: Dict[str, KnowledgeCategory] = {}
        self.cached_hierarchies: Dict[str, Dict[str, Any]] = {}
        
        # Statistics and metrics
        self.stats = {
            "num_categories": 0,
            "num_hierarchical_relationships": 0,
            "max_observed_depth": 0,
            "last_update_time": None,
            "category_counts": {},
            "relationship_type_counts": {}
        }
    
    def create_category(
        self,
        name: str,
        description: str = "",
        parent_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new knowledge category.
        
        Args:
            name: Name of the category
            description: Description of the category
            parent_id: Optional parent category ID
            importance: Importance of the category (0.0-1.0)
            metadata: Optional additional metadata
            
        Returns:
            ID of the created category
        """
        # Generate ID
        category_id = f"category_{int(time.time())}_{len(self.categories)}"
        
        # Create category
        category = KnowledgeCategory(
            id=category_id,
            name=name,
            description=description,
            parent_id=parent_id,
            importance=importance,
            metadata=metadata or {}
        )
        
        # Add to schema graph as a concept node
        self.schema_graph.add_concept_node(
            label=name,
            importance=importance,
            tags=["category"],
            metadata={
                "description": description,
                "category_id": category_id,
                "created_at": datetime.now().isoformat()
            }
        )
        
        # If parent exists, create hierarchical relationship
        if parent_id and parent_id in self.categories:
            parent_node_id = f"concept_category_{parent_id}"
            child_node_id = f"concept_category_{category_id}"
            
            # Add edge
            self.schema_graph.add_edge(
                source_id=parent_node_id,
                target_id=child_node_id,
                weight=1.0,
                edge_type="parent_child",
                metadata={"relationship_type": "is_a"}
            )
            
            # Update stats
            self.stats["num_hierarchical_relationships"] += 1
            self.stats["relationship_type_counts"]["is_a"] = (
                self.stats["relationship_type_counts"].get("is_a", 0) + 1
            )
        
        # Store category
        self.categories[category_id] = category
        
        # Update stats
        self.stats["num_categories"] += 1
        self.stats["last_update_time"] = datetime.now().isoformat()
        
        return category_id
    
    def add_concept_to_hierarchy(
        self,
        concept_id: str,
        parent_id: str,
        relationship_type: str = "is_a",
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a concept to the knowledge hierarchy.
        
        Args:
            concept_id: ID of the concept to add
            parent_id: ID of the parent concept
            relationship_type: Type of hierarchical relationship
            weight: Weight of the relationship
            metadata: Optional additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Verify nodes exist
        if concept_id not in self.schema_graph.graph.nodes:
            logger.warning(f"Concept {concept_id} not found in schema graph")
            return False
            
        if parent_id not in self.schema_graph.graph.nodes:
            logger.warning(f"Parent concept {parent_id} not found in schema graph")
            return False
        
        # Add hierarchical edge
        self.schema_graph.add_edge(
            source_id=parent_id,
            target_id=concept_id,
            weight=weight,
            edge_type="hierarchical",
            metadata={
                "relationship_type": relationship_type,
                **(metadata or {})
            }
        )
        
        # Clear cache
        self.cached_hierarchies = {}
        
        # Update stats
        self.stats["num_hierarchical_relationships"] += 1
        self.stats["relationship_type_counts"][relationship_type] = (
            self.stats["relationship_type_counts"].get(relationship_type, 0) + 1
        )
        self.stats["last_update_time"] = datetime.now().isoformat()
        
        return True
    
    def get_concept_hierarchy(self, root_id: Optional[str] = None, max_depth: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the concept hierarchy starting from a specific root.
        
        Args:
            root_id: Optional root concept ID (None for complete hierarchy)
            max_depth: Maximum depth to traverse (None for max_hierarchy_depth)
            
        Returns:
            Dictionary representing the concept hierarchy
        """
        # Use cache if available
        cache_key = f"{root_id}_{max_depth}"
        if cache_key in self.cached_hierarchies:
            return self.cached_hierarchies[cache_key]
        
        # Set default max depth
        if max_depth is None:
            max_depth = self.max_hierarchy_depth
            
        # Get all concept nodes
        if root_id is None:
            # Find top-level concepts (no incoming hierarchical edges)
            root_nodes = []
            for node in self.schema_graph.graph.nodes:
                if self.schema_graph.graph.nodes[node].get("node_type") == "concept":
                    has_parent = False
                    for _, target, data in self.schema_graph.graph.in_edges(node, data=True):
                        if data.get("edge_type") in ["hierarchical", "parent_child"]:
                            has_parent = True
                            break
                    
                    if not has_parent:
                        root_nodes.append(node)
        else:
            # Use specified root
            if root_id not in self.schema_graph.graph.nodes:
                logger.warning(f"Root concept {root_id} not found in schema graph")
                return {"concepts": []}
            
            root_nodes = [root_id]
            
        # Build hierarchy
        hierarchy = {"concepts": []}
        
        # Process each root node
        for root_node in root_nodes:
            hierarchy["concepts"].append(self._build_concept_subtree(root_node, max_depth))
            
        # Update stats
        max_depth_found = self._find_max_depth(hierarchy)
        if max_depth_found > self.stats["max_observed_depth"]:
            self.stats["max_observed_depth"] = max_depth_found
            
        # Cache result
        self.cached_hierarchies[cache_key] = hierarchy
        
        return hierarchy
    
    def _build_concept_subtree(self, node_id: str, max_depth: int, current_depth: int = 0) -> Dict[str, Any]:
        """
        Recursively build a concept subtree.
        
        Args:
            node_id: Current node ID
            max_depth: Maximum depth to traverse
            current_depth: Current depth in traversal
            
        Returns:
            Dictionary representing the concept subtree
        """
        # Get node data
        node_data = self.schema_graph.graph.nodes[node_id]
        
        # Initialize concept info
        concept_info = {
            "id": node_id,
            "label": node_data.get("label", ""),
            "importance": node_data.get("importance", 0.5),
            "tags": node_data.get("tags", []),
            "created_at": node_data.get("metadata", {}).get("created_at"),
            "children": []
        }
        
        # Stop recursion if max depth reached
        if current_depth >= max_depth:
            return concept_info
            
        # Get child relationships
        for _, child_id, edge_data in self.schema_graph.graph.out_edges(node_id, data=True):
            # Check if hierarchical relationship
            if edge_data.get("edge_type") in ["hierarchical", "parent_child"]:
                # Recursively build child subtree
                child_info = self._build_concept_subtree(child_id, max_depth, current_depth + 1)
                concept_info["children"].append(child_info)
        
        return concept_info
    
    def _find_max_depth(self, hierarchy: Dict[str, Any]) -> int:
        """
        Find the maximum depth in a hierarchy.
        
        Args:
            hierarchy: Hierarchy dictionary
            
        Returns:
            Maximum depth found
        """
        max_depth = 0
        
        def process_node(node, depth):
            nonlocal max_depth
            if depth > max_depth:
                max_depth = depth
                
            for child in node.get("children", []):
                process_node(child, depth + 1)
                
        for concept in hierarchy.get("concepts", []):
            process_node(concept, 1)
            
        return max_depth
    
    def categorize_memory(
        self,
        memory: Union[Memory, str, UUID],
        category_id: str,
        confidence: float = 1.0
    ) -> bool:
        """
        Categorize a memory within the knowledge hierarchy.
        
        Args:
            memory: Memory object or ID to categorize
            category_id: Category ID to assign
            confidence: Confidence of the categorization (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        # Get memory
        if isinstance(memory, (str, UUID)):
            if not self.memory_store:
                logger.warning("No memory store provided for memory retrieval")
                return False
                
            memory_obj = self.memory_store.get_memory(str(memory))
            if not memory_obj:
                logger.warning(f"Memory {memory} not found")
                return False
        else:
            memory_obj = memory
            
        # Get category
        if category_id not in self.categories:
            logger.warning(f"Category {category_id} not found")
            return False
            
        # Get memory node ID
        memory_node_id = f"memory_{memory_obj.uuid}"
        
        # Get category node ID
        category_node_id = f"concept_category_{category_id}"
        
        # Add memory to schema if not present
        if memory_node_id not in self.schema_graph.graph.nodes:
            self.schema_graph.add_memory_node(memory_obj)
            
        # Add category to schema if not present
        if category_node_id not in self.schema_graph.graph.nodes:
            category = self.categories[category_id]
            self.schema_graph.add_concept_node(
                label=category.name,
                importance=category.importance,
                tags=["category"],
                metadata={
                    "description": category.description,
                    "category_id": category_id,
                    "created_at": datetime.now().isoformat()
                }
            )
            
        # Add categorization edge
        self.schema_graph.add_edge(
            source_id=category_node_id,
            target_id=memory_node_id,
            weight=confidence,
            edge_type="categorization",
            metadata={
                "confidence": confidence,
                "categorization_time": datetime.now().isoformat()
            }
        )
        
        # Update stats
        self.stats["category_counts"][category_id] = (
            self.stats["category_counts"].get(category_id, 0) + 1
        )
        self.stats["last_update_time"] = datetime.now().isoformat()
        
        return True
    
    def get_memories_in_category(self, category_id: str, min_confidence: float = 0.0) -> List[str]:
        """
        Get all memories in a specific category.
        
        Args:
            category_id: Category ID to query
            min_confidence: Minimum confidence for categorization
            
        Returns:
            List of memory IDs in the category
        """
        # Get category node ID
        category_node_id = f"concept_category_{category_id}"
        
        # Check if category node exists
        if category_node_id not in self.schema_graph.graph.nodes:
            logger.warning(f"Category node {category_node_id} not found")
            return []
            
        # Get all memory nodes connected to this category
        memory_ids = []
        
        for _, memory_node_id, edge_data in self.schema_graph.graph.out_edges(category_node_id, data=True):
            # Check if categorization edge with sufficient confidence
            if (edge_data.get("edge_type") == "categorization" and 
                edge_data.get("metadata", {}).get("confidence", 0.0) >= min_confidence):
                
                # Extract memory ID from node ID
                if memory_node_id.startswith("memory_"):
                    memory_id = self.schema_graph.graph.nodes[memory_node_id].get("memory_id")
                    if memory_id:
                        memory_ids.append(memory_id)
        
        return memory_ids
    
    def get_category_hierarchy(self) -> Dict[str, Any]:
        """
        Get the complete category hierarchy.
        
        Returns:
            Dictionary representing the category hierarchy
        """
        # Build hierarchy
        hierarchy = {"categories": []}
        
        # Get all root categories (no parent)
        root_categories = [cat_id for cat_id, cat in self.categories.items() if cat.parent_id is None]
        
        # Process each root category
        for category_id in root_categories:
            hierarchy["categories"].append(self._build_category_subtree(category_id))
            
        return hierarchy
    
    def _build_category_subtree(self, category_id: str, current_depth: int = 0) -> Dict[str, Any]:
        """
        Recursively build a category subtree.
        
        Args:
            category_id: Current category ID
            current_depth: Current depth in traversal
            
        Returns:
            Dictionary representing the category subtree
        """
        # Get category
        category = self.categories.get(category_id)
        if not category:
            return {}
            
        # Initialize category info
        category_info = {
            "id": category.id,
            "name": category.name,
            "description": category.description,
            "importance": category.importance,
            "metadata": category.metadata,
            "children": []
        }
        
        # Stop recursion if max depth reached
        if current_depth >= self.max_hierarchy_depth:
            return category_info
            
        # Get child categories
        child_categories = [cat_id for cat_id, cat in self.categories.items() if cat.parent_id == category_id]
        
        # Process each child
        for child_id in child_categories:
            child_info = self._build_category_subtree(child_id, current_depth + 1)
            category_info["children"].append(child_info)
        
        return category_info
    
    def search_hierarchical_knowledge(self, query: str, top_k: int = 10, use_embeddings: bool = True) -> List[Dict[str, Any]]:
        """
        Search for concepts or memories in the knowledge hierarchy.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            use_embeddings: Whether to use semantic search with embeddings (if available)
            
        Returns:
            List of matching items with relevance scores
        """
        results = []
        
        # Use semantic search if embedding engine is available and embeddings are enabled
        if use_embeddings and self.embedding_engine:
            return self._semantic_search(query, top_k)
        else:
            # Fall back to simple text matching
            return self._text_search(query, top_k)
    
    def _semantic_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of matching items with relevance scores
        """
        # Get query embedding
        query_embedding = self.embedding_engine.get_embedding(query)
        if query_embedding is None:
            logger.warning(f"Failed to get embedding for query: {query}")
            return self._text_search(query, top_k)
        
        # Collect embeddings and data for concepts
        concept_data = []
        for node, data in self.schema_graph.graph.nodes(data=True):
            if data.get("node_type") == "concept":
                label = data.get("label", "")
                description = data.get("metadata", {}).get("description", "")
                content = f"{label}: {description}"
                
                # Get or create embedding
                embedding = self.embedding_engine.get_embedding(content)
                if embedding is not None:
                    concept_data.append({
                        "id": node,
                        "type": "concept",
                        "content": label,
                        "embedding": embedding
                    })
        
        # Collect embeddings and data for memories
        memory_data = []
        if self.memory_store:
            for memory in self.memory_store.get_all_memories():
                # Use existing embedding if available, otherwise create one
                embedding = getattr(memory, "embedding", None)
                if embedding is None and memory.content:
                    embedding = self.embedding_engine.get_embedding(memory.content)
                
                if embedding is not None:
                    memory_data.append({
                        "id": str(memory.uuid),
                        "type": "memory",
                        "content": memory.content[:100] + ("..." if len(memory.content) > 100 else ""),
                        "embedding": embedding
                    })
        
        # Combine all data
        all_data = concept_data + memory_data
        
        # Calculate similarity scores
        results = []
        for item in all_data:
            similarity = self.coherence_scorer.compute_cosine_similarity(
                query_embedding, item["embedding"]
            )
            results.append({
                "id": item["id"],
                "type": item["type"],
                "content": item["content"],
                "relevance": float(similarity)
            })
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]
    
    def _text_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform simple text-based search.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of matching items with relevance scores
        """
        results = []
        
        # Search concepts
        for node, data in self.schema_graph.graph.nodes(data=True):
            if data.get("node_type") == "concept":
                label = data.get("label", "")
                if query.lower() in label.lower():
                    # Add to results
                    results.append({
                        "id": node,
                        "type": "concept",
                        "content": label,
                        "relevance": 0.5 + (0.5 * (len(query) / len(label) if len(label) > 0 else 0))
                    })
        
        # Search memories via memory store if available
        if self.memory_store:
            # This is a simplistic approach - real implementation would use proper search
            memory_count = 0
            for memory in self.memory_store.get_all_memories():
                if memory_count >= top_k * 2:  # Limit search to avoid performance issues
                    break
                    
                content = memory.content
                if query.lower() in content.lower():
                    # Add to results
                    results.append({
                        "id": str(memory.uuid),
                        "type": "memory",
                        "content": content[:100] + ("..." if len(content) > 100 else ""),
                        "relevance": 0.5 + (0.5 * (len(query) / len(content) if len(content) > 0 else 0))
                    })
                memory_count += 1
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the hierarchical knowledge structure.
        
        Returns:
            Dictionary with knowledge statistics
        """
        # Update stats with current counts
        self.stats["num_categories"] = len(self.categories)
        
        # Count hierarchical relationships
        hierarchical_count = 0
        relationship_counts = {}
        
        for _, _, edge_data in self.schema_graph.graph.edges(data=True):
            if edge_data.get("edge_type") in ["hierarchical", "parent_child"]:
                hierarchical_count += 1
                
                # Count by relationship type
                rel_type = edge_data.get("metadata", {}).get("relationship_type", "unspecified")
                relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        self.stats["num_hierarchical_relationships"] = hierarchical_count
        self.stats["relationship_type_counts"] = relationship_counts
        
        # Get additional metrics
        stats = {
            **self.stats,
            "num_concepts": sum(1 for _, data in self.schema_graph.graph.nodes(data=True) 
                              if data.get("node_type") == "concept"),
            "num_memories": sum(1 for _, data in self.schema_graph.graph.nodes(data=True) 
                              if data.get("node_type") == "memory"),
            "avg_children_per_concept": self._calculate_avg_children()
        }
        
        return stats
    
    def _calculate_avg_children(self) -> float:
        """
        Calculate average number of children per concept.
        
        Returns:
            Average number of children
        """
        child_counts = []
        
        for node, data in self.schema_graph.graph.nodes(data=True):
            if data.get("node_type") == "concept":
                # Count children
                child_count = 0
                for _, _, edge_data in self.schema_graph.graph.out_edges(node, data=True):
                    if edge_data.get("edge_type") in ["hierarchical", "parent_child"]:
                        child_count += 1
                        
                child_counts.append(child_count)
        
        if not child_counts:
            return 0.0
            
        return sum(child_counts) / len(child_counts)
    
    def visualize_hierarchy(
        self,
        filename: Optional[str] = None,
        root_id: Optional[str] = None,
        max_depth: int = 3
    ) -> Optional[str]:
        """
        Visualize the knowledge hierarchy.
        
        Args:
            filename: Optional filename to save visualization
            root_id: Optional root node ID (None for complete hierarchy)
            max_depth: Maximum depth to visualize
            
        Returns:
            Path to saved visualization file if filename provided, None otherwise
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            logger.error("Matplotlib is required for visualization. Install with 'pip install matplotlib'")
            return None
        
        # Create a new graph for visualization to avoid modifying the original
        vis_graph = nx.DiGraph()
        
        # Get hierarchy to visualize
        hierarchy = self.get_concept_hierarchy(root_id=root_id, max_depth=max_depth)
        
        # Track nodes to include based on the hierarchy
        nodes_to_include = set()
        node_colors = {}
        node_sizes = {}
        node_labels = {}
        edge_labels = {}
        
        # Process hierarchy to build visualization graph
        def process_hierarchy_node(node_data, level=0, parent_id=None):
            node_id = node_data["id"]
            nodes_to_include.add(node_id)
            
            # Set node properties
            label = node_data.get("label", "")
            importance = node_data.get("importance", 0.5)
            
            # Add node to visualization graph
            vis_graph.add_node(node_id)
            
            # Store node properties
            node_labels[node_id] = label
            node_sizes[node_id] = 300 + (importance * 200)  # Size based on importance
            
            # Assign color based on level
            cmap = plt.cm.viridis
            node_colors[node_id] = cmap(level / max(max_depth, 1))
            
            # Add edge from parent if exists
            if parent_id:
                vis_graph.add_edge(parent_id, node_id)
                
                # Get relationship type if available
                relationship = "is_a"  # Default
                for _, _, edge_data in self.schema_graph.graph.edges(data=True):
                    if edge_data.get("source_id") == parent_id and edge_data.get("target_id") == node_id:
                        relationship = edge_data.get("metadata", {}).get("relationship_type", "is_a")
                        break
                
                edge_labels[(parent_id, node_id)] = relationship
            
            # Process children
            if level < max_depth:
                for child in node_data.get("children", []):
                    process_hierarchy_node(child, level + 1, node_id)
        
        # Process each root in the hierarchy
        for concept in hierarchy.get("concepts", []):
            process_hierarchy_node(concept)
        
        # Create subgraph with only included nodes
        vis_subgraph = vis_graph.subgraph(nodes_to_include)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(vis_subgraph, seed=42)  # Consistent layout with seed
        
        # Draw nodes
        nx.draw_networkx_nodes(
            vis_subgraph, 
            pos,
            node_size=[node_sizes.get(n, 300) for n in vis_subgraph.nodes()],
            node_color=[node_colors.get(n, (0.5, 0.5, 0.5, 1.0)) for n in vis_subgraph.nodes()]
        )
        
        # Draw edges
        nx.draw_networkx_edges(vis_subgraph, pos, arrows=True, arrowsize=15)
        
        # Draw node labels
        nx.draw_networkx_labels(
            vis_subgraph,
            pos,
            labels={n: node_labels.get(n, str(n)) for n in vis_subgraph.nodes()}
        )
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(
            vis_subgraph,
            pos,
            edge_labels=edge_labels,
            font_size=8
        )
        
        plt.title("Hierarchical Knowledge Graph")
        plt.axis("off")
        
        # Save or show visualization
        if filename:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
            logger.info(f"Visualization saved to {filename}")
            return filename
        else:
            plt.show()
            plt.close()
            return None
    
    def find_related_concepts(
        self, 
        concept_id: str, 
        relationship_types: Optional[List[str]] = None,
        max_results: int = 10,
        include_similarity: bool = True,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find concepts related to a given concept through hierarchical relationships
        and semantic similarity.
        
        Args:
            concept_id: ID of the concept to find relations for
            relationship_types: Optional list of relationship types to consider
                               (None means all relationship types)
            max_results: Maximum number of results to return
            include_similarity: Whether to include semantically similar concepts
                               that aren't directly connected
            similarity_threshold: Minimum similarity score for semantic relations
            
        Returns:
            List of related concepts with relationship information
        """
        if concept_id not in self.schema_graph.graph.nodes:
            logger.warning(f"Concept {concept_id} not found in schema graph")
            return []
        
        # Default to all relationship types if none specified
        if relationship_types is None:
            relationship_types = ["is_a", "part_of", "has_part", "related_to", "instance_of"]
        
        # Get concept data for reference
        concept_data = self.schema_graph.graph.nodes[concept_id]
        concept_label = concept_data.get("label", "")
        
        related_concepts = []
        
        # Find directly connected concepts (both incoming and outgoing)
        for source, target, edge_data in self.schema_graph.graph.edges(data=True):
            # Check if this edge involves our concept and is a hierarchical relationship
            is_hierarchical = edge_data.get("edge_type") in ["hierarchical", "parent_child"]
            rel_type = edge_data.get("metadata", {}).get("relationship_type")
            
            if not is_hierarchical or rel_type not in relationship_types:
                continue
                
            if source == concept_id and target != concept_id:
                # Outgoing relationship
                target_data = self.schema_graph.graph.nodes[target]
                if target_data.get("node_type") == "concept":
                    related_concepts.append({
                        "id": target,
                        "type": "concept",
                        "label": target_data.get("label", ""),
                        "relationship": rel_type,
                        "direction": "outgoing",
                        "weight": edge_data.get("weight", 1.0),
                        "relevance": edge_data.get("weight", 1.0)
                    })
            elif target == concept_id and source != concept_id:
                # Incoming relationship
                source_data = self.schema_graph.graph.nodes[source]
                if source_data.get("node_type") == "concept":
                    # Invert the relationship type for proper representation
                    inv_rel_type = self._invert_relationship_type(rel_type)
                    related_concepts.append({
                        "id": source,
                        "type": "concept",
                        "label": source_data.get("label", ""),
                        "relationship": inv_rel_type,
                        "direction": "incoming",
                        "weight": edge_data.get("weight", 1.0),
                        "relevance": edge_data.get("weight", 1.0)
                    })
        
        # Find semantically similar concepts if requested and embedding engine is available
        if include_similarity and self.embedding_engine and concept_label:
            # Get embedding for our concept
            concept_embedding = self.embedding_engine.get_embedding(concept_label)
            if concept_embedding is not None:
                # Find similar concepts
                for node, data in self.schema_graph.graph.nodes(data=True):
                    # Skip if not a concept or is the same concept
                    if data.get("node_type") != "concept" or node == concept_id:
                        continue
                    
                    # Skip if already found through direct relationships
                    if any(rel["id"] == node for rel in related_concepts):
                        continue
                    
                    label = data.get("label", "")
                    if not label:
                        continue
                    
                    # Get embedding for candidate concept
                    embedding = self.embedding_engine.get_embedding(label)
                    if embedding is None:
                        continue
                    
                    # Calculate similarity
                    if self.coherence_scorer:
                        similarity = self.coherence_scorer.compute_cosine_similarity(
                            concept_embedding, embedding
                        )
                        
                        # Add if above threshold
                        if similarity >= similarity_threshold:
                            related_concepts.append({
                                "id": node,
                                "type": "concept",
                                "label": label,
                                "relationship": "semantically_related",
                                "direction": "none",
                                "weight": 1.0,
                                "relevance": float(similarity)
                            })
        
        # Sort by relevance and limit results
        related_concepts.sort(key=lambda x: x["relevance"], reverse=True)
        return related_concepts[:max_results]
    
    def _invert_relationship_type(self, relationship_type: str) -> str:
        """
        Invert a relationship type (e.g., 'is_a' -> 'has_instance', 'part_of' -> 'has_part').
        
        Args:
            relationship_type: Relationship type to invert
            
        Returns:
            Inverted relationship type
        """
        inversion_map = {
            "is_a": "has_instance",
            "part_of": "has_part",
            "has_part": "part_of",
            "instance_of": "has_instance",
            "has_instance": "instance_of",
            "subclass_of": "has_subclass",
            "has_subclass": "subclass_of",
            "related_to": "related_to"  # Symmetric relationship
        }
        
        return inversion_map.get(relationship_type, f"inverse_{relationship_type}")
    
    def infer_relationships(
        self,
        source_id: str,
        target_id: str,
        max_path_length: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Infer possible relationships between two concepts by traversing the hierarchy.
        
        Args:
            source_id: ID of the source concept
            target_id: ID of the target concept
            max_path_length: Maximum path length to explore
            
        Returns:
            List of inferred relationships with path information
        """
        if source_id not in self.schema_graph.graph.nodes:
            logger.warning(f"Source concept {source_id} not found in schema graph")
            return []
            
        if target_id not in self.schema_graph.graph.nodes:
            logger.warning(f"Target concept {target_id} not found in schema graph")
            return []
            
        # Get source and target data
        source_data = self.schema_graph.graph.nodes[source_id]
        target_data = self.schema_graph.graph.nodes[target_id]
        
        # Verify both are concepts
        if source_data.get("node_type") != "concept" or target_data.get("node_type") != "concept":
            logger.warning("Source and target must both be concepts")
            return []
            
        # Find paths using graph algorithms
        try:
            # Create a subgraph of only hierarchical edges for traversal
            hierarchical_edges = [
                (u, v) for u, v, data in self.schema_graph.graph.edges(data=True)
                if data.get("edge_type") in ["hierarchical", "parent_child"]
            ]
            
            subgraph = nx.DiGraph()
            subgraph.add_nodes_from(self.schema_graph.graph.nodes(data=True))
            subgraph.add_edges_from(hierarchical_edges)
            
            # Find all simple paths up to max length
            all_paths = []
            for path in nx.all_simple_paths(subgraph, source=source_id, target=target_id, cutoff=max_path_length):
                all_paths.append(path)
                
            # If no direct paths, try bidirectional exploration for common ancestors
            if not all_paths and max_path_length >= 2:
                # Find common ancestors
                common_ancestors = self._find_common_ancestors(source_id, target_id)
                
                # For each common ancestor, find paths from source and to target
                for ancestor_id in common_ancestors:
                    # Find path from source to ancestor
                    for source_to_ancestor in nx.all_simple_paths(
                        subgraph, source=source_id, target=ancestor_id, cutoff=max_path_length//2
                    ):
                        # Find path from ancestor to target
                        for ancestor_to_target in nx.all_simple_paths(
                            subgraph, source=ancestor_id, target=target_id, cutoff=max_path_length//2
                        ):
                            # Combine paths (removing duplicate ancestor)
                            combined_path = source_to_ancestor + ancestor_to_target[1:]
                            if len(combined_path) <= max_path_length:
                                all_paths.append(combined_path)
                
            # Format the results
            inferred_relationships = []
            
            for path in all_paths:
                # Extract relationship chain
                relationship_chain = []
                for i in range(len(path) - 1):
                    source_node = path[i]
                    target_node = path[i + 1]
                    
                    # Find edge data
                    relationship_data = None
                    for _, _, edge_data in self.schema_graph.graph.edges(data=True):
                        if (edge_data.get("source_id") == source_node and 
                            edge_data.get("target_id") == target_node):
                            relationship_data = edge_data
                            break
                    
                    if relationship_data:
                        rel_type = relationship_data.get("metadata", {}).get("relationship_type", "related_to")
                        relationship_chain.append({
                            "source": source_node,
                            "source_label": self.schema_graph.graph.nodes[source_node].get("label", ""),
                            "target": target_node,
                            "target_label": self.schema_graph.graph.nodes[target_node].get("label", ""),
                            "relationship": rel_type,
                            "weight": relationship_data.get("weight", 1.0)
                        })
                
                # Calculate overall path strength (product of weights)
                path_strength = 1.0
                for rel in relationship_chain:
                    path_strength *= rel["weight"]
                
                # Generate inferred relationship description
                if relationship_chain:
                    inferred_relationships.append({
                        "path": path,
                        "relationship_chain": relationship_chain,
                        "path_length": len(path) - 1,
                        "path_strength": path_strength
                    })
            
            # Sort by path strength
            inferred_relationships.sort(key=lambda x: x["path_strength"], reverse=True)
            return inferred_relationships
            
        except Exception as e:
            logger.error(f"Error inferring relationships: {str(e)}")
            return []
    
    def _find_common_ancestors(self, node1_id: str, node2_id: str) -> List[str]:
        """
        Find common ancestors of two nodes in the hierarchy.
        
        Args:
            node1_id: ID of the first node
            node2_id: ID of the second node
            
        Returns:
            List of common ancestor IDs
        """
        # Create a subgraph of only hierarchical edges
        hierarchical_edges = [
            (u, v) for u, v, data in self.schema_graph.graph.edges(data=True)
            if data.get("edge_type") in ["hierarchical", "parent_child"]
        ]
        
        subgraph = nx.DiGraph()
        subgraph.add_nodes_from(self.schema_graph.graph.nodes(data=True))
        subgraph.add_edges_from(hierarchical_edges)
        
        # Find ancestors of each node
        ancestors1 = set(nx.ancestors(subgraph, node1_id))
        ancestors1.add(node1_id)  # Include self for potential direct relationships
        
        ancestors2 = set(nx.ancestors(subgraph, node2_id))
        ancestors2.add(node2_id)  # Include self for potential direct relationships
        
        # Find common ancestors
        common_ancestors = ancestors1.intersection(ancestors2)
        
        # Sort by distance from the nodes (assuming closer ancestors are more relevant)
        sorted_ancestors = []
        for ancestor_id in common_ancestors:
            # Calculate sum of distances (shorter is better)
            try:
                dist1 = nx.shortest_path_length(subgraph, source=ancestor_id, target=node1_id)
            except nx.NetworkXNoPath:
                dist1 = float('inf')
                
            try:
                dist2 = nx.shortest_path_length(subgraph, source=ancestor_id, target=node2_id)
            except nx.NetworkXNoPath:
                dist2 = float('inf')
                
            total_dist = dist1 + dist2
            if total_dist < float('inf'):
                sorted_ancestors.append((ancestor_id, total_dist))
        
        # Sort by total distance
        sorted_ancestors.sort(key=lambda x: x[1])
        
        # Return sorted ancestor IDs
        return [a[0] for a in sorted_ancestors] 