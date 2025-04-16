"""
Self-Model Entropy Estimator

This module provides tools for estimating the complexity of an agent's self-schema
and its current understanding of its own identity. Higher entropy indicates a more
complex, nuanced self-model but can also indicate confusion or identity fragmentation.

The entropy estimator helps measure the level of stability in an agent's identity
and can be used to detect potential identity instability or drift over time.
"""

import math
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from psi_c_ai_sdk.schema.schema import SchemaGraph
from psi_c_ai_sdk.memory.memory_store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class SelfModelStats:
    """Statistics about the agent's self-model."""
    entropy_score: float = 0.0  # Overall entropy of the self-model (0-1)
    concept_entropy: float = 0.0  # Entropy of self-related concepts
    reference_entropy: float = 0.0  # Entropy of self-references
    stability_index: float = 1.0  # How stable the self-model is (0-1)
    complexity_score: float = 0.0  # Complexity of the self-model (0+)
    self_reference_count: int = 0  # Number of self-references
    self_concept_count: int = 0  # Number of self-concepts
    dominant_self_references: List[str] = field(default_factory=list)  # Most common self-references
    identity_fragmentation: float = 0.0  # How fragmented the identity is (0-1)
    semantic_coherence: float = 1.0  # How coherent the self-concepts are (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entropy_score": self.entropy_score,
            "concept_entropy": self.concept_entropy,
            "reference_entropy": self.reference_entropy,
            "stability_index": self.stability_index,
            "complexity_score": self.complexity_score,
            "self_reference_count": self.self_reference_count,
            "self_concept_count": self.self_concept_count,
            "dominant_self_references": self.dominant_self_references,
            "identity_fragmentation": self.identity_fragmentation,
            "semantic_coherence": self.semantic_coherence
        }


class SelfEntropyEstimator:
    """
    Estimates the entropy of an agent's self-model.
    
    This class analyzes how an agent references itself in its schema and
    memory store, calculating entropy metrics to capture the complexity
    and stability of the agent's self-model.
    """
    
    def __init__(
        self,
        schema_graph: SchemaGraph,
        memory_store: Optional[MemoryStore] = None,
        self_reference_terms: Optional[List[str]] = None
    ):
        """
        Initialize the self-model entropy estimator.
        
        Args:
            schema_graph: The agent's schema graph
            memory_store: The agent's memory store (optional)
            self_reference_terms: List of terms used for self-reference (default: ["I", "me", "my", "myself"])
        """
        self.schema_graph = schema_graph
        self.memory_store = memory_store
        self.self_reference_terms = self_reference_terms or ["I", "me", "my", "myself", "self"]
        
        # Cache of computed values
        self._self_nodes: Optional[Set[str]] = None
        self._self_references: Optional[Dict[str, int]] = None
        self._concept_frequencies: Optional[Dict[str, int]] = None
        self._reference_history: List[Dict[str, int]] = []
        self._entropy_history: List[float] = []
        
        # Settings
        self.history_window_size = 10
        self.min_reference_significance = 0.05
    
    def compute_entropy(self) -> SelfModelStats:
        """
        Compute the entropy of the agent's self-model.
        
        Returns:
            Statistics about the self-model
        """
        # Get self-related nodes and references
        self_nodes = self._get_self_nodes()
        self_references = self._get_self_references()
        
        # Calculate entropy scores
        concept_entropy = self._calculate_concept_entropy(self_nodes)
        reference_entropy = self._calculate_reference_entropy(self_references)
        
        # Calculate overall entropy as a weighted combination
        entropy_score = 0.6 * concept_entropy + 0.4 * reference_entropy
        
        # Calculate complexity score (unbounded)
        complexity_score = len(self_nodes) * entropy_score
        
        # Calculate stability by comparing to history
        stability_index = self._calculate_stability_index(self_references)
        
        # Calculate identity fragmentation
        identity_fragmentation = self._calculate_identity_fragmentation(self_nodes)
        
        # Calculate semantic coherence
        semantic_coherence = self._calculate_semantic_coherence(self_nodes)
        
        # Find dominant self-references
        dominant_refs = self._get_dominant_references(self_references)
        
        # Create stats object
        stats = SelfModelStats(
            entropy_score=entropy_score,
            concept_entropy=concept_entropy,
            reference_entropy=reference_entropy,
            stability_index=stability_index,
            complexity_score=complexity_score,
            self_reference_count=sum(self_references.values()),
            self_concept_count=len(self_nodes),
            dominant_self_references=dominant_refs,
            identity_fragmentation=identity_fragmentation,
            semantic_coherence=semantic_coherence
        )
        
        # Update history
        self._update_history(self_references, entropy_score)
        
        return stats
    
    def _get_self_nodes(self) -> Set[str]:
        """
        Get all self-related nodes from the schema graph.
        
        Returns:
            Set of node IDs that relate to the agent's self-model
        """
        if self._self_nodes is not None:
            return self._self_nodes
        
        self_nodes = set()
        
        # Find nodes with self-reference in content or metadata
        for node_id, data in self.schema_graph.graph.nodes(data=True):
            # Check node content
            content = data.get("content", "").lower()
            if any(term.lower() in content for term in self.self_reference_terms):
                self_nodes.add(node_id)
                continue
            
            # Check node type
            node_type = data.get("node_type", "")
            if node_type == "self_concept":
                self_nodes.add(node_id)
                continue
            
            # Check metadata
            metadata = data.get("metadata", {})
            if metadata.get("is_self_reference", False):
                self_nodes.add(node_id)
                continue
            
            # Check tags
            tags = metadata.get("tags", [])
            if any(tag in ["self", "identity", "self_model", "self_concept"] for tag in tags):
                self_nodes.add(node_id)
                continue
        
        # Cache results
        self._self_nodes = self_nodes
        return self_nodes
    
    def _get_self_references(self) -> Dict[str, int]:
        """
        Count occurrences of different self-reference terms.
        
        Returns:
            Dictionary mapping self-reference terms to counts
        """
        if self._self_references is not None:
            return self._self_references
        
        # Initialize counters
        self_references = {term.lower(): 0 for term in self.self_reference_terms}
        
        # Count occurrences in schema
        for _, data in self.schema_graph.graph.nodes(data=True):
            content = data.get("content", "").lower()
            for term in self.self_reference_terms:
                term_lower = term.lower()
                # Count rough occurrences (simplified - in real implementation,
                # this would use NLP to ensure these are actual self-references)
                count = content.count(term_lower)
                if count > 0:
                    self_references[term_lower] += count
        
        # If memory store is available, count occurrences there too
        if self.memory_store:
            for memory in self.memory_store.get_all_memories():
                content = memory.content.lower()
                for term in self.self_reference_terms:
                    term_lower = term.lower()
                    count = content.count(term_lower)
                    if count > 0:
                        self_references[term_lower] += count
        
        # Remove terms with zero occurrences
        self_references = {k: v for k, v in self_references.items() if v > 0}
        
        # Cache results
        self._self_references = self_references
        return self_references
    
    def _calculate_concept_entropy(self, self_nodes: Set[str]) -> float:
        """
        Calculate the entropy of self-concept distribution.
        
        Args:
            self_nodes: Set of self-related node IDs
            
        Returns:
            Entropy score between 0 and 1
        """
        if not self_nodes:
            return 0.0
        
        # Get concept categories by looking at node types and tags
        concept_categories = defaultdict(int)
        
        for node_id in self_nodes:
            data = self.schema_graph.graph.nodes[node_id]
            node_type = data.get("node_type", "unknown")
            
            # Increment the appropriate category
            concept_categories[node_type] += 1
            
            # Also check tags for more specific categorization
            metadata = data.get("metadata", {})
            tags = metadata.get("tags", [])
            for tag in tags:
                if tag in ["belief", "value", "trait", "goal", "preference", "ability"]:
                    concept_categories[tag] += 1
        
        # If no categories found, use node IDs as categories (maximum entropy)
        if not concept_categories:
            for node_id in self_nodes:
                concept_categories[node_id] = 1
        
        # Calculate entropy
        total = sum(concept_categories.values())
        entropy = 0.0
        
        for count in concept_categories.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
        
        # Normalize to [0, 1] range
        max_entropy = math.log2(len(concept_categories))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def _calculate_reference_entropy(self, self_references: Dict[str, int]) -> float:
        """
        Calculate the entropy of self-reference distribution.
        
        Args:
            self_references: Dictionary mapping self-reference terms to counts
            
        Returns:
            Entropy score between 0 and 1
        """
        if not self_references:
            return 0.0
        
        # Calculate probabilities
        total = sum(self_references.values())
        probabilities = [count / total for count in self_references.values()]
        
        # Calculate entropy
        entropy = 0.0
        for p in probabilities:
            entropy -= p * math.log2(p)
        
        # Normalize to [0, 1] range
        max_entropy = math.log2(len(self_references))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def _update_history(self, self_references: Dict[str, int], entropy_score: float) -> None:
        """
        Update history with current values.
        
        Args:
            self_references: Current self-reference counts
            entropy_score: Current entropy score
        """
        # Add to history
        self._reference_history.append(self_references.copy())
        self._entropy_history.append(entropy_score)
        
        # Trim history to window size
        if len(self._reference_history) > self.history_window_size:
            self._reference_history = self._reference_history[-self.history_window_size:]
            self._entropy_history = self._entropy_history[-self.history_window_size:]
    
    def _calculate_stability_index(self, current_references: Dict[str, int]) -> float:
        """
        Calculate stability index by comparing to history.
        
        Args:
            current_references: Current self-reference counts
            
        Returns:
            Stability index between 0 and 1
        """
        if not self._reference_history:
            return 1.0  # Default: stable
        
        # If we only have one history point, compare to it
        if len(self._reference_history) == 1:
            prev_refs = self._reference_history[0]
            return self._calculate_reference_similarity(prev_refs, current_references)
        
        # Calculate reference similarity over history window
        similarities = []
        for prev_refs in self._reference_history:
            similarity = self._calculate_reference_similarity(prev_refs, current_references)
            similarities.append(similarity)
        
        # Average similarity is the stability index
        return sum(similarities) / len(similarities)
    
    def _calculate_reference_similarity(
        self, refs1: Dict[str, int], refs2: Dict[str, int]
    ) -> float:
        """
        Calculate similarity between two reference distributions.
        
        Args:
            refs1: First reference distribution
            refs2: Second reference distribution
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get all reference terms
        all_terms = set(refs1.keys()) | set(refs2.keys())
        
        # Calculate total references in each
        total1 = sum(refs1.values())
        total2 = sum(refs2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = sum(refs1.get(term, 0) * refs2.get(term, 0) for term in all_terms)
        magnitude1 = math.sqrt(sum(count**2 for count in refs1.values()))
        magnitude2 = math.sqrt(sum(count**2 for count in refs2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return similarity
    
    def _calculate_identity_fragmentation(self, self_nodes: Set[str]) -> float:
        """
        Calculate identity fragmentation.
        
        Higher values indicate a more fragmented identity with disconnected
        self-concept nodes.
        
        Args:
            self_nodes: Set of self-related node IDs
            
        Returns:
            Fragmentation score between 0 and 1
        """
        if not self_nodes or len(self_nodes) < 2:
            return 0.0  # Can't be fragmented with 0 or 1 nodes
        
        # Count connections between self-nodes
        connection_count = 0
        possible_connections = len(self_nodes) * (len(self_nodes) - 1) / 2  # n(n-1)/2
        
        for node1 in self_nodes:
            for node2 in self_nodes:
                if node1 != node2 and self.schema_graph.graph.has_edge(node1, node2):
                    connection_count += 1
        
        # Adjust for double-counting in undirected graphs
        if not self.schema_graph.graph.is_directed():
            connection_count /= 2
        
        # Calculate fragmentation as 1 - connectivity_ratio
        connectivity_ratio = connection_count / possible_connections if possible_connections > 0 else 0
        fragmentation = 1.0 - connectivity_ratio
        
        return fragmentation
    
    def _calculate_semantic_coherence(self, self_nodes: Set[str]) -> float:
        """
        Calculate semantic coherence of self-concepts.
        
        Args:
            self_nodes: Set of self-related node IDs
            
        Returns:
            Coherence score between 0 and 1
        """
        if not self_nodes or len(self_nodes) < 2:
            return 1.0  # Default: coherent
        
        # In a real implementation, this would use embedding similarity 
        # between node contents to calculate semantic coherence
        # For now, we'll use a simplified approach based on schema structure
        
        # Calculate the average clustering coefficient of self-nodes
        # Higher clustering indicates more coherence
        clustering_sum = 0.0
        for node_id in self_nodes:
            neighbors = set(self.schema_graph.graph.neighbors(node_id))
            self_neighbors = neighbors.intersection(self_nodes)
            
            if len(self_neighbors) < 2:
                continue
                
            # Calculate local clustering coefficient
            possible_edges = len(self_neighbors) * (len(self_neighbors) - 1) / 2
            actual_edges = 0
            
            for n1 in self_neighbors:
                for n2 in self_neighbors:
                    if n1 != n2 and self.schema_graph.graph.has_edge(n1, n2):
                        actual_edges += 1
            
            # Adjust for double-counting in undirected graphs
            if not self.schema_graph.graph.is_directed():
                actual_edges /= 2
                
            # Calculate local clustering coefficient
            local_clustering = actual_edges / possible_edges if possible_edges > 0 else 0
            clustering_sum += local_clustering
        
        # Calculate average clustering coefficient
        avg_clustering = clustering_sum / len(self_nodes) if len(self_nodes) > 0 else 0
        
        # Coherence is directly related to clustering
        return avg_clustering
    
    def _get_dominant_references(self, self_references: Dict[str, int]) -> List[str]:
        """
        Get the dominant self-reference terms.
        
        Args:
            self_references: Dictionary mapping self-reference terms to counts
            
        Returns:
            List of dominant reference terms
        """
        if not self_references:
            return []
        
        # Calculate significance threshold
        total = sum(self_references.values())
        threshold = total * self.min_reference_significance
        
        # Get terms above threshold, sorted by count
        significant_terms = [(term, count) for term, count in self_references.items() 
                            if count >= threshold]
        significant_terms.sort(key=lambda x: x[1], reverse=True)
        
        return [term for term, _ in significant_terms]
    
    def get_self_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the agent's self-model.
        
        Returns:
            Dictionary with key metrics and insights
        """
        # Compute current stats
        stats = self.compute_entropy()
        
        # Get self-concept categories
        self_nodes = self._get_self_nodes()
        self_categories = self._categorize_self_concepts(self_nodes)
        
        # Determine primary identity components
        primary_components = self._get_primary_identity_components(self_nodes)
        
        # Determine identity stability trend
        stability_trend = self._calculate_stability_trend()
        
        return {
            "stats": stats.to_dict(),
            "self_categories": self_categories,
            "primary_components": primary_components,
            "stability_trend": stability_trend,
            "identity_health": self._evaluate_identity_health(stats)
        }
    
    def _categorize_self_concepts(self, self_nodes: Set[str]) -> Dict[str, int]:
        """
        Categorize self-concepts by type.
        
        Args:
            self_nodes: Set of self-related node IDs
            
        Returns:
            Dictionary mapping category to count
        """
        categories = defaultdict(int)
        
        for node_id in self_nodes:
            data = self.schema_graph.graph.nodes[node_id]
            node_type = data.get("node_type", "unknown")
            metadata = data.get("metadata", {})
            tags = metadata.get("tags", [])
            
            # Check for specific categories in tags
            categorized = False
            for category in ["belief", "value", "trait", "goal", "capability", 
                            "preference", "experience", "limitation"]:
                if category in tags:
                    categories[category] += 1
                    categorized = True
            
            # Fall back to node type if no specific category found
            if not categorized:
                categories[node_type] += 1
        
        return dict(categories)
    
    def _get_primary_identity_components(self, self_nodes: Set[str]) -> List[str]:
        """
        Get the primary components of the agent's identity.
        
        Args:
            self_nodes: Set of self-related node IDs
            
        Returns:
            List of primary identity components
        """
        if not self_nodes:
            return []
        
        # Calculate centrality of each self-node
        centrality = {}
        for node_id in self_nodes:
            # Count connections to other self-nodes
            connections = 0
            for neighbor in self.schema_graph.graph.neighbors(node_id):
                if neighbor in self_nodes:
                    connections += 1
            
            # Calculate centrality as combination of connections and node importance
            importance = self.schema_graph.graph.nodes[node_id].get("importance", 0.5)
            centrality[node_id] = connections * importance
        
        # Get the top 5 nodes by centrality
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Extract content or labels from these nodes
        components = []
        for node_id, _ in top_nodes:
            data = self.schema_graph.graph.nodes[node_id]
            content = data.get("content", "")
            label = data.get("label", "")
            
            # Use the shorter of content or label
            if label and (not content or len(label) < len(content)):
                components.append(label)
            else:
                # Truncate long content
                if len(content) > 50:
                    content = content[:47] + "..."
                components.append(content)
        
        return components
    
    def _calculate_stability_trend(self) -> str:
        """
        Calculate the trend in identity stability.
        
        Returns:
            Trend description: "stable", "increasing", "decreasing", or "fluctuating"
        """
        if len(self._entropy_history) < 3:
            return "stable"  # Not enough history to determine a trend
        
        # Calculate differences between consecutive entropy values
        diffs = [self._entropy_history[i] - self._entropy_history[i-1] 
                for i in range(1, len(self._entropy_history))]
        
        # Calculate average and variance of differences
        avg_diff = sum(diffs) / len(diffs)
        var_diff = sum((d - avg_diff) ** 2 for d in diffs) / len(diffs)
        
        # Determine trend based on average difference and variance
        if abs(avg_diff) < 0.01:
            if var_diff > 0.005:
                return "fluctuating"
            else:
                return "stable"
        elif avg_diff > 0:
            return "increasing"  # Entropy increasing = stability decreasing
        else:
            return "decreasing"  # Entropy decreasing = stability increasing
    
    def _evaluate_identity_health(self, stats: SelfModelStats) -> str:
        """
        Evaluate the overall health of the agent's identity model.
        
        Args:
            stats: Current self-model statistics
            
        Returns:
            Health assessment: "robust", "developing", "unstable", or "fragmented"
        """
        # Check for extreme values
        if stats.self_concept_count < 3:
            return "underdeveloped"  # Too few self-concepts
        
        if stats.identity_fragmentation > 0.8:
            return "fragmented"  # Highly disconnected self-concepts
        
        if stats.stability_index < 0.4:
            return "unstable"  # Low stability
        
        # Base health assessment on combination of factors
        score = (0.4 * stats.stability_index +
                0.3 * stats.semantic_coherence +
                0.3 * (1.0 - stats.identity_fragmentation))
        
        if score >= 0.8:
            return "robust"
        elif score >= 0.6:
            return "stable"
        elif score >= 0.4:
            return "developing"
        else:
            return "unstable"
    
    def reset_cache(self) -> None:
        """Reset cached values to force recalculation."""
        self._self_nodes = None
        self._self_references = None
        self._concept_frequencies = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert key state to dictionary for serialization."""
        return {
            "entropy_history": self._entropy_history,
            "reference_history": [
                {k: v for k, v in refs.items()} 
                for refs in self._reference_history
            ],
            "self_reference_terms": self.self_reference_terms,
            "history_window_size": self.history_window_size,
            "min_reference_significance": self.min_reference_significance
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        schema_graph: SchemaGraph,
        memory_store: Optional[MemoryStore] = None
    ) -> 'SelfEntropyEstimator':
        """
        Create an instance from serialized data.
        
        Args:
            data: Serialized state
            schema_graph: SchemaGraph instance
            memory_store: MemoryStore instance
            
        Returns:
            New SelfEntropyEstimator instance
        """
        estimator = cls(
            schema_graph=schema_graph,
            memory_store=memory_store,
            self_reference_terms=data.get("self_reference_terms", ["I", "me", "my", "myself"])
        )
        
        estimator.history_window_size = data.get("history_window_size", 10)
        estimator.min_reference_significance = data.get("min_reference_significance", 0.05)
        estimator._entropy_history = data.get("entropy_history", [])
        estimator._reference_history = [
            {k: v for k, v in refs.items()} 
            for refs in data.get("reference_history", [])
        ]
        
        return estimator


def estimate_self_entropy(
    schema_graph: SchemaGraph,
    memory_store: Optional[MemoryStore] = None
) -> SelfModelStats:
    """
    Convenience function to estimate self-model entropy.
    
    Args:
        schema_graph: The agent's schema graph
        memory_store: The agent's memory store (optional)
        
    Returns:
        Statistics about the self-model
    """
    estimator = SelfEntropyEstimator(
        schema_graph=schema_graph,
        memory_store=memory_store
    )
    return estimator.compute_entropy()


def generate_self_model_report(
    schema_graph: SchemaGraph,
    memory_store: Optional[MemoryStore] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive report on the agent's self-model.
    
    Args:
        schema_graph: The agent's schema graph
        memory_store: The agent's memory store (optional)
        
    Returns:
        Dictionary with report data
    """
    estimator = SelfEntropyEstimator(
        schema_graph=schema_graph,
        memory_store=memory_store
    )
    return estimator.get_self_model_summary() 