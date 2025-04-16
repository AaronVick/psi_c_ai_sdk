"""
Entropy Calculation for ΨC-AI SDK

This module provides tools for measuring entropy in AI memory systems. Entropy in this context
refers to the level of disorder or randomness in a system's memory embeddings and their semantic
relationships. High entropy can indicate incoherent, contradictory, or chaotic memory patterns
that may lead to unstable behavior in AI systems.

The module provides multiple entropy measures, including:
1. Embedding entropy - measures disorder in vector embedding space
2. Semantic coherence entropy - measures contradiction and incoherence in memory content
3. Temporal coherence entropy - measures the disorder in temporal patterns of memory access

Each measure can be used individually or combined through the EntropyCalculator class.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from collections import defaultdict

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.coherence.scorer import CoherenceScorer, BasicCoherenceScorer

logger = logging.getLogger(__name__)

class EntropyMeasure:
    """Base class for entropy measurement implementations."""
    
    def calculate_memory_entropy(self, memory: Memory) -> float:
        """
        Calculate entropy for a single memory.
        
        Args:
            memory: Memory object to calculate entropy for
            
        Returns:
            Entropy score between 0.0 and 1.0
        """
        raise NotImplementedError("Subclasses must implement calculate_memory_entropy")
    
    def calculate_memory_store_entropy(self, memory_store: MemoryStore) -> float:
        """
        Calculate overall entropy for a memory store.
        
        Args:
            memory_store: MemoryStore to calculate entropy for
            
        Returns:
            Entropy score between 0.0 and 1.0
        """
        raise NotImplementedError("Subclasses must implement calculate_memory_store_entropy")


class EmbeddingEntropyMeasure(EntropyMeasure):
    """
    Measures entropy based on embedding vector distributions.
    
    This measure calculates entropy by analyzing the distribution and clustering
    of memory embeddings in vector space. Well-clustered embeddings with clear
    semantic groupings have lower entropy, while scattered, uniformly distributed
    embeddings have higher entropy.
    """
    
    def __init__(self, cluster_threshold: float = 0.7):
        """
        Initialize the embedding entropy measure.
        
        Args:
            cluster_threshold: Similarity threshold for considering embeddings part of the same cluster
        """
        self.cluster_threshold = cluster_threshold
    
    def calculate_memory_entropy(self, memory: Memory) -> float:
        """
        Calculate entropy for a single memory based on its embedding.
        
        Args:
            memory: Memory object to calculate entropy for
            
        Returns:
            Entropy score between 0.0 and 1.0
        """
        if memory.embedding is None:
            logger.warning(f"Memory {memory.id} has no embedding, returning max entropy")
            return 1.0
        
        # For a single memory, entropy is calculated based on the vector's properties
        # More extreme values (further from 0) in the embedding indicate higher entropy
        embedding = np.array(memory.embedding)
        
        # Calculate z-scores to measure how extreme the values are
        mean = np.mean(embedding)
        std = np.std(embedding)
        if std == 0:
            return 1.0  # Maximum entropy for constant vectors
        
        z_scores = np.abs((embedding - mean) / std)
        
        # Calculate proportion of extreme values (beyond 2 standard deviations)
        extreme_proportion = np.mean(z_scores > 2.0)
        
        # Calculate normalized entropy score (0 to 1)
        entropy = min(1.0, 0.3 + (extreme_proportion * 0.7))
        
        return entropy
    
    def calculate_memory_store_entropy(self, memory_store: MemoryStore) -> float:
        """
        Calculate entropy based on embedding distribution for the entire memory store.
        
        Args:
            memory_store: MemoryStore to calculate entropy for
            
        Returns:
            Entropy score between 0.0 and 1.0
        """
        memories = memory_store.get_all_memories()
        if not memories:
            logger.warning("Empty memory store, returning minimum entropy")
            return 0.0
        
        # Filter memories that have embeddings
        memories_with_embeddings = [m for m in memories if m.embedding is not None]
        if not memories_with_embeddings:
            logger.warning("No memories with embeddings, returning maximum entropy")
            return 1.0
        
        # Perform embedding clustering analysis
        clusters = self._cluster_embeddings(memories_with_embeddings)
        
        # Calculate cluster statistics
        num_clusters = len(clusters)
        avg_cluster_size = np.mean([len(cluster) for cluster in clusters])
        singleton_ratio = sum(1 for cluster in clusters if len(cluster) == 1) / num_clusters if num_clusters > 0 else 1.0
        
        # Calculate entropy based on clustering
        # More clusters with smaller average size and higher singleton ratio = higher entropy
        normalized_cluster_count = min(1.0, num_clusters / max(len(memories_with_embeddings) / 3, 1))
        normalized_singleton_ratio = min(1.0, singleton_ratio * 1.5)
        
        # Combine factors into final entropy score
        entropy = 0.4 * normalized_cluster_count + 0.6 * normalized_singleton_ratio
        
        return min(1.0, entropy)
    
    def _cluster_embeddings(self, memories: List[Memory]) -> List[List[Memory]]:
        """
        Cluster memories based on embedding similarity.
        
        Args:
            memories: List of memories with embeddings to cluster
            
        Returns:
            List of memory clusters
        """
        if not memories:
            return []
        
        # Simple greedy clustering for demonstration
        clusters = []
        unassigned = list(memories)
        
        while unassigned:
            # Start a new cluster with the first unassigned memory
            current = unassigned.pop(0)
            current_cluster = [current]
            
            # Find all similar memories for this cluster
            i = 0
            while i < len(unassigned):
                memory = unassigned[i]
                similarity = self._calculate_similarity(current.embedding, memory.embedding)
                
                if similarity >= self.cluster_threshold:
                    current_cluster.append(memory)
                    unassigned.pop(i)
                else:
                    i += 1
            
            clusters.append(current_cluster)
        
        return clusters
    
    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity between the vectors (-1 to 1)
        """
        # Convert to numpy arrays
        v1 = np.array(emb1)
        v2 = np.array(emb2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Bound between 0 and 1 for our purposes
        return max(0.0, min(1.0, (similarity + 1) / 2))


class SemanticCoherenceEntropyMeasure(EntropyMeasure):
    """
    Measures entropy based on semantic coherence between memories.
    
    This measure uses a coherence scorer to evaluate how well each memory
    fits with the others in terms of semantic meaning. Contradictions,
    incoherence, and semantic drift all contribute to higher entropy scores.
    """
    
    def __init__(self, coherence_scorer: Optional[CoherenceScorer] = None):
        """
        Initialize the semantic coherence entropy measure.
        
        Args:
            coherence_scorer: CoherenceScorer to use for measuring semantic coherence
        """
        self.coherence_scorer = coherence_scorer or BasicCoherenceScorer()
    
    def calculate_memory_entropy(self, memory: Memory) -> float:
        """
        Calculate entropy for a single memory based on its semantic content.
        
        Args:
            memory: Memory object to calculate entropy for
            
        Returns:
            Entropy score between 0.0 and 1.0
        """
        # For a single memory, we need a context to evaluate against
        # This would normally be provided in calculate_memory_store_entropy
        # For isolated memories, we use a simpler heuristic
        
        # Look for indicators of high entropy in text
        content = memory.content.lower() if memory.content else ""
        
        # Count contradictory phrases
        contradiction_markers = [
            "but", "however", "although", "nevertheless", "nonetheless",
            "otherwise", "conversely", "instead", "rather", "despite",
            "in contrast", "on the contrary", "on the other hand",
            "vs", "versus", "not", "never", "impossible"
        ]
        
        contradiction_count = sum(1 for marker in contradiction_markers if marker in content.split())
        
        # Normalize to entropy score
        entropy = min(1.0, contradiction_count / 5)
        
        return entropy
    
    def calculate_memory_store_entropy(self, memory_store: MemoryStore) -> float:
        """
        Calculate entropy based on semantic coherence for the entire memory store.
        
        Args:
            memory_store: MemoryStore to calculate entropy for
            
        Returns:
            Entropy score between 0.0 and 1.0
        """
        memories = memory_store.get_all_memories()
        if not memories or len(memories) < 2:
            logger.warning("Insufficient memories for semantic coherence calculation")
            return 0.0
        
        # Calculate average coherence score for all memories
        total_score = 0.0
        memory_count = 0
        
        # Sample pairs of memories for coherence calculation (to avoid O(n²) complexity)
        max_pairs = min(500, len(memories) * 10)  # Limit number of pairs evaluated
        for _ in range(max_pairs):
            idx1 = np.random.randint(0, len(memories))
            idx2 = np.random.randint(0, len(memories))
            if idx1 != idx2:
                m1 = memories[idx1]
                m2 = memories[idx2]
                coherence = self.coherence_scorer.calculate_coherence(m1, m2)
                total_score += coherence
                memory_count += 1
        
        if memory_count == 0:
            return 0.5  # Default mid-level entropy
        
        avg_coherence = total_score / memory_count
        
        # Invert coherence to get entropy (high coherence = low entropy)
        entropy = 1.0 - avg_coherence
        
        return entropy


class TemporalCoherenceEntropyMeasure(EntropyMeasure):
    """
    Measures entropy based on temporal patterns in memory access and creation.
    
    This measure analyzes the temporal relationships between memories,
    detecting patterns of access and creation that may indicate
    chaotic or unstable memory usage.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize the temporal coherence entropy measure.
        
        Args:
            window_size: Number of recent memories to consider for temporal patterns
        """
        self.window_size = window_size
    
    def calculate_memory_entropy(self, memory: Memory) -> float:
        """
        Calculate entropy for a single memory based on temporal patterns.
        
        Args:
            memory: Memory object to calculate entropy for
            
        Returns:
            Entropy score between 0.0 and 1.0
        """
        # For a single memory, temporal entropy is based on its recency and access pattern
        if not memory.last_accessed or not memory.created_at:
            return 0.5  # Default to mid-level entropy
        
        # Calculate time since creation and last access
        now = memory.last_accessed
        time_since_creation = (now - memory.created_at).total_seconds()
        
        # Analyze access history if available
        if hasattr(memory, 'access_history') and memory.access_history:
            # Calculate irregularity in access pattern
            access_times = sorted(memory.access_history)
            if len(access_times) > 1:
                intervals = [access_times[i] - access_times[i-1] for i in range(1, len(access_times))]
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                # High variance in access intervals indicates higher entropy
                if mean_interval > 0:
                    coefficient_of_variation = std_interval / mean_interval
                    return min(1.0, coefficient_of_variation / 2)
        
        # Without access history, use a simplified metric
        # Older memories with recent access might indicate unstable memory usage
        if time_since_creation > 3600:  # More than an hour old
            recent_access_factor = 0.7  # Higher entropy for old memories accessed recently
        else:
            recent_access_factor = 0.3  # Lower entropy for recently created memories
            
        return recent_access_factor
    
    def calculate_memory_store_entropy(self, memory_store: MemoryStore) -> float:
        """
        Calculate entropy based on temporal patterns for the entire memory store.
        
        Args:
            memory_store: MemoryStore to calculate entropy for
            
        Returns:
            Entropy score between 0.0 and 1.0
        """
        memories = memory_store.get_all_memories()
        if not memories:
            return 0.0
        
        # Sort memories by last access time (most recent first)
        recent_memories = sorted(
            [m for m in memories if m.last_accessed], 
            key=lambda m: m.last_accessed,
            reverse=True
        )[:self.window_size]
        
        if not recent_memories:
            return 0.5  # Default to mid-level entropy
        
        # Analyze access patterns in the recent window
        topic_switches = 0
        importance_variance = 0
        
        # Check for topic switching (using embeddings)
        for i in range(1, len(recent_memories)):
            curr = recent_memories[i]
            prev = recent_memories[i-1]
            
            if curr.embedding and prev.embedding:
                similarity = self._embedding_similarity(curr.embedding, prev.embedding)
                if similarity < 0.5:  # Low similarity indicates topic switch
                    topic_switches += 1
        
        # Calculate importance variance
        if len(recent_memories) > 1:
            importances = [m.importance for m in recent_memories]
            importance_variance = np.std(importances)
        
        # Normalize factors into entropy score
        normalized_switches = min(1.0, topic_switches / (len(recent_memories) - 1) if len(recent_memories) > 1 else 0)
        normalized_variance = min(1.0, importance_variance * 3)
        
        # Combine factors
        entropy = 0.7 * normalized_switches + 0.3 * normalized_variance
        
        return entropy
    
    def _embedding_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        v1 = np.array(emb1)
        v2 = np.array(emb2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return max(0.0, min(1.0, (dot_product / (norm1 * norm2) + 1) / 2))


class EntropyCalculator:
    """
    Main class for calculating entropy in memory systems.
    
    This class combines multiple entropy measures to provide a comprehensive
    assessment of system entropy. It can be configured to use different
    measures and weighting strategies.
    """
    
    def __init__(
        self, 
        measures: Optional[Dict[str, Tuple[EntropyMeasure, float]]] = None
    ):
        """
        Initialize the entropy calculator with specified measures.
        
        Args:
            measures: Dictionary mapping measure names to (measure, weight) tuples
        """
        # Default measures if none provided
        if measures is None:
            self.measures = {
                "embedding": (EmbeddingEntropyMeasure(), 0.6),
                "semantic": (SemanticCoherenceEntropyMeasure(), 0.4)
            }
        else:
            self.measures = measures
        
        # Calculate total weight for normalization
        self.total_weight = sum(weight for _, weight in self.measures.values())
    
    def calculate_memory_entropy(self, memory: Memory) -> float:
        """
        Calculate weighted entropy for a single memory.
        
        Args:
            memory: Memory object to calculate entropy for
            
        Returns:
            Weighted entropy score between 0.0 and 1.0
        """
        if self.total_weight == 0:
            return 0.5  # Default to mid-level entropy
        
        weighted_sum = 0.0
        for measure, weight in self.measures.values():
            try:
                score = measure.calculate_memory_entropy(memory)
                weighted_sum += score * weight
            except Exception as e:
                logger.warning(f"Error calculating entropy with {measure.__class__.__name__}: {e}")
        
        return weighted_sum / self.total_weight
    
    def calculate_memory_store_entropy(self, memory_store: MemoryStore) -> float:
        """
        Calculate weighted entropy for a memory store.
        
        Args:
            memory_store: MemoryStore to calculate entropy for
            
        Returns:
            Weighted entropy score between 0.0 and 1.0
        """
        if self.total_weight == 0:
            return 0.5  # Default to mid-level entropy
        
        weighted_sum = 0.0
        for measure, weight in self.measures.values():
            try:
                score = measure.calculate_memory_store_entropy(memory_store)
                weighted_sum += score * weight
            except Exception as e:
                logger.warning(f"Error calculating entropy with {measure.__class__.__name__}: {e}")
        
        return weighted_sum / self.total_weight
    
    def get_detailed_entropy_metrics(self, memory_store: MemoryStore) -> Dict[str, float]:
        """
        Get detailed entropy metrics from all measures.
        
        Args:
            memory_store: MemoryStore to analyze
            
        Returns:
            Dictionary mapping measure names to entropy scores
        """
        metrics = {}
        for name, (measure, _) in self.measures.items():
            try:
                metrics[name] = measure.calculate_memory_store_entropy(memory_store)
            except Exception as e:
                logger.warning(f"Error calculating {name} entropy: {e}")
                metrics[name] = 0.5  # Default value on error
        
        # Add overall entropy
        metrics["overall"] = self.calculate_memory_store_entropy(memory_store)
        
        return metrics 