"""
Coherence Module for ΨC-AI SDK

This module provides tools for measuring semantic coherence between memories
and detecting contradictions or inconsistencies in the memory system.

Coherence is a fundamental property in conscious systems, representing how well
different pieces of information relate and form a unified model of reality.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from uuid import UUID

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore


class CoherenceScorer:
    """
    Calculates coherence scores between memories and identifies contradictions.
    
    The CoherenceScorer uses embedding similarity to determine how well memories
    fit together, detecting semantic relationships and potential contradictions.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        contradiction_threshold: float = -0.6,
        min_context_size: int = 5,
        global_weighting: float = 0.3,
        importance_factor: float = 0.5
    ):
        """
        Initialize the CoherenceScorer.
        
        Args:
            similarity_threshold: Minimum similarity for memories to be considered related
            contradiction_threshold: Similarity below this value indicates a contradiction
            min_context_size: Minimum number of memories to consider for context
            global_weighting: Weight of global coherence vs. local coherence
            importance_factor: How much memory importance affects coherence calculations
        """
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.min_context_size = min_context_size
        self.global_weighting = global_weighting
        self.importance_factor = importance_factor
        self.recent_coherence_scores: List[float] = []
        self.max_history = 100
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        if not embedding1 or not embedding2:
            return 0.0
            
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalize the vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Calculate cosine similarity
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def calculate_memory_coherence(
        self, 
        target_memory: Memory, 
        context_memories: List[Memory]
    ) -> float:
        """
        Calculate how coherent a memory is with a set of context memories.
        
        Args:
            target_memory: The memory to evaluate
            context_memories: List of memories to compare against
            
        Returns:
            Coherence score from 0 to 1
        """
        if not context_memories or not target_memory.embedding:
            return 0.5  # Neutral score when no context
            
        similarities = []
        importance_weights = []
        
        for memory in context_memories:
            if memory.uuid == target_memory.uuid or not memory.embedding:
                continue
                
            sim = self.calculate_similarity(target_memory.embedding, memory.embedding)
            
            # Apply importance weighting
            weight = 1.0
            if self.importance_factor > 0:
                weight = (1.0 - self.importance_factor) + (self.importance_factor * memory.importance)
                
            similarities.append(sim)
            importance_weights.append(weight)
            
        if not similarities:
            return 0.5  # Neutral score when no valid comparisons
            
        # Calculate weighted average similarity
        total_weight = sum(importance_weights)
        if total_weight == 0:
            return 0.5
            
        weighted_sim = sum(s * w for s, w in zip(similarities, importance_weights)) / total_weight
        
        # Convert from similarity (-1 to 1) to coherence (0 to 1)
        coherence = (weighted_sim + 1) / 2
        return coherence
    
    def calculate_global_coherence(
        self, 
        memory_store: MemoryStore, 
        recent_only: bool = False,
        sample_size: int = 100
    ) -> float:
        """
        Calculate the overall coherence of the memory system.
        
        Args:
            memory_store: The memory store to analyze
            recent_only: If True, only consider recent memories
            sample_size: Maximum number of memories to sample for calculation
            
        Returns:
            Global coherence score from 0 to 1
        """
        memories = memory_store.get_all_memories()
        
        if recent_only and len(memories) > sample_size:
            # Sort by recency and take the most recent
            memories.sort(key=lambda m: m.last_accessed_at or m.created_at, reverse=True)
            memories = memories[:sample_size]
            
        if len(memories) < self.min_context_size:
            return 0.5  # Neutral score for small memory stores
            
        # If we have too many memories, sample randomly
        if len(memories) > sample_size:
            # Sample, but ensure we keep high importance memories
            high_imp = [m for m in memories if m.importance > 0.7]
            remaining = [m for m in memories if m.importance <= 0.7]
            
            # Ensure at least 30% of high importance memories
            high_imp_count = min(len(high_imp), int(sample_size * 0.3))
            normal_count = sample_size - high_imp_count
            
            if high_imp_count > 0:
                high_imp = sorted(high_imp, key=lambda m: m.importance, reverse=True)[:high_imp_count]
                
            if normal_count > 0 and remaining:
                remaining = np.random.choice(remaining, size=normal_count, replace=False).tolist()
                
            memories = high_imp + remaining
            
        # Calculate pairwise coherence for each memory against others
        memory_coherence_scores = {}
        
        for memory in memories:
            if not memory.embedding:
                continue
                
            context = [m for m in memories if m.uuid != memory.uuid]
            coherence = self.calculate_memory_coherence(memory, context)
            memory_coherence_scores[memory.uuid] = coherence
            
        if not memory_coherence_scores:
            return 0.5
            
        # Calculate average coherence
        global_coherence = sum(memory_coherence_scores.values()) / len(memory_coherence_scores)
        
        # Update history
        self.recent_coherence_scores.append(global_coherence)
        if len(self.recent_coherence_scores) > self.max_history:
            self.recent_coherence_scores.pop(0)
            
        return global_coherence
    
    def detect_contradictions(
        self, 
        memory_store: MemoryStore,
        max_contradictions: int = 10
    ) -> List[Tuple[Memory, Memory, float]]:
        """
        Detect potential contradictions in the memory system.
        
        Args:
            memory_store: The memory store to analyze
            max_contradictions: Maximum number of contradictions to return
            
        Returns:
            List of tuples (memory1, memory2, similarity_score) for contradictory pairs
        """
        memories = memory_store.get_all_memories()
        contradictions = []
        
        # Consider all memories with embeddings
        valid_memories = [m for m in memories if m.embedding]
        
        # Compare all pairs
        for i, memory1 in enumerate(valid_memories):
            for memory2 in valid_memories[i+1:]:
                sim = self.calculate_similarity(memory1.embedding, memory2.embedding)
                
                # Check if this is a strong negative correlation
                if sim <= self.contradiction_threshold:
                    contradictions.append((memory1, memory2, sim))
                    
        # Sort by strength of contradiction (most negative first)
        contradictions.sort(key=lambda x: x[2])
        
        return contradictions[:max_contradictions]
    
    def calculate_entropy(self, memory_store: MemoryStore) -> float:
        """
        Calculate the entropy of the memory system, representing cognitive disorder.
        
        A high entropy indicates a disorganized memory system with poor structure.
        
        Args:
            memory_store: The memory store to analyze
            
        Returns:
            Entropy score from 0 to 1
        """
        # Get coherence history to measure stability
        if not self.recent_coherence_scores:
            return 0.5  # Neutral score with no history
            
        # Calculate entropy from coherence variance and mean
        if len(self.recent_coherence_scores) > 5:
            variance = np.var(self.recent_coherence_scores)
            mean = np.mean(self.recent_coherence_scores)
            
            # Low mean coherence and high variance indicates high entropy
            entropy = (1 - mean) * (1 + min(1.0, variance * 10))
            return min(1.0, max(0.0, entropy))
        else:
            # Not enough history, use inverse of current coherence as approximation
            current = self.recent_coherence_scores[-1]
            return 1 - current
    
    def get_most_related_memories(
        self, 
        target_memory: Memory, 
        memory_store: MemoryStore,
        min_similarity: float = 0.0,  
        max_memories: int = 10
    ) -> List[Tuple[Memory, float]]:
        """
        Find memories most related to the target memory.
        
        Args:
            target_memory: The reference memory
            memory_store: The memory store to search
            min_similarity: Minimum similarity threshold
            max_memories: Maximum number of memories to return
            
        Returns:
            List of tuples (memory, similarity_score) sorted by relevance
        """
        if not target_memory.embedding:
            return []
            
        memories = memory_store.get_all_memories()
        related = []
        
        for memory in memories:
            if memory.uuid == target_memory.uuid or not memory.embedding:
                continue
                
            sim = self.calculate_similarity(target_memory.embedding, memory.embedding)
            if sim >= min_similarity:
                related.append((memory, sim))
                
        # Sort by decreasing similarity
        related.sort(key=lambda x: x[1], reverse=True)
        
        return related[:max_memories]
    
    def calculate_coherence_change(self, before: float, after: float) -> Dict[str, Any]:
        """
        Calculate the change in coherence after a reflection operation.
        
        Args:
            before: Coherence score before reflection
            after: Coherence score after reflection
            
        Returns:
            Dict with change metrics
        """
        absolute_change = after - before
        relative_change = (after - before) / max(0.01, before)  # Avoid division by zero
        
        result = {
            "before": before,
            "after": after,
            "absolute_change": absolute_change,
            "relative_change": relative_change,
            "improved": absolute_change > 0
        }
        
        return result 