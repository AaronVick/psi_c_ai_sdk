"""
Entropy-based memory pruning functionality.

This module provides tools for identifying and pruning high-entropy memories
that might be causing incoherence or cognitive fragmentation in the system.
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
import logging

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.entropy.entropy import EntropyCalculator, EmbeddingEntropyMeasure

logger = logging.getLogger(__name__)


class EntropyBasedPruner:
    """
    Identify and prune high-entropy memories to maintain system coherence.
    
    This class provides functionality to identify memories with high entropy
    and prune them based on configurable strategies and thresholds.
    """
    
    def __init__(
        self,
        entropy_threshold: float = 0.8,
        importance_boost: float = 0.2,
        recency_protection: int = 10,
        max_pruning_ratio: float = 0.1
    ):
        """
        Initialize the entropy-based pruner.
        
        Args:
            entropy_threshold: Threshold above which memories are considered high-entropy
            importance_boost: Factor to boost memory importance when determining pruning
            recency_protection: Number of most recent memories to protect from pruning
            max_pruning_ratio: Maximum ratio of memories that can be pruned at once
        """
        self.entropy_threshold = entropy_threshold
        self.importance_boost = importance_boost
        self.recency_protection = recency_protection
        self.max_pruning_ratio = max_pruning_ratio
        self.entropy_calculator = EntropyCalculator()
        
        # Tracking for pruned memories
        self.pruning_history: List[Dict[str, Any]] = []
        self.last_pruning_stats: Dict[str, Any] = {}
    
    def identify_prunable_memories(
        self,
        memory_store: MemoryStore,
        max_candidates: int = 100
    ) -> List[Tuple[Memory, float, float]]:
        """
        Identify memories that are candidates for pruning.
        
        Args:
            memory_store: The memory store to analyze
            max_candidates: Maximum number of pruning candidates to return
            
        Returns:
            List of (memory, entropy, adjusted_score) sorted by pruning priority
        """
        # Get all memories
        memories = memory_store.get_all_memories()
        
        if not memories:
            return []
            
        # Sort by recency to protect recent memories
        memories.sort(key=lambda m: m.last_accessed_at or m.created_at, reverse=True)
        
        # Protect most recent memories from pruning
        protected = set()
        if self.recency_protection > 0:
            for memory in memories[:min(self.recency_protection, len(memories))]:
                protected.add(memory.uuid)
        
        # Calculate pruning scores
        candidates = []
        for memory in memories:
            if memory.uuid in protected or not memory.embedding:
                continue
                
            # Calculate memory entropy
            entropy = self.entropy_calculator.calculate_memory_entropy(memory)
            
            # Only consider memories above the threshold
            if entropy >= self.entropy_threshold:
                # Calculate pruning score: higher = more likely to prune
                # Importance reduces pruning likelihood
                adjusted_score = entropy - (self.importance_boost * memory.importance)
                candidates.append((memory, entropy, adjusted_score))
                
        # Sort by adjusted score (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates[:max_candidates]
    
    def prune_high_entropy_memories(
        self,
        memory_store: MemoryStore,
        dry_run: bool = False,
        max_to_prune: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Prune high-entropy memories from the memory store.
        
        Args:
            memory_store: The memory store to prune
            dry_run: If True, only identify memories but don't actually prune
            max_to_prune: Maximum number of memories to prune (None = use max_pruning_ratio)
            
        Returns:
            Statistics about the pruning operation
        """
        # Get all memories
        all_memories = memory_store.get_all_memories()
        total_count = len(all_memories)
        
        if total_count == 0:
            return {
                "pruned_count": 0,
                "total_count": 0, 
                "pruning_ratio": 0.0,
                "average_entropy": 0.0,
                "pruned_memory_ids": []
            }
            
        # Calculate max to prune based on ratio if not specified
        if max_to_prune is None:
            max_to_prune = int(total_count * self.max_pruning_ratio)
            max_to_prune = max(1, min(max_to_prune, 50))  # At least 1, at most 50
            
        # Identify candidates
        candidates = self.identify_prunable_memories(memory_store)
        
        # Limit to max_to_prune
        candidates = candidates[:max_to_prune]
        
        # Perform pruning
        pruned_memory_ids = []
        total_entropy = 0.0
        
        for memory, entropy, _ in candidates:
            memory_id = str(memory.uuid)
            
            # Log the pruning decision
            logger.info(f"Pruning high-entropy memory: {memory_id}, entropy: {entropy:.4f}, importance: {memory.importance:.2f}")
            
            # Actually delete unless this is a dry run
            if not dry_run:
                memory_store.delete_memory(memory.uuid)
                
            pruned_memory_ids.append(memory_id)
            total_entropy += entropy
            
        # Calculate statistics
        pruned_count = len(pruned_memory_ids)
        average_entropy = total_entropy / max(1, pruned_count)
        pruning_ratio = pruned_count / max(1, total_count)
        
        # Record results
        stats = {
            "pruned_count": pruned_count,
            "total_count": total_count,
            "pruning_ratio": pruning_ratio,
            "average_entropy": average_entropy,
            "pruned_memory_ids": pruned_memory_ids,
            "dry_run": dry_run
        }
        
        self.last_pruning_stats = stats
        self.pruning_history.append(stats)
        
        # Limit history size
        if len(self.pruning_history) > 50:
            self.pruning_history = self.pruning_history[-50:]
            
        return stats
    
    def get_pruning_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of pruning operations.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of pruning statistics dictionaries, newest first
        """
        history = list(reversed(self.pruning_history))
        return history[:limit]
    
    def adjust_thresholds(self, memory_store: MemoryStore) -> None:
        """
        Automatically adjust pruning thresholds based on memory store statistics.
        
        Args:
            memory_store: The memory store to analyze
        """
        # Calculate current system entropy
        current_entropy = self.entropy_calculator.calculate_memory_store_entropy(memory_store)
        
        # Adjust threshold based on current entropy
        # Lower system entropy → higher pruning threshold (more selective)
        # Higher system entropy → lower pruning threshold (more aggressive)
        base_adjustment = (current_entropy - 0.5) * 0.2
        self.entropy_threshold = max(0.5, min(0.95, self.entropy_threshold - base_adjustment))
        
        # Adjust importance boost based on entropy trend
        trend = self.entropy_calculator.calculate_entropy_trend()
        if trend["trend"] > 0.05:  # Entropy increasing rapidly
            # Reduce importance protection to be more aggressive
            self.importance_boost = max(0.1, self.importance_boost - 0.05)
        elif trend["trend"] < -0.05:  # Entropy decreasing
            # Increase importance protection to be more conservative
            self.importance_boost = min(0.5, self.importance_boost + 0.05)
            
        logger.info(f"Adjusted pruning thresholds: entropy_threshold={self.entropy_threshold:.2f}, importance_boost={self.importance_boost:.2f}")


def prune_high_entropy_memories(
    memory_store: MemoryStore,
    entropy_threshold: float = 0.8,
    max_to_prune: int = 10,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Utility function to prune high-entropy memories directly.
    
    Args:
        memory_store: The memory store to prune
        entropy_threshold: Threshold for considering memories high-entropy
        max_to_prune: Maximum number of memories to prune
        dry_run: If True, only identify memories but don't actually prune
        
    Returns:
        Statistics about the pruning operation
    """
    pruner = EntropyBasedPruner(entropy_threshold=entropy_threshold)
    return pruner.prune_high_entropy_memories(
        memory_store=memory_store,
        dry_run=dry_run,
        max_to_prune=max_to_prune
    ) 