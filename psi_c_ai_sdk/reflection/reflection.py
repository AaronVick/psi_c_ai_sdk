"""
Reflection Engine for ΨC-AI SDK

This module implements the core reflection mechanism of consciousness, allowing
the system to introspect on its memories, detect inconsistencies, and improve
overall coherence through memory integration and consolidation.

Reflection represents a key aspect of consciousness, enabling the system to
maintain a consistent and adaptive internal world model.
"""

import time
import logging
import random
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from uuid import UUID

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReflectionTrigger(Enum):
    """Possible triggers for reflection cycles."""
    
    SCHEDULED = "scheduled"
    COHERENCE_THRESHOLD = "coherence_threshold"
    ENTROPY_THRESHOLD = "entropy_threshold"
    CONTRADICTION = "contradiction"
    MANUAL = "manual"


@dataclass
class ReflectionState:
    """State of a reflection cycle."""
    
    id: str
    trigger: ReflectionTrigger
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    # Before/after metrics
    coherence_before: float = 0.0
    coherence_after: float = 0.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    contradictions_before: int = 0
    contradictions_after: int = 0
    
    # Actions taken
    consolidated_memories: List[UUID] = field(default_factory=list)
    pruned_memories: List[UUID] = field(default_factory=list)
    augmented_memories: List[UUID] = field(default_factory=list)
    
    # Results
    successful: bool = False
    coherence_improvement: float = 0.0
    entropy_reduction: float = 0.0
    
    def calculate_metrics(self):
        """Calculate improvement metrics after reflection."""
        if self.end_time is None:
            self.end_time = time.time()
            
        self.duration = self.end_time - self.start_time
        self.coherence_improvement = self.coherence_after - self.coherence_before
        self.entropy_reduction = self.entropy_before - self.entropy_after
        self.successful = (
            self.coherence_improvement > 0 or 
            self.entropy_reduction > 0 or
            self.contradictions_before > self.contradictions_after
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage or serialization."""
        return {
            "id": self.id,
            "trigger": self.trigger.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "coherence_before": self.coherence_before,
            "coherence_after": self.coherence_after,
            "entropy_before": self.entropy_before,
            "entropy_after": self.entropy_after,
            "contradictions_before": self.contradictions_before,
            "contradictions_after": self.contradictions_after,
            "consolidated_memories": [str(uuid) for uuid in self.consolidated_memories],
            "pruned_memories": [str(uuid) for uuid in self.pruned_memories],
            "augmented_memories": [str(uuid) for uuid in self.augmented_memories],
            "successful": self.successful,
            "coherence_improvement": self.coherence_improvement,
            "entropy_reduction": self.entropy_reduction
        }


class ReflectionScheduler:
    """
    Manages when reflection cycles should occur based on various triggers.
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.6,
        entropy_threshold: float = 0.2,
        min_interval: float = 30.0,
        max_interval: float = 300.0,
        max_memory_count: int = 200,
        memory_threshold: int = 50,
        time_threshold: float = 3600.0  # 1 hour
    ):
        """
        Initialize the reflection scheduler.
        
        Args:
            coherence_threshold: Minimum coherence score before triggering reflection
            entropy_threshold: Maximum entropy before triggering reflection
            min_interval: Minimum seconds between reflection cycles
            max_interval: Maximum seconds between reflection cycles
            max_memory_count: Maximum memories before forced reflection
            memory_threshold: Number of new memories to trigger reflection
            time_threshold: Maximum time before forced reflection
        """
        self.coherence_threshold = coherence_threshold
        self.entropy_threshold = entropy_threshold
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_memory_count = max_memory_count
        self.memory_threshold = memory_threshold
        self.time_threshold = time_threshold
        
        self.last_reflection_time = 0.0
        self.memory_count_at_last_reflection = 0
        self.next_scheduled_reflection = time.time() + random.uniform(min_interval, max_interval)
    
    def should_reflect(
        self, 
        coherence_score: float,
        entropy_score: float,
        memory_count: int,
        contradictions: int
    ) -> Tuple[bool, Optional[ReflectionTrigger]]:
        """
        Determine if a reflection cycle should be triggered.
        
        Args:
            coherence_score: Current coherence score
            entropy_score: Current entropy score
            memory_count: Current number of memories
            contradictions: Number of current contradictions
            
        Returns:
            Tuple of (should_reflect, trigger_reason)
        """
        current_time = time.time()
        time_since_last = current_time - self.last_reflection_time
        
        # Check if minimum interval has passed
        if time_since_last < self.min_interval:
            return False, None
        
        # Check coherence threshold trigger
        if coherence_score < self.coherence_threshold:
            logger.info(f"Coherence below threshold: {coherence_score} < {self.coherence_threshold}")
            return True, ReflectionTrigger.COHERENCE_THRESHOLD
        
        # Check entropy threshold trigger
        if entropy_score > self.entropy_threshold:
            logger.info(f"Entropy above threshold: {entropy_score} > {self.entropy_threshold}")
            return True, ReflectionTrigger.ENTROPY_THRESHOLD
        
        # Check contradiction trigger
        if contradictions > 0:
            logger.info(f"Contradictions detected: {contradictions}")
            return True, ReflectionTrigger.CONTRADICTION
        
        # Check memory count trigger
        memory_increase = memory_count - self.memory_count_at_last_reflection
        if memory_increase >= self.memory_threshold:
            logger.info(f"Memory threshold reached: {memory_increase} new memories")
            return True, ReflectionTrigger.SCHEDULED
        
        # Check time threshold trigger
        if time_since_last >= self.time_threshold:
            logger.info(f"Time threshold reached: {time_since_last/60:.1f} minutes since last reflection")
            return True, ReflectionTrigger.SCHEDULED
        
        # Check next scheduled reflection
        if current_time >= self.next_scheduled_reflection:
            logger.info("Scheduled reflection time reached")
            self.next_scheduled_reflection = current_time + random.uniform(self.min_interval, self.max_interval)
            return True, ReflectionTrigger.SCHEDULED
        
        # Check maximum memory count
        if memory_count >= self.max_memory_count:
            logger.info(f"Maximum memory count reached: {memory_count} >= {self.max_memory_count}")
            return True, ReflectionTrigger.SCHEDULED
        
        return False, None
    
    def update_after_reflection(self, memory_count: int):
        """
        Update scheduler state after a reflection cycle.
        
        Args:
            memory_count: Current memory count after reflection
        """
        self.last_reflection_time = time.time()
        self.memory_count_at_last_reflection = memory_count
        self.next_scheduled_reflection = self.last_reflection_time + random.uniform(self.min_interval, self.max_interval)
        
    def time_until_next_reflection(self) -> float:
        """
        Calculate time until next scheduled reflection.
        
        Returns:
            Seconds until next scheduled reflection
        """
        return max(0.0, self.next_scheduled_reflection - time.time())


class ReflectionEngine:
    """
    Core reflection engine that performs introspection and memory restructuring.
    
    The reflection process involves:
    1. Detecting coherence issues, contradictions, and high entropy
    2. Consolidating related memories
    3. Resolving contradictions
    4. Pruning low-importance memories when necessary
    5. Augmenting memories with new insights
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        coherence_scorer: CoherenceScorer,
        scheduler: ReflectionScheduler,
        max_reflections_history: int = 10,
        on_reflection_complete: Optional[Callable[[ReflectionState], None]] = None,
        credit_system: Optional["ReflectionCreditSystem"] = None
    ):
        """
        Initialize the reflection engine.
        
        Args:
            memory_store: Memory store to operate on
            coherence_scorer: Coherence scorer for evaluating memory coherence
            scheduler: Reflection scheduler to determine when to reflect
            max_reflections_history: Maximum reflection cycles to keep in history
            on_reflection_complete: Optional callback when reflection completes
            credit_system: Optional reflection credit system for managing cognitive budget
        """
        self.memory_store = memory_store
        self.coherence_scorer = coherence_scorer
        self.scheduler = scheduler
        self.max_reflections_history = max_reflections_history
        self.on_reflection_complete = on_reflection_complete
        self.credit_system = credit_system
        
        self.reflection_history: List[ReflectionState] = []
        self.currently_reflecting = False
        self.total_reflections = 0
        self.successful_reflections = 0
        
    def check_reflection_needed(self) -> Tuple[bool, Optional[ReflectionTrigger]]:
        """
        Check if a reflection cycle is needed.
        
        Returns:
            Tuple of (should_reflect, trigger_reason)
        """
        if self.currently_reflecting:
            return False, None
            
        # Check if we have budget for reflection
        if self.credit_system:
            can_reflect, reason = self.credit_system.can_reflect()
            if not can_reflect:
                logger.debug(f"Credit system denied reflection: {reason}")
                return False, None
            
        # Get current system state
        coherence = self.coherence_scorer.calculate_global_coherence(self.memory_store)
        entropy = self.coherence_scorer.calculate_entropy(self.memory_store)
        memory_count = len(self.memory_store.get_all_memories())
        contradictions = len(self.coherence_scorer.detect_contradictions(self.memory_store))
        
        return self.scheduler.should_reflect(coherence, entropy, memory_count, contradictions)
        
    def start_reflection_cycle(self, trigger: ReflectionTrigger = ReflectionTrigger.MANUAL) -> str:
        """
        Start a reflection cycle.
        
        Args:
            trigger: What triggered this reflection cycle
            
        Returns:
            ID of the reflection cycle
        """
        if self.currently_reflecting:
            logger.warning("Reflection cycle already in progress")
            return ""
            
        self.currently_reflecting = True
        reflection_id = f"reflection_{self.total_reflections + 1}_{int(time.time())}"
        
        # Check credit budget if credit system is active
        if self.credit_system:
            can_reflect, reason = self.credit_system.can_reflect()
            if not can_reflect:
                logger.warning(f"Reflection denied by credit system: {reason}")
                self.currently_reflecting = False
                return ""
                
            # Reserve credit for this reflection
            self.credit_system.reserve_credit(reflection_id)
        
        # Initialize reflection state
        reflection_state = ReflectionState(
            id=reflection_id,
            trigger=trigger,
            start_time=time.time()
        )
        
        # Capture initial state metrics
        reflection_state.coherence_before = self.coherence_scorer.calculate_global_coherence(self.memory_store)
        reflection_state.entropy_before = self.coherence_scorer.calculate_entropy(self.memory_store)
        contradictions = self.coherence_scorer.detect_contradictions(self.memory_store)
        reflection_state.contradictions_before = len(contradictions)
        
        logger.info(f"Starting reflection cycle {reflection_id}")
        logger.info(f"Initial coherence: {reflection_state.coherence_before:.4f}")
        logger.info(f"Initial entropy: {reflection_state.entropy_before:.4f}")
        logger.info(f"Initial contradictions: {reflection_state.contradictions_before}")
        
        # Perform reflection actions
        actions_taken = self._perform_reflection_cycle(reflection_state, contradictions)
        
        # Finalize and evaluate reflection cycle
        reflection_state.coherence_after = self.coherence_scorer.calculate_global_coherence(self.memory_store)
        reflection_state.entropy_after = self.coherence_scorer.calculate_entropy(self.memory_store)
        reflection_state.contradictions_after = len(self.coherence_scorer.detect_contradictions(self.memory_store))
        reflection_state.end_time = time.time()
        reflection_state.calculate_metrics()
        
        # Update statistics
        self.total_reflections += 1
        if reflection_state.successful:
            self.successful_reflections += 1
            
        # Add to history
        self.reflection_history.append(reflection_state)
        if len(self.reflection_history) > self.max_reflections_history:
            self.reflection_history.pop(0)
            
        # Update scheduler
        self.scheduler.update_after_reflection(len(self.memory_store.get_all_memories()))
        
        # Process outcome in credit system
        if self.credit_system:
            refund = self.credit_system.process_reflection_outcome(reflection_state)
            if refund > 0:
                logger.info(f"Reflection credit refund: {refund:.1f}")
            elif refund < 0:
                logger.info(f"Reflection credit penalty: {abs(refund):.1f}")
        
        # Reset state
        self.currently_reflecting = False
        
        # Call callback if provided
        if self.on_reflection_complete:
            self.on_reflection_complete(reflection_state)
            
        logger.info(f"Completed reflection cycle {reflection_id}")
        logger.info(f"Final coherence: {reflection_state.coherence_after:.4f}")
        logger.info(f"Final entropy: {reflection_state.entropy_after:.4f}")
        logger.info(f"Final contradictions: {reflection_state.contradictions_after}")
        logger.info(f"Improvement: {reflection_state.coherence_improvement:.4f}")
        logger.info(f"Success: {reflection_state.successful}")
        
        return reflection_id
            
    def _perform_reflection_cycle(
        self, 
        reflection_state: ReflectionState,
        contradictions: List[Tuple[Memory, Memory, float]]
    ) -> Dict[str, int]:
        """
        Perform the actual reflection cycle operations.
        
        Args:
            reflection_state: Current reflection state
            contradictions: List of detected contradictions
            
        Returns:
            Dictionary with counts of actions taken
        """
        actions = {
            "consolidated": 0,
            "pruned": 0,
            "augmented": 0,
            "contradictions_resolved": 0
        }
        
        # 1. Resolve contradictions if any
        if contradictions:
            resolved = self._resolve_contradictions(contradictions, reflection_state)
            actions["contradictions_resolved"] = resolved
        
        # 2. Consolidate related memories
        consolidated = self._consolidate_related_memories(reflection_state)
        actions["consolidated"] = consolidated
        
        # 3. Prune low-importance memories if we have too many
        if len(self.memory_store.get_all_memories()) > self.scheduler.max_memory_count * 0.9:
            pruned = self._prune_low_importance_memories(reflection_state)
            actions["pruned"] = pruned
            
        # 4. Augment memories with insights
        augmented = self._augment_memories_with_insights(reflection_state)
        actions["augmented"] = augmented
        
        return actions
        
    def _resolve_contradictions(
        self,
        contradictions: List[Tuple[Memory, Memory, float]],
        reflection_state: ReflectionState
    ) -> int:
        """
        Resolve detected contradictions in memory.
        
        Args:
            contradictions: List of contradictions to resolve
            reflection_state: Current reflection state
            
        Returns:
            Number of contradictions resolved
        """
        resolved = 0
        
        for memory1, memory2, sim_score in contradictions:
            # For each contradiction, we have options:
            # 1. Keep the more important memory
            # 2. Keep the more recent memory
            # 3. Create a new synthesized memory
            
            # Simple strategy: Keep the more important memory, or more recent if tied
            if memory1.importance > memory2.importance + 0.1:
                # Keep memory1, update memory2 to reference the conflict
                memory2.metadata["contradiction_with"] = str(memory1.uuid)
                memory2.metadata["contradiction_resolved"] = "superseded"
                memory2.metadata["contradiction_reason"] = "Lower importance"
                memory2.importance *= 0.8  # Reduce importance of contradicted memory
                memory1.importance *= 1.1  # Boost importance of maintained memory
                self.memory_store.update_memory(memory2)
                self.memory_store.update_memory(memory1)
                reflection_state.pruned_memories.append(memory2.uuid)
                resolved += 1
                
            elif memory2.importance > memory1.importance + 0.1:
                # Keep memory2, update memory1 to reference the conflict
                memory1.metadata["contradiction_with"] = str(memory2.uuid)
                memory1.metadata["contradiction_resolved"] = "superseded"
                memory1.metadata["contradiction_reason"] = "Lower importance"
                memory1.importance *= 0.8  # Reduce importance of contradicted memory
                memory2.importance *= 1.1  # Boost importance of maintained memory
                self.memory_store.update_memory(memory1)
                self.memory_store.update_memory(memory2)
                reflection_state.pruned_memories.append(memory1.uuid)
                resolved += 1
                
            else:
                # Importance is similar, keep the more recent one
                if memory1.created_at > memory2.created_at:
                    memory2.metadata["contradiction_with"] = str(memory1.uuid)
                    memory2.metadata["contradiction_resolved"] = "superseded"
                    memory2.metadata["contradiction_reason"] = "Older memory"
                    memory2.importance *= 0.8
                    self.memory_store.update_memory(memory2)
                    reflection_state.pruned_memories.append(memory2.uuid)
                else:
                    memory1.metadata["contradiction_with"] = str(memory2.uuid)
                    memory1.metadata["contradiction_resolved"] = "superseded"
                    memory1.metadata["contradiction_reason"] = "Older memory"
                    memory1.importance *= 0.8
                    self.memory_store.update_memory(memory1)
                    reflection_state.pruned_memories.append(memory1.uuid)
                resolved += 1
                
        return resolved
        
    def _consolidate_related_memories(self, reflection_state: ReflectionState) -> int:
        """
        Consolidate highly related memories to improve coherence.
        
        Args:
            reflection_state: Current reflection state
            
        Returns:
            Number of memories consolidated
        """
        memories = self.memory_store.get_all_memories()
        consolidated = 0
        
        # Skip if too few memories
        if len(memories) < 10:
            return 0
            
        # Find clusters of highly related memories based on embeddings
        processed_memories = set()
        
        for memory in memories:
            if memory.uuid in processed_memories or not memory.embedding:
                continue
                
            # Find related memories
            related = self.coherence_scorer.get_most_related_memories(
                memory, 
                self.memory_store,
                min_similarity=0.85  # High similarity threshold for consolidation
            )
            
            # Skip if not enough related memories
            if len(related) < 2:
                continue
                
            # Get the UUIDs of related memories
            related_uuids = [mem.uuid for mem, _ in related]
            
            # Check if we have at least 3 highly related memories
            if len(related_uuids) >= 2:
                # Create a consolidated memory
                related_memories = [memory] + [mem for mem, _ in related]
                
                # Create consolidated content
                common_topics = set.intersection(*[set(mem.tags) for mem in related_memories if mem.tags])
                
                # Build consolidated content
                content = f"Consolidated memory from {len(related_memories)} related memories.\n\n"
                
                # Add memory contents with attribution
                for i, mem in enumerate(related_memories[:5]):  # Limit to 5 to avoid huge memories
                    content += f"Memory {i+1}: {mem.content[:100]}...\n"
                    
                # Create consolidated memory
                consolidated_memory = Memory(
                    content=content,
                    source="reflection_consolidation",
                    importance=max([mem.importance for mem in related_memories]) * 1.2,
                    tags=list(common_topics) if common_topics else ["consolidated"],
                    metadata={
                        "consolidated_from": [str(mem.uuid) for mem in related_memories],
                        "consolidation_time": datetime.now().isoformat(),
                        "memory_count": len(related_memories)
                    }
                )
                
                # Add embedding by averaging related memory embeddings
                if all(mem.embedding for mem in related_memories):
                    embeddings = [mem.embedding for mem in related_memories if mem.embedding]
                    consolidated_memory.embedding = [
                        sum(e[i] for e in embeddings) / len(embeddings)
                        for i in range(len(embeddings[0]))
                    ]
                
                # Add to memory store
                self.memory_store.add_memory(consolidated_memory)
                reflection_state.consolidated_memories.append(consolidated_memory.uuid)
                
                # Lower importance of original memories
                for mem in related_memories:
                    mem.importance *= 0.7
                    mem.metadata["consolidated_into"] = str(consolidated_memory.uuid)
                    self.memory_store.update_memory(mem)
                    
                # Mark as processed
                processed_memories.update([mem.uuid for mem in related_memories])
                consolidated += len(related_memories)
                
        return consolidated
        
    def _prune_low_importance_memories(self, reflection_state: ReflectionState) -> int:
        """
        Prune low-importance memories to manage memory size.
        
        Args:
            reflection_state: Current reflection state
            
        Returns:
            Number of memories pruned
        """
        memories = self.memory_store.get_all_memories()
        
        # Skip if we don't have enough memories
        if len(memories) < self.scheduler.max_memory_count * 0.8:
            return 0
            
        # Sort by importance (ascending)
        memories.sort(key=lambda m: m.importance)
        
        # Determine how many to prune (keep under 90% of max)
        target_count = int(self.scheduler.max_memory_count * 0.8)
        to_prune = max(0, len(memories) - target_count)
        
        # Don't prune more than 10% at a time
        max_prune = int(len(memories) * 0.1)
        to_prune = min(to_prune, max_prune)
        
        if to_prune <= 0:
            return 0
            
        # Prune the least important memories
        pruned = 0
        for memory in memories[:to_prune]:
            # Don't prune very new memories
            if (datetime.now() - memory.created_at) < timedelta(hours=1):
                continue
                
            # Don't prune high importance memories
            if memory.importance > 0.6:
                continue
                
            # Add to pruned list before deleting
            reflection_state.pruned_memories.append(memory.uuid)
            self.memory_store.delete_memory(memory.uuid)
            pruned += 1
            
        return pruned
        
    def _augment_memories_with_insights(self, reflection_state: ReflectionState) -> int:
        """
        Augment existing memories with new insights from reflection.
        
        Args:
            reflection_state: Current reflection state
            
        Returns:
            Number of memories augmented
        """
        # This would be more sophisticated in a real implementation
        # For demo purposes, we'll just add reflection metadata to important memories
        
        memories = self.memory_store.get_all_memories()
        augmented = 0
        
        # Find important memories without reflection metadata
        candidates = [
            mem for mem in memories 
            if mem.importance > 0.7 and "reflection_insight" not in mem.metadata
        ]
        
        # Limit to 3 augmentations per cycle
        for memory in candidates[:3]:
            # Add reflection insight
            memory.metadata["reflection_insight"] = {
                "reflection_id": reflection_state.id,
                "reflection_time": datetime.now().isoformat(),
                "insight": f"This memory has been identified as particularly important during reflection cycle {reflection_state.id}."
            }
            
            # Boost importance slightly
            memory.importance = min(1.0, memory.importance * 1.05)
            
            # Update memory
            self.memory_store.update_memory(memory)
            reflection_state.augmented_memories.append(memory.uuid)
            augmented += 1
            
        return augmented
        
    def get_reflection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about reflection cycles.
        
        Returns:
            Dictionary with reflection statistics
        """
        total_duration = sum(r.duration or 0 for r in self.reflection_history)
        avg_duration = total_duration / max(1, len(self.reflection_history))
        
        avg_coherence_improvement = 0
        avg_entropy_reduction = 0
        
        if self.reflection_history:
            avg_coherence_improvement = sum(r.coherence_improvement for r in self.reflection_history) / len(self.reflection_history)
            avg_entropy_reduction = sum(r.entropy_reduction for r in self.reflection_history) / len(self.reflection_history)
        
        stats = {
            "total_reflections": self.total_reflections,
            "successful_reflections": self.successful_reflections,
            "success_rate": self.successful_reflections / max(1, self.total_reflections),
            "avg_duration": avg_duration,
            "avg_coherence_improvement": avg_coherence_improvement,
            "avg_entropy_reduction": avg_entropy_reduction,
            "next_reflection_in": self.scheduler.time_until_next_reflection(),
            "recent_history": [r.to_dict() for r in self.reflection_history[-3:]]
        }
        
        # Add credit system stats if available
        if self.credit_system:
            credit_stats = self.credit_system.get_credit_stats()
            cognitive_debt = calculate_cognitive_debt(self.credit_system, self.reflection_history)
            
            stats.update({
                "credit": credit_stats,
                "cognitive_debt": cognitive_debt
            })
            
        return stats
        
    def force_reflection(self) -> str:
        """
        Force a manual reflection cycle.
        
        Returns:
            ID of the reflection cycle
        """
        return self.start_reflection_cycle(ReflectionTrigger.MANUAL) 