"""
Integrity Guardian for the ΨC-AI SDK.

This module provides a guardian system that maintains the integrity of the AI system by:
1. Implementing thermal throttling for frequent reflections
2. Detecting duplicate memories/content based on hashing
3. Tracking conflicts and persistent contradictions

The IntegrityGuardian helps prevent system overload, redundancy, and logical inconsistencies
that could compromise the stability and coherence of the AI's cognitive processes.
"""

import time
import logging
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import deque, Counter
from dataclasses import dataclass

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.contradiction.detector import ContradictionDetector
from psi_c_ai_sdk.coherence.scorer import CoherenceScorer

logger = logging.getLogger(__name__)

@dataclass
class ConflictRecord:
    """Record of a detected conflict between memories or beliefs."""
    id: str
    memory_ids: Tuple[str, ...]
    description: str
    severity: float  # 0.0 to 1.0
    timestamp: float
    resolved: bool = False
    resolution_strategy: Optional[str] = None
    resolution_timestamp: Optional[float] = None

@dataclass
class ThrottleMetrics:
    """Metrics for the thermal throttling system."""
    reflection_count: int = 0
    last_reflection_time: float = 0
    cooling_factor: float = 0.95
    current_temperature: float = 0.0
    max_temperature: float = 100.0
    reflection_interval_seconds: float = 1.0
    throttle_threshold: float = 70.0
    critical_threshold: float = 90.0
    
    def update_temperature(self, current_time: float) -> None:
        """Update the temperature based on cooling since last reflection."""
        time_diff = current_time - self.last_reflection_time
        # Apply exponential cooling
        if time_diff > 0:
            self.current_temperature *= (self.cooling_factor ** time_diff)

class IntegrityGuardian:
    """
    Guardian system that maintains the integrity of the AI system.
    
    This system:
    1. Throttles frequent reflections to prevent computational overload
    2. Detects duplicate memories or content using hashing
    3. Tracks conflicts and persistent contradictions
    """
    
    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        coherence_scorer: Optional[CoherenceScorer] = None,
        throttle_config: Optional[Dict[str, float]] = None,
        duplicate_threshold: float = 0.95,
        enable_logging: bool = True
    ):
        """
        Initialize the IntegrityGuardian.
        
        Args:
            memory_store: MemoryStore instance to guard
            contradiction_detector: ContradictionDetector for finding conflicts
            coherence_scorer: CoherenceScorer for assessing coherence
            throttle_config: Configuration for thermal throttling
            duplicate_threshold: Threshold for determining duplicates (0.0-1.0)
            enable_logging: Whether to enable logging
        """
        self.memory_store = memory_store
        self.contradiction_detector = contradiction_detector
        self.coherence_scorer = coherence_scorer
        
        # Initialize throttle metrics with provided config
        self.throttle = ThrottleMetrics()
        if throttle_config:
            for key, value in throttle_config.items():
                if hasattr(self.throttle, key):
                    setattr(self.throttle, key, value)
        
        self.duplicate_threshold = duplicate_threshold
        self.enable_logging = enable_logging
        
        # Memory hashes for duplicate detection
        self.memory_hashes: Dict[str, str] = {}  # memory_id -> hash
        self.content_hashes: Dict[str, Set[str]] = {}  # content_hash -> set(memory_ids)
        
        # Conflict tracking
        self.conflicts: List[ConflictRecord] = []
        self.conflict_counter: Counter = Counter()  # memory_id -> count of conflicts
        self.persistent_contradictions: Dict[Tuple[str, str], int] = {}  # (mem_id1, mem_id2) -> count
        
        # Stats
        self.throttled_reflections = 0
        self.duplicates_detected = 0
        self.total_conflicts = 0
        self.resolved_conflicts = 0
        
        if self.enable_logging:
            logger.info("IntegrityGuardian initialized")
    
    # ===== Thermal Throttling Methods =====
    
    def is_overheating(self) -> bool:
        """Check if the system is overheating (temperature exceeds threshold)."""
        return self.throttle.current_temperature >= self.throttle.throttle_threshold
    
    def is_critical_temperature(self) -> bool:
        """Check if the system has reached a critical temperature."""
        return self.throttle.current_temperature >= self.throttle.critical_threshold
    
    def should_throttle_reflection(self) -> Tuple[bool, float]:
        """
        Determine if reflection should be throttled based on frequency and temperature.
        
        Returns:
            Tuple of (should_throttle, current_temperature)
        """
        current_time = time.time()
        
        # Update temperature based on cooling since last check
        self.throttle.update_temperature(current_time)
        
        # Check if we're within minimum interval
        time_since_last = current_time - self.throttle.last_reflection_time
        if time_since_last < self.throttle.reflection_interval_seconds:
            # Increase temperature for rapid successive reflections
            temp_increase = (self.throttle.reflection_interval_seconds - time_since_last) * 10
            self.throttle.current_temperature += temp_increase
        
        # Check if we should throttle
        should_throttle = self.is_overheating()
        
        if should_throttle and self.enable_logging:
            logger.warning(
                f"Reflection throttled: temperature {self.throttle.current_temperature:.2f}, "
                f"threshold {self.throttle.throttle_threshold:.2f}"
            )
            self.throttled_reflections += 1
        
        return should_throttle, self.throttle.current_temperature
    
    def register_reflection(self) -> None:
        """Register that a reflection has occurred, updating the temperature."""
        current_time = time.time()
        
        # Update existing temperature first
        self.throttle.update_temperature(current_time)
        
        # Then add heat for this reflection
        self.throttle.current_temperature += 10.0
        
        # Cap at max temperature
        self.throttle.current_temperature = min(
            self.throttle.current_temperature, 
            self.throttle.max_temperature
        )
        
        # Update reflection metrics
        self.throttle.reflection_count += 1
        self.throttle.last_reflection_time = current_time
    
    # ===== Duplicate Detection Methods =====
    
    def _compute_memory_hash(self, memory: Memory) -> str:
        """
        Compute a hash of the memory's content for duplicate detection.
        
        Args:
            memory: The memory to hash
            
        Returns:
            String hash of the memory content
        """
        # Create a string representation of the memory content
        content_str = f"{memory.content}-{memory.metadata.get('source', '')}"
        
        # Hash the content
        content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()
        return content_hash
    
    def detect_duplicate(self, memory: Memory) -> Optional[List[str]]:
        """
        Detect if a memory is a duplicate of existing memories.
        
        Args:
            memory: The memory to check
            
        Returns:
            List of memory IDs that this memory duplicates, or None if no duplicates
        """
        # Compute hash for this memory
        memory_hash = self._compute_memory_hash(memory)
        
        # Check if this hash exists in our content hashes
        if memory_hash in self.content_hashes:
            duplicate_ids = list(self.content_hashes[memory_hash])
            if duplicate_ids:
                if self.enable_logging:
                    logger.info(f"Duplicate detected: memory {memory.id} duplicates {duplicate_ids}")
                self.duplicates_detected += 1
                return duplicate_ids
        
        # Store the hash for future duplicate checks
        self.memory_hashes[memory.id] = memory_hash
        if memory_hash not in self.content_hashes:
            self.content_hashes[memory_hash] = set()
        self.content_hashes[memory_hash].add(memory.id)
        
        return None
    
    def register_memories(self, memories: List[Memory]) -> Dict[str, List[str]]:
        """
        Register multiple memories and detect duplicates.
        
        Args:
            memories: List of memories to register
            
        Returns:
            Dict mapping memory IDs to lists of duplicate memory IDs
        """
        duplicates = {}
        for memory in memories:
            dupe_ids = self.detect_duplicate(memory)
            if dupe_ids:
                duplicates[memory.id] = dupe_ids
        return duplicates
    
    # ===== Conflict Tracking Methods =====
    
    def register_conflict(
        self, 
        memory_ids: Tuple[str, ...], 
        description: str, 
        severity: float = 0.5
    ) -> str:
        """
        Register a conflict between memories.
        
        Args:
            memory_ids: Tuple of memory IDs involved in the conflict
            description: Description of the conflict
            severity: Severity score from 0.0 to 1.0
            
        Returns:
            ID of the conflict record
        """
        # Create conflict record
        conflict_id = f"conflict_{len(self.conflicts)}"
        conflict = ConflictRecord(
            id=conflict_id,
            memory_ids=memory_ids,
            description=description,
            severity=severity,
            timestamp=time.time()
        )
        
        # Update conflict tracking
        self.conflicts.append(conflict)
        self.total_conflicts += 1
        
        # Update conflict counter for each memory
        for mem_id in memory_ids:
            self.conflict_counter[mem_id] += 1
        
        # Track persistent contradictions
        if len(memory_ids) == 2:
            mem_pair = tuple(sorted(memory_ids))
            self.persistent_contradictions[mem_pair] = self.persistent_contradictions.get(mem_pair, 0) + 1
        
        if self.enable_logging:
            logger.info(f"Conflict registered: {conflict_id}, severity {severity}, memories {memory_ids}")
        
        return conflict_id
    
    def resolve_conflict(self, conflict_id: str, resolution_strategy: str) -> bool:
        """
        Mark a conflict as resolved.
        
        Args:
            conflict_id: ID of the conflict to resolve
            resolution_strategy: Description of how the conflict was resolved
            
        Returns:
            True if the conflict was found and resolved, False otherwise
        """
        for conflict in self.conflicts:
            if conflict.id == conflict_id and not conflict.resolved:
                conflict.resolved = True
                conflict.resolution_strategy = resolution_strategy
                conflict.resolution_timestamp = time.time()
                self.resolved_conflicts += 1
                
                if self.enable_logging:
                    logger.info(f"Conflict {conflict_id} resolved using strategy: {resolution_strategy}")
                
                return True
        
        return False
    
    def get_conflict_score(self, memory_id: str) -> float:
        """
        Get a conflict score for a memory based on its involvement in conflicts.
        
        Args:
            memory_id: ID of the memory to check
            
        Returns:
            Conflict score between 0.0 and 1.0
        """
        # Get the raw count of conflicts
        conflict_count = self.conflict_counter.get(memory_id, 0)
        
        # Normalize to a 0-1 score using a sigmoid-like function
        if conflict_count == 0:
            return 0.0
        
        # Sigmoid normalization: 1/(1+e^(-k*(x-x0)))
        # where k controls steepness and x0 is the midpoint
        normalized_score = 1.0 / (1.0 + 10.0 * (2.0 ** -conflict_count))
        return normalized_score
    
    def get_persistent_contradictions(self, threshold: int = 2) -> List[Tuple[Tuple[str, str], int]]:
        """
        Get pairs of memories that have persistent contradictions.
        
        Args:
            threshold: Minimum number of contradictions to be considered persistent
            
        Returns:
            List of ((memory_id1, memory_id2), count) tuples
        """
        return [(pair, count) for pair, count in self.persistent_contradictions.items() 
                if count >= threshold]
    
    # ===== Utility Methods =====
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the integrity guardian's operations."""
        return {
            "throttle": {
                "current_temperature": self.throttle.current_temperature,
                "reflection_count": self.throttle.reflection_count,
                "throttled_reflections": self.throttled_reflections,
                "throttle_threshold": self.throttle.throttle_threshold,
                "cooling_factor": self.throttle.cooling_factor
            },
            "duplicates": {
                "duplicates_detected": self.duplicates_detected,
                "unique_memories": len(self.memory_hashes),
                "unique_content_hashes": len(self.content_hashes)
            },
            "conflicts": {
                "total_conflicts": self.total_conflicts,
                "resolved_conflicts": self.resolved_conflicts,
                "unresolved_conflicts": self.total_conflicts - self.resolved_conflicts,
                "persistent_contradictions": len(self.get_persistent_contradictions()),
                "memories_with_conflicts": len(self.conflict_counter)
            }
        }
    
    def reset(self) -> None:
        """Reset all metrics and tracking (for testing or new sessions)."""
        # Reset throttling
        self.throttle = ThrottleMetrics()
        self.throttled_reflections = 0
        
        # Reset duplicate detection
        self.memory_hashes = {}
        self.content_hashes = {}
        self.duplicates_detected = 0
        
        # Reset conflict tracking
        self.conflicts = []
        self.conflict_counter = Counter()
        self.persistent_contradictions = {}
        self.total_conflicts = 0
        self.resolved_conflicts = 0
        
        if self.enable_logging:
            logger.info("IntegrityGuardian reset") 