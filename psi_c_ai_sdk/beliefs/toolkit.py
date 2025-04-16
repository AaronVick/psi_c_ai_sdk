"""
Belief Toolkit: A unified toolkit for managing belief systems and handling contradictions.

This module provides a combined toolkit that integrates contradiction detection, belief
revision, trust calibration, and other belief-related functionalities into a cohesive
system for managing AI belief systems.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set

from psi_c_ai_sdk.memory.memory import Memory
from psi_c_ai_sdk.memory.memory_store import MemoryStore
from psi_c_ai_sdk.beliefs.contradiction import ContradictionDetector, ContradictionType
from psi_c_ai_sdk.beliefs.revision import (
    BeliefRevisionSystem,
    DecisionStrategy,
    DecisionOutcome,
    TrustLevel
)
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer
from psi_c_ai_sdk.embedding.engine import EmbeddingEngine

logger = logging.getLogger(__name__)

class BeliefToolkit:
    """
    Unified toolkit for managing belief systems in AI.
    
    This toolkit combines:
    1. Contradiction detection - Finding inconsistencies in beliefs
    2. Belief revision - Resolving contradictions
    3. Trust calibration - Adjusting trust levels for information sources
    4. Decision logging - Recording and analyzing belief decisions
    
    It provides a simplified interface for managing the full belief system lifecycle.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_engine: EmbeddingEngine,
        coherence_scorer: Optional[CoherenceScorer] = None,
        similarity_threshold: float = 0.8,
        confidence_threshold: float = 0.6,
        coherence_weight: float = 0.3,
        trust_weight: float = 0.3,
        recency_weight: float = 0.3,
        entropy_weight: float = 0.1,
        decision_log_path: Optional[str] = None
    ):
        """
        Initialize the belief toolkit.
        
        Args:
            memory_store: Memory store containing beliefs/memories
            embedding_engine: Engine for generating embeddings
            coherence_scorer: Optional scorer for evaluating memory coherence
            similarity_threshold: Threshold for memory similarity (default: 0.8)
            confidence_threshold: Threshold for contradiction confidence (default: 0.6)
            coherence_weight: Weight for coherence in belief revision
            trust_weight: Weight for trust in belief revision
            recency_weight: Weight for recency in belief revision
            entropy_weight: Weight for entropy in belief revision
            decision_log_path: Optional path to log belief decisions
        """
        self.memory_store = memory_store
        
        # Initialize contradiction detector
        self.contradiction_detector = ContradictionDetector(
            embedding_engine=embedding_engine,
            similarity_threshold=similarity_threshold,
            confidence_threshold=confidence_threshold
        )
        
        # Initialize belief revision system
        self.belief_revision = BeliefRevisionSystem(
            memory_store=memory_store,
            contradiction_detector=self.contradiction_detector,
            coherence_scorer=coherence_scorer,
            strategy=DecisionStrategy.COMBINED,
            coherence_weight=coherence_weight,
            trust_weight=trust_weight,
            recency_weight=recency_weight,
            entropy_weight=entropy_weight,
            decision_log_path=decision_log_path
        )
        
        # Statistics
        self.stats = {
            "contradictions_found": 0,
            "contradictions_resolved": 0,
            "memories_removed": 0,
            "memory_merges": 0
        }
        
    def add_memory(
        self, 
        content: str, 
        source: Optional[str] = None,
        trust_level: Optional[TrustLevel] = None,
        check_contradictions: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Add a memory and optionally check for contradictions.
        
        Args:
            content: Content of the memory to add
            source: Optional source of the memory
            trust_level: Optional trust level for the source
            check_contradictions: Whether to check for contradictions
            metadata: Optional additional metadata
            
        Returns:
            Tuple of (memory_id, decisions) where decisions is a list of
            any contradiction resolution decisions made
        """
        # Set up metadata
        if metadata is None:
            metadata = {}
            
        if source:
            metadata["source"] = source
            
            # Set trust level for source if provided
            if trust_level:
                self.belief_revision.set_source_trust(source, trust_level)
        
        # Add the memory
        memory_id = self.memory_store.add_memory(content, metadata)
        
        # Check for contradictions if requested
        decisions = []
        if check_contradictions:
            decisions = self.check_contradictions()
            
        return memory_id, decisions
    
    def check_contradictions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Check for contradictions in the memory store and resolve them.
        
        Args:
            limit: Optional maximum number of contradictions to resolve
            
        Returns:
            List of decisions made during contradiction resolution
        """
        decisions = self.belief_revision.find_and_resolve_contradictions(limit)
        
        # Update statistics
        self.stats["contradictions_found"] += len(decisions)
        self.stats["contradictions_resolved"] += len(decisions)
        
        for decision in decisions:
            if decision["outcome"] == DecisionOutcome.MERGE.name:
                self.stats["memory_merges"] += 1
            elif decision["outcome"] in (
                DecisionOutcome.KEEP_FIRST.name, 
                DecisionOutcome.KEEP_SECOND.name
            ):
                self.stats["memories_removed"] += 1
            elif decision["outcome"] == DecisionOutcome.REMOVE_BOTH.name:
                self.stats["memories_removed"] += 2
                
        return decisions
    
    def add_pattern(
        self,
        name: str,
        pattern: str,
        contradiction_type: ContradictionType = ContradictionType.UNCERTAIN,
        confidence: float = 0.6
    ) -> bool:
        """
        Add a new pattern for contradiction detection.
        
        Args:
            name: Name of the pattern
            pattern: Regex pattern string
            contradiction_type: Type of contradiction
            confidence: Confidence score for pattern
            
        Returns:
            True if pattern was added successfully, False otherwise
        """
        return self.contradiction_detector.add_pattern(
            name=name,
            pattern=pattern,
            contradiction_type=contradiction_type,
            confidence=confidence
        )
        
    def add_antonym_pair(self, word1: str, word2: str) -> None:
        """
        Add a new antonym pair for semantic contradiction detection.
        
        Args:
            word1: First word in the antonym pair
            word2: Second word in the antonym pair
        """
        self.contradiction_detector.add_antonym_pair(word1, word2)
        
    def set_source_trust(self, source: str, trust_level: TrustLevel) -> None:
        """
        Set the trust level for a source.
        
        Args:
            source: Source identifier
            trust_level: Trust level to set
        """
        self.belief_revision.set_source_trust(source, trust_level)
        
    def get_source_trust(self, source: str) -> TrustLevel:
        """
        Get the trust level for a source.
        
        Args:
            source: Source identifier
            
        Returns:
            Trust level for the source
        """
        return self.belief_revision.get_source_trust(source)
    
    def get_decisions(self) -> List[Dict[str, Any]]:
        """
        Get the history of belief revision decisions.
        
        Returns:
            List of decisions
        """
        return self.belief_revision.get_decisions()
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about contradiction detection and resolution.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats
        
    def save_state(self, filepath: str) -> None:
        """
        Save the current state to a file.
        
        Args:
            filepath: Path to save state to
        """
        self.belief_revision.save_state(filepath)
        
    def load_state(self, filepath: str) -> bool:
        """
        Load state from a file.
        
        Args:
            filepath: Path to load state from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        return self.belief_revision.load_state(filepath) 