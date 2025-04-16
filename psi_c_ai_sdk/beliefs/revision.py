"""
Belief Revision System for ΨC-AI SDK

This module provides tools for detecting and resolving contradictions in an AI's
belief system, including arbitration between contradictory memories, trust calibration
based on sources, and logging of belief revision decisions.
"""

import logging
import uuid
from typing import Dict, List, Tuple, Optional, Set, Any, NamedTuple, Union
from datetime import datetime
from enum import Enum, auto

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.contradiction.detector import ContradictionDetector, ContradictionResult
from psi_c_ai_sdk.coherence.scorer import CoherenceScorer

logger = logging.getLogger(__name__)


class DecisionStrategy(Enum):
    """Strategies for resolving contradictions."""
    WEIGHTED = auto()  # Use weighted scores across multiple factors
    COHERENCE = auto()  # Prioritize coherence with other memories
    TRUST = auto()     # Prioritize trust in sources
    RECENCY = auto()   # Prioritize recent memories
    

class DecisionOutcome(Enum):
    """Possible outcomes of a belief revision decision."""
    KEPT_FIRST = auto()       # First memory was kept
    KEPT_SECOND = auto()      # Second memory was kept
    KEPT_BOTH = auto()        # Both memories were kept (no clear winner)
    MODIFIED = auto()         # A memory was modified to resolve the contradiction
    QUARANTINED = auto()      # Memories were quarantined for further review
    MERGED = auto()           # Memories were merged


class TrustLevel(Enum):
    """Trust levels for memory sources."""
    VERIFIED = 1.0     # Fully verified and trusted
    HIGH = 0.8         # High trust
    MEDIUM = 0.6       # Medium trust
    LOW = 0.4          # Low trust
    UNTRUSTED = 0.2    # Not trusted
    UNKNOWN = 0.5      # Unknown/default trust level


class RevisionDecision(NamedTuple):
    """Represents a decision made during belief revision."""
    id: str
    timestamp: datetime
    kept_memory_id: str
    rejected_memory_id: str
    reason: str
    weights: Dict[str, float]  # Weights used for the decision
    scores: Dict[str, Dict[str, float]]   # Individual scores for each factor
    outcome: DecisionOutcome    # Outcome of the decision


class TrustSource(NamedTuple):
    """Represents a source of trust with an associated level."""
    name: str
    level: float


class BeliefRevisionSystem:
    """
    System for detecting and resolving contradictions in an AI's belief system.
    
    This class provides tools for arbitrating between contradictory memories,
    calibrating trust in different sources, and tracking belief revisions.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        contradiction_detector: ContradictionDetector,
        coherence_scorer: CoherenceScorer,
        trust_sources: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        decision_strategy: DecisionStrategy = DecisionStrategy.WEIGHTED
    ):
        """
        Initialize the belief revision system.
        
        Args:
            memory_store: Memory store containing beliefs to manage
            contradiction_detector: Detector for finding contradictions
            coherence_scorer: Scorer for measuring memory coherence
            trust_sources: Initial trust levels for different sources, 0.0-1.0
            weights: Weights for different factors in belief arbitration
                (coherence, trust, recency, entropy)
            decision_strategy: Strategy to use for arbitration decisions
        """
        self.memory_store = memory_store
        self.contradiction_detector = contradiction_detector
        self.coherence_scorer = coherence_scorer
        self.decision_strategy = decision_strategy
        
        # Initialize trust sources
        self.trust_sources = trust_sources or {}
        
        # Initialize weights for belief arbitration
        self._init_weights(weights)
        
        # Decision tracking
        self.decisions: List[RevisionDecision] = []
        
        # Quarantined memories needing human review
        self.quarantined_pairs: List[Tuple[str, str, str]] = []
        
        # Statistics
        self.stats = {
            "contradictions_found": 0,
            "contradictions_resolved": 0,
            "trust_adjustments": 0,
            "memories_kept": 0,
            "memories_rejected": 0,
            "memories_quarantined": 0,
            "decision_by_outcome": {}
        }
    
    def _init_weights(self, weights: Optional[Dict[str, float]]) -> None:
        """
        Initialize and validate weights for belief arbitration.
        
        Args:
            weights: Weights for different factors, or None for defaults
        """
        default_weights = {
            "coherence": 0.4,   # Weight for coherence with other memories
            "trust": 0.3,       # Weight for source trust
            "recency": 0.2,     # Weight for recency
            "entropy": 0.1      # Weight for low entropy (anti-chaotic)
        }
        
        if weights is None:
            self.weights = default_weights
        else:
            # Ensure all required weights are present
            self.weights = {**default_weights, **weights}
            
            # Validate weights sum to 1.0
            total = sum(self.weights.values())
            if abs(total - 1.0) > 0.001:
                logger.warning(f"Weights sum to {total}, not 1.0. Normalizing.")
                self.weights = {k: v / total for k, v in self.weights.items()}
    
    def find_and_resolve_contradictions(
        self,
        limit: Optional[int] = None,
        threshold: float = 0.7,
        quarantine_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Find and resolve contradictions in the memory store.
        
        This method identifies contradictory memory pairs, applies the arbitration
        logic to decide which memory to keep, and updates the system accordingly.
        
        Args:
            limit: Maximum number of contradictions to process, None for all
            threshold: Confidence threshold for contradiction resolution (0.0-1.0)
            quarantine_threshold: Score difference below which to quarantine (0.0-1.0)
            
        Returns:
            Statistics about the contradiction resolution process
        """
        # Get all memories from the store
        memories = self.memory_store.get_all_memories()
        if len(memories) < 2:
            return {"contradictions_found": 0, "contradictions_resolved": 0}
        
        # Find contradictions
        contradiction_pairs = self.contradiction_detector.find_contradictions(memories)
        self.stats["contradictions_found"] += len(contradiction_pairs)
        
        if not contradiction_pairs:
            return {"contradictions_found": 0, "contradictions_resolved": 0}
        
        # Limit number of contradictions to process
        if limit is not None and len(contradiction_pairs) > limit:
            contradiction_pairs = contradiction_pairs[:limit]
        
        # Process each contradiction
        decisions = []
        for memory1, memory2, confidence in contradiction_pairs:
            # Skip low-confidence contradictions
            if confidence < threshold:
                logger.debug(f"Skipping low-confidence contradiction ({confidence:.2f}) between {memory1.id} and {memory2.id}")
                continue
                
            decision = self._arbitrate_contradiction(
                memory1, memory2, quarantine_threshold
            )
            
            if decision:
                decisions.append(decision)
                self.stats["contradictions_resolved"] += 1
                
                # Update outcome statistics
                outcome_name = decision.outcome.name
                if outcome_name not in self.stats["decision_by_outcome"]:
                    self.stats["decision_by_outcome"][outcome_name] = 0
                self.stats["decision_by_outcome"][outcome_name] += 1
                
                # Update memory statistics
                if decision.outcome == DecisionOutcome.KEPT_FIRST or decision.outcome == DecisionOutcome.KEPT_SECOND:
                    self.stats["memories_kept"] += 1
                    self.stats["memories_rejected"] += 1
                elif decision.outcome == DecisionOutcome.QUARANTINED:
                    self.stats["memories_quarantined"] += 2
        
        return {
            "contradictions_found": len(contradiction_pairs),
            "contradictions_resolved": len(decisions),
            "decisions": [d._asdict() for d in decisions]
        }
    
    def _arbitrate_contradiction(
        self,
        memory1: Memory,
        memory2: Memory,
        quarantine_threshold: float = 0.1
    ) -> Optional[RevisionDecision]:
        """
        Arbitrate between two contradictory memories.
        
        Args:
            memory1: First memory in contradiction
            memory2: Second memory in contradiction
            quarantine_threshold: Score difference below which to quarantine (0.0-1.0)
            
        Returns:
            A RevisionDecision if arbitration succeeds, None otherwise
        """
        logger.info(f"Arbitrating contradiction between memories {memory1.id} and {memory2.id}")
        
        # Calculate scores for each memory based on strategy
        scores1 = self._calculate_memory_scores(memory1)
        scores2 = self._calculate_memory_scores(memory2)
        
        # Determine final scores based on strategy
        if self.decision_strategy == DecisionStrategy.WEIGHTED:
            # Apply weights to get final scores
            final_score1 = sum(scores1[k] * self.weights.get(k, 0) for k in scores1)
            final_score2 = sum(scores2[k] * self.weights.get(k, 0) for k in scores2)
        elif self.decision_strategy == DecisionStrategy.COHERENCE:
            final_score1 = scores1["coherence"]
            final_score2 = scores2["coherence"]
        elif self.decision_strategy == DecisionStrategy.TRUST:
            final_score1 = scores1["trust"]
            final_score2 = scores2["trust"]
        elif self.decision_strategy == DecisionStrategy.RECENCY:
            final_score1 = scores1["recency"]
            final_score2 = scores2["recency"]
        else:
            # Default to weighted
            final_score1 = sum(scores1[k] * self.weights.get(k, 0) for k in scores1)
            final_score2 = sum(scores2[k] * self.weights.get(k, 0) for k in scores2)
        
        # Calculate score difference
        score_diff = abs(final_score1 - final_score2)
        
        # If scores are too close, quarantine instead of deciding
        if score_diff < quarantine_threshold:
            logger.info(f"Scores too close ({final_score1:.4f} vs {final_score2:.4f}), quarantining memories")
            reason = f"Score difference ({score_diff:.4f}) below threshold ({quarantine_threshold})"
            
            # Add to quarantine list
            quarantine_id = str(uuid.uuid4())
            self.quarantined_pairs.append((memory1.id, memory2.id, quarantine_id))
            
            # Create decision record
            decision = RevisionDecision(
                id=quarantine_id,
                timestamp=datetime.now(),
                kept_memory_id="",  # No memory kept
                rejected_memory_id="",  # No memory rejected
                reason=reason,
                weights=self.weights.copy(),
                scores={
                    "memory1": scores1,
                    "memory2": scores2,
                    "final1": final_score1,
                    "final2": final_score2
                },
                outcome=DecisionOutcome.QUARANTINED
            )
            
            self.decisions.append(decision)
            return decision
        
        # Determine which memory to keep
        if final_score1 >= final_score2:
            kept_memory = memory1
            rejected_memory = memory2
            kept_scores = scores1
            reason = f"Memory {memory1.id} scored higher ({final_score1:.4f} vs {final_score2:.4f})"
            outcome = DecisionOutcome.KEPT_FIRST
        else:
            kept_memory = memory2
            rejected_memory = memory1
            kept_scores = scores2
            reason = f"Memory {memory2.id} scored higher ({final_score2:.4f} vs {final_score1:.4f})"
            outcome = DecisionOutcome.KEPT_SECOND
        
        # Create decision record
        decision = RevisionDecision(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            kept_memory_id=kept_memory.id,
            rejected_memory_id=rejected_memory.id,
            reason=reason,
            weights=self.weights.copy(),
            scores={
                "memory1": scores1,
                "memory2": scores2,
                "final1": final_score1,
                "final2": final_score2
            },
            outcome=outcome
        )
        
        # Log the decision
        logger.info(f"Decision: {reason}")
        self.decisions.append(decision)
        
        # Update trust levels based on the decision
        if hasattr(kept_memory, 'source') and hasattr(rejected_memory, 'source'):
            self._update_trust_based_on_decision(
                kept_memory.source, rejected_memory.source
            )
        
        # Remove the rejected memory from the store
        self.memory_store.delete_memory(rejected_memory.id)
        
        return decision
    
    def _calculate_memory_scores(self, memory: Memory) -> Dict[str, float]:
        """
        Calculate scores for a memory across different factors.
        
        Args:
            memory: Memory to calculate scores for
            
        Returns:
            Dictionary of scores by factor
        """
        scores = {}
        
        # Coherence: how well this memory fits with others
        all_memories = self.memory_store.get_all_memories()
        coherence_scores = [
            self.coherence_scorer.calculate_coherence(memory, other)
            for other in all_memories
            if other.id != memory.id
        ]
        scores["coherence"] = sum(coherence_scores) / max(1, len(coherence_scores))
        
        # Trust: based on the memory's source if available
        source = getattr(memory, 'source', None)
        scores["trust"] = self.trust_sources.get(source, 0.5) if source else 0.5
        
        # Recency: newer memories are preferred
        if hasattr(memory, 'created_at') and memory.created_at:
            # Calculate age in days (capped at 30 days)
            age_days = min(30, (datetime.now() - memory.created_at).total_seconds() / 86400)
            # Convert to 0-1 score (1 being newest)
            scores["recency"] = 1.0 - (age_days / 30.0)
        else:
            scores["recency"] = 0.5  # Default if no timestamp
        
        # Entropy: lower entropy is better (less chaotic/ambiguous)
        if hasattr(memory, 'entropy'):
            # Convert to 0-1 score (1 being lowest entropy)
            scores["entropy"] = 1.0 - min(1.0, memory.entropy)
        else:
            scores["entropy"] = 0.5  # Default if no entropy
        
        return scores
    
    def _update_trust_based_on_decision(
        self,
        kept_source: str,
        rejected_source: str,
        adjustment: float = 0.05
    ) -> None:
        """
        Update trust levels based on a belief revision decision.
        
        Args:
            kept_source: Source of the kept memory
            rejected_source: Source of the rejected memory
            adjustment: Amount to adjust trust levels (0.0-1.0)
        """
        # Increase trust in kept source
        current_kept_trust = self.trust_sources.get(kept_source, 0.5)
        new_kept_trust = min(1.0, current_kept_trust + adjustment)
        self.trust_sources[kept_source] = new_kept_trust
        
        # Decrease trust in rejected source
        current_rejected_trust = self.trust_sources.get(rejected_source, 0.5)
        new_rejected_trust = max(0.0, current_rejected_trust - adjustment)
        self.trust_sources[rejected_source] = new_rejected_trust
        
        logger.info(
            f"Updated trust levels - {kept_source}: {current_kept_trust:.2f} → {new_kept_trust:.2f}, "
            f"{rejected_source}: {current_rejected_trust:.2f} → {new_rejected_trust:.2f}"
        )
        
        self.stats["trust_adjustments"] += 1
    
    def get_trust_levels(self) -> Dict[str, float]:
        """
        Get current trust levels for all sources.
        
        Returns:
            Dictionary of source names to trust levels (0.0-1.0)
        """
        return self.trust_sources.copy()
    
    def set_trust_level(self, source: str, level: float) -> None:
        """
        Set the trust level for a specific source.
        
        Args:
            source: Name of the source
            level: Trust level (0.0-1.0)
        """
        if not 0.0 <= level <= 1.0:
            logger.warning(f"Trust level must be between 0.0 and 1.0, got {level}. Clamping.")
            level = max(0.0, min(1.0, level))
        
        self.trust_sources[source] = level
        logger.info(f"Set trust level for {source} to {level:.2f}")
    
    def get_decision_history(
        self,
        limit: Optional[int] = None,
        include_details: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get history of belief revision decisions.
        
        Args:
            limit: Maximum number of decisions to return, None for all
            include_details: Whether to include detailed scores
            
        Returns:
            List of decision dictionaries
        """
        # Sort by timestamp (newest first)
        sorted_decisions = sorted(
            self.decisions,
            key=lambda d: d.timestamp,
            reverse=True
        )
        
        # Apply limit
        if limit is not None:
            sorted_decisions = sorted_decisions[:limit]
        
        # Convert to dictionaries
        if include_details:
            return [d._asdict() for d in sorted_decisions]
        else:
            # Simplified version without detailed scores
            return [
                {
                    "id": d.id,
                    "timestamp": d.timestamp,
                    "kept_memory_id": d.kept_memory_id,
                    "rejected_memory_id": d.rejected_memory_id,
                    "reason": d.reason,
                    "outcome": d.outcome.name
                }
                for d in sorted_decisions
            ]
    
    def get_quarantined_contradictions(self) -> List[Dict[str, Any]]:
        """
        Get list of quarantined contradictions awaiting resolution.
        
        Returns:
            List of quarantined contradiction dictionaries
        """
        result = []
        for memory1_id, memory2_id, quarantine_id in self.quarantined_pairs:
            mem1 = self.memory_store.get_memory(memory1_id)
            mem2 = self.memory_store.get_memory(memory2_id)
            
            if mem1 and mem2:
                # Find the corresponding decision
                decision = next(
                    (d for d in self.decisions if d.id == quarantine_id),
                    None
                )
                
                result.append({
                    "quarantine_id": quarantine_id,
                    "memory1_id": memory1_id,
                    "memory2_id": memory2_id,
                    "memory1_text": getattr(mem1, 'text', ''),
                    "memory2_text": getattr(mem2, 'text', ''),
                    "quarantined_at": decision.timestamp if decision else datetime.now(),
                    "scores": decision.scores if decision else {}
                })
        
        return result
    
    def resolve_quarantined_contradiction(
        self,
        quarantine_id: str,
        keep_memory_id: Optional[str] = None,
        keep_both: bool = False
    ) -> Dict[str, Any]:
        """
        Resolve a previously quarantined contradiction.
        
        Args:
            quarantine_id: ID of the quarantined contradiction
            keep_memory_id: ID of the memory to keep, None to keep both
            keep_both: Whether to keep both memories
            
        Returns:
            Result of the resolution
        """
        # Find the quarantined pair
        pair_index = None
        for i, (mem1_id, mem2_id, q_id) in enumerate(self.quarantined_pairs):
            if q_id == quarantine_id:
                pair_index = i
                break
        
        if pair_index is None:
            return {
                "success": False,
                "error": f"Quarantined contradiction {quarantine_id} not found"
            }
        
        memory1_id, memory2_id, _ = self.quarantined_pairs[pair_index]
        
        # Remove from quarantine list
        del self.quarantined_pairs[pair_index]
        
        # If keeping both, just update the decision
        if keep_both or keep_memory_id is None:
            # Find and update the decision
            for i, decision in enumerate(self.decisions):
                if decision.id == quarantine_id:
                    # Create updated decision
                    updated_decision = RevisionDecision(
                        id=decision.id,
                        timestamp=decision.timestamp,
                        kept_memory_id=memory1_id,
                        rejected_memory_id=memory2_id,
                        reason=f"Manual resolution: keeping both memories",
                        weights=decision.weights,
                        scores=decision.scores,
                        outcome=DecisionOutcome.KEPT_BOTH
                    )
                    
                    # Replace decision
                    self.decisions[i] = updated_decision
                    
                    return {
                        "success": True,
                        "action": "kept_both",
                        "memory1_id": memory1_id,
                        "memory2_id": memory2_id
                    }
        
        # If keeping one memory, delete the other
        if keep_memory_id in [memory1_id, memory2_id]:
            reject_memory_id = memory2_id if keep_memory_id == memory1_id else memory1_id
            
            # Remove the rejected memory
            self.memory_store.delete_memory(reject_memory_id)
            
            # Find and update the decision
            for i, decision in enumerate(self.decisions):
                if decision.id == quarantine_id:
                    # Create updated decision
                    updated_decision = RevisionDecision(
                        id=decision.id,
                        timestamp=decision.timestamp,
                        kept_memory_id=keep_memory_id,
                        rejected_memory_id=reject_memory_id,
                        reason=f"Manual resolution: keeping memory {keep_memory_id}",
                        weights=decision.weights,
                        scores=decision.scores,
                        outcome=(
                            DecisionOutcome.KEPT_FIRST 
                            if keep_memory_id == memory1_id 
                            else DecisionOutcome.KEPT_SECOND
                        )
                    )
                    
                    # Replace decision
                    self.decisions[i] = updated_decision
                    
                    # Update memory statistics
                    self.stats["memories_kept"] += 1
                    self.stats["memories_rejected"] += 1
                    self.stats["memories_quarantined"] -= 2
                    
                    # Update trust levels if sources available
                    mem1 = self.memory_store.get_memory(memory1_id)
                    mem2 = self.memory_store.get_memory(memory2_id)
                    
                    if (mem1 and mem2 and hasattr(mem1, 'source') 
                            and hasattr(mem2, 'source')):
                        if keep_memory_id == memory1_id:
                            self._update_trust_based_on_decision(
                                mem1.source, mem2.source
                            )
                        else:
                            self._update_trust_based_on_decision(
                                mem2.source, mem1.source
                            )
                    
                    return {
                        "success": True,
                        "action": "kept_one",
                        "kept_memory_id": keep_memory_id,
                        "rejected_memory_id": reject_memory_id
                    }
        
        return {
            "success": False,
            "error": f"Invalid memory ID {keep_memory_id}"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the belief revision system.
        
        Returns:
            Dictionary of statistics
        """
        # Add current state information
        stats = {
            **self.stats,
            "total_decisions": len(self.decisions),
            "quarantined_count": len(self.quarantined_pairs),
            "trust_sources_count": len(self.trust_sources),
            "current_decision_strategy": self.decision_strategy.name,
            "current_weights": self.weights
        }
        
        # Calculate outcomes distribution
        outcome_counts = {}
        for decision in self.decisions:
            outcome = decision.outcome.name
            if outcome not in outcome_counts:
                outcome_counts[outcome] = 0
            outcome_counts[outcome] += 1
        
        stats["outcome_distribution"] = outcome_counts
        
        return stats 