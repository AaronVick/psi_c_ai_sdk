"""
Epistemic Status Tracker Module for ΨC-AI SDK

This module implements a system for tracking the epistemic status of beliefs and memories,
including classification of knowledge states, confidence calculation, and uncertainty propagation.
It provides tools for maintaining an accurate model of what the system knows, believes, 
and is uncertain about.
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from collections import defaultdict, Counter

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.beliefs.contradiction import ContradictionDetector
from psi_c_ai_sdk.beliefs.revision import BeliefRevisionSystem
from psi_c_ai_sdk.memory.coherence import CoherenceScorer

logger = logging.getLogger(__name__)


class KnowledgeState(Enum):
    """
    Possible epistemic states for a belief or memory.
    
    These states represent the system's level of certainty and the 
    stability of a particular piece of knowledge.
    """
    # High stability, high certainty - well-established knowledge
    ESTABLISHED = auto()
    
    # High stability, medium certainty - reliable but not fully verified 
    CONFIDENT = auto()
    
    # Medium stability, medium certainty - somewhat reliable but could change
    PROBABLE = auto()
    
    # Medium stability, low certainty - tentative knowledge
    TENTATIVE = auto()
    
    # Low stability, low certainty - highly uncertain and subject to revision
    UNCERTAIN = auto()
    
    # Unstable state due to contradictions or rapidly changing evidence
    UNSTABLE = auto()
    
    # Explicit stance of having insufficient information
    AGNOSTIC = auto()
    
    # Explicitly recognized as unknown or unverifiable
    UNKNOWN = auto()


class EpistemicClassifier:
    """
    Classifies memories and beliefs into epistemic states based on their
    properties and history.
    """
    
    def __init__(
        self,
        stability_threshold: float = 0.6,
        certainty_threshold: float = 0.7,
        confidence_decay_rate: float = 0.05,
        history_window: int = 10
    ):
        """
        Initialize the epistemic classifier.
        
        Args:
            stability_threshold: Threshold for considering a belief stable
            certainty_threshold: Threshold for considering a belief certain
            confidence_decay_rate: Rate at which confidence decays over time
            history_window: Number of past states to track for stability analysis
        """
        self.stability_threshold = stability_threshold
        self.certainty_threshold = certainty_threshold
        self.confidence_decay_rate = confidence_decay_rate
        self.history_window = history_window
        
        # Track state history for each memory/belief
        self.state_history: Dict[str, List[Tuple[KnowledgeState, datetime]]] = defaultdict(list)
        
        # Track contradictions encountered for each memory
        self.contradiction_counts: Dict[str, int] = defaultdict(int)
        
        # Track verification attempts and successes
        self.verification_history: Dict[str, List[bool]] = defaultdict(list)

    def classify(
        self, 
        memory: Memory, 
        additional_context: Optional[Dict[str, Any]] = None
    ) -> KnowledgeState:
        """
        Classify a memory into an epistemic state based on its properties.
        
        Args:
            memory: The memory to classify
            additional_context: Optional additional context for classification
                May include contradiction information, verification results, etc.
        
        Returns:
            The epistemic state of the memory
        """
        context = additional_context or {}
        
        # Extract relevant factors
        confidence = self._calculate_confidence(memory, context)
        stability = self._calculate_stability(memory.id, context)
        contradictions = context.get("contradictions", self.contradiction_counts.get(memory.id, 0))
        verifications = context.get("verifications", len(self.verification_history.get(memory.id, [])))
        verification_success_rate = self._get_verification_success_rate(memory.id)
        
        # Check if we know nothing about this topic
        if memory.metadata.get("is_placeholder", False) or context.get("is_knowledge_gap", False):
            return KnowledgeState.UNKNOWN
        
        # Check if we have insufficient information
        if context.get("insufficient_information", False) or confidence < 0.3:
            return KnowledgeState.AGNOSTIC
        
        # Check for unstable state with active contradictions
        if contradictions > 0 and stability < self.stability_threshold:
            return KnowledgeState.UNSTABLE
        
        # High stability and high certainty
        if stability > 0.8 and confidence > 0.85 and verification_success_rate > 0.9:
            return KnowledgeState.ESTABLISHED
        
        # High stability and medium-high certainty
        if stability > 0.7 and confidence > 0.75:
            return KnowledgeState.CONFIDENT
        
        # Medium stability and medium certainty  
        if stability > 0.5 and confidence > 0.6:
            return KnowledgeState.PROBABLE
        
        # Medium-low stability or certainty
        if stability > 0.4 or confidence > 0.4:
            return KnowledgeState.TENTATIVE
        
        # Low stability and certainty
        return KnowledgeState.UNCERTAIN
    
    def _calculate_confidence(
        self, 
        memory: Memory, 
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate the confidence level for a memory.
        
        Args:
            memory: The memory to calculate confidence for
            context: Additional context for calculation
            
        Returns:
            Confidence score between 0 and 1
        """
        # Start with base confidence from memory metadata, if available
        base_confidence = memory.metadata.get("confidence", 0.5)
        
        # Adjust based on source reliability
        source_trust = context.get("source_trust", 0.5)
        
        # Adjust based on time since creation (confidence decays)
        time_factor = 1.0
        if memory.creation_time:
            days_since_creation = (datetime.now() - memory.creation_time).days
            time_factor = max(0.3, 1.0 - (days_since_creation * self.confidence_decay_rate))
        
        # Adjust based on verification history
        verification_boost = 0.0
        verifications = self.verification_history.get(memory.id, [])
        if verifications:
            verification_boost = sum(verifications) / max(1, len(verifications)) * 0.2
        
        # Adjust based on coherence with other memories
        coherence_factor = context.get("coherence", 0.5)
        
        # Combine factors with appropriate weights
        confidence = (
            (base_confidence * 0.3) +
            (source_trust * 0.25) +
            (time_factor * 0.15) +
            (verification_boost * 0.1) +
            (coherence_factor * 0.2)
        )
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_stability(
        self, 
        memory_id: str, 
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate the stability of a memory based on its history.
        
        Args:
            memory_id: ID of the memory to check stability for
            context: Additional context for calculation
            
        Returns:
            Stability score between 0 and 1
        """
        # Check state history for this memory
        history = self.state_history.get(memory_id, [])
        
        # If no history, use default medium stability
        if not history:
            return 0.5
        
        # Calculate volatility based on state changes
        changes = 0
        for i in range(1, min(self.history_window, len(history))):
            if history[i][0] != history[i-1][0]:
                changes += 1
        
        volatility = changes / max(1, min(self.history_window, len(history)) - 1)
        
        # Calculate recency-weighted stability 
        # (more weight to recent stability/volatility)
        stability = 1.0 - volatility
        
        # Adjust for contradictions
        contradiction_penalty = min(0.5, self.contradiction_counts.get(memory_id, 0) * 0.1)
        stability -= contradiction_penalty
        
        # Consider explicit stability indicators from context
        if "stability_override" in context:
            return context["stability_override"]
        
        return min(1.0, max(0.0, stability))
    
    def update_verification(self, memory_id: str, was_verified: bool) -> None:
        """
        Update verification history for a memory.
        
        Args:
            memory_id: ID of the memory
            was_verified: Whether verification was successful
        """
        history = self.verification_history.get(memory_id, [])
        history.append(was_verified)
        # Keep only recent verification history
        if len(history) > self.history_window:
            history = history[-self.history_window:]
        self.verification_history[memory_id] = history
    
    def update_contradiction_count(self, memory_id: str, increment: int = 1) -> None:
        """
        Update contradiction count for a memory.
        
        Args:
            memory_id: ID of the memory
            increment: Amount to increment by (can be negative to reduce contradictions)
        """
        current = self.contradiction_counts.get(memory_id, 0)
        self.contradiction_counts[memory_id] = max(0, current + increment)
    
    def update_state_history(self, memory_id: str, state: KnowledgeState) -> None:
        """
        Update state history for a memory.
        
        Args:
            memory_id: ID of the memory
            state: New epistemic state
        """
        history = self.state_history.get(memory_id, [])
        history.append((state, datetime.now()))
        # Keep only recent history
        if len(history) > self.history_window:
            history = history[-self.history_window:]
        self.state_history[memory_id] = history
    
    def _get_verification_success_rate(self, memory_id: str) -> float:
        """
        Get the verification success rate for a memory.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Verification success rate between 0 and 1
        """
        verifications = self.verification_history.get(memory_id, [])
        if not verifications:
            return 0.5  # Default neutral value
        return sum(verifications) / len(verifications)


class UncertaintyPropagator:
    """
    Propagates uncertainty between beliefs and memories that are connected
    semantically or logically.
    """
    
    def __init__(
        self,
        coherence_scorer: CoherenceScorer,
        propagation_threshold: float = 0.7,
        uncertainty_decay: float = 0.2
    ):
        """
        Initialize the uncertainty propagator.
        
        Args:
            coherence_scorer: Scorer to determine relationships between memories
            propagation_threshold: Minimum coherence required for propagation
            uncertainty_decay: How much uncertainty decays during propagation
        """
        self.coherence_scorer = coherence_scorer
        self.propagation_threshold = propagation_threshold
        self.uncertainty_decay = uncertainty_decay
        
        # Cached influence network for fast propagation
        self.influence_network: Dict[str, List[Tuple[str, float]]] = {}
        
        # Track propagation history to prevent cycles
        self.propagation_history: Dict[str, Set[str]] = defaultdict(set)
    
    def build_influence_network(self, memories: List[Memory]) -> None:
        """
        Build or update the influence network between memories.
        
        Args:
            memories: List of memories to include in the network
        """
        # Build influence relationships between memories
        for i, memory1 in enumerate(memories):
            self.influence_network.setdefault(memory1.id, [])
            
            for j, memory2 in enumerate(memories):
                if i == j:
                    continue
                
                # Calculate coherence between memories
                coherence = self.coherence_scorer.calculate_coherence(memory1, memory2)
                
                # Only include relationships above threshold
                if coherence >= self.propagation_threshold:
                    self.influence_network[memory1.id].append((memory2.id, coherence))
    
    def propagate_uncertainty(
        self, 
        source_memory_id: str, 
        uncertainty: float,
        visited: Optional[Set[str]] = None,
        max_depth: int = 3
    ) -> Dict[str, float]:
        """
        Propagate uncertainty from a source memory to related memories.
        
        Args:
            source_memory_id: ID of the source memory
            uncertainty: Uncertainty value to propagate (0-1)
            visited: Set of already visited memory IDs
            max_depth: Maximum propagation depth
            
        Returns:
            Dictionary mapping memory IDs to propagated uncertainty values
        """
        if max_depth <= 0 or uncertainty <= 0.05:
            return {}
        
        if visited is None:
            visited = set()
        
        # Prevent cycles and redundant propagation
        if source_memory_id in visited:
            return {}
        
        visited.add(source_memory_id)
        
        # Get connected memories
        influenced_memories = self.influence_network.get(source_memory_id, [])
        results = {}
        
        # Propagate to each connected memory
        for target_id, coherence in influenced_memories:
            # Calculate propagated uncertainty
            propagated_uncertainty = uncertainty * coherence * (1 - self.uncertainty_decay)
            
            # Only propagate significant uncertainty
            if propagated_uncertainty >= 0.05:
                results[target_id] = propagated_uncertainty
                
                # Recursive propagation
                downstream = self.propagate_uncertainty(
                    target_id, 
                    propagated_uncertainty,
                    visited.copy(),
                    max_depth - 1
                )
                
                # Combine results
                for memory_id, value in downstream.items():
                    results[memory_id] = max(results.get(memory_id, 0), value)
        
        return results


class EpistemicStatusTracker:
    """
    Tracks the epistemic status of the AI system's knowledge and beliefs,
    providing insights into what it knows, what it's uncertain about, and 
    what it needs to learn.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        classifier: Optional[EpistemicClassifier] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        coherence_scorer: Optional[CoherenceScorer] = None,
        uncertainty_propagator: Optional[UncertaintyPropagator] = None,
        belief_revision_system: Optional[BeliefRevisionSystem] = None,
        scan_interval: int = 60
    ):
        """
        Initialize the epistemic status tracker.
        
        Args:
            memory_store: Store containing memories to track
            classifier: Classifier for epistemic states
            contradiction_detector: Detector for contradictions between memories
            coherence_scorer: Scorer for memory coherence
            uncertainty_propagator: Propagator for uncertainty between memories
            belief_revision_system: System for resolving contradictions
            scan_interval: Interval between full scans (in seconds)
        """
        self.memory_store = memory_store
        self.classifier = classifier or EpistemicClassifier()
        self.contradiction_detector = contradiction_detector
        self.coherence_scorer = coherence_scorer or CoherenceScorer()
        
        # Set up uncertainty propagator if not provided
        if uncertainty_propagator is None and coherence_scorer is not None:
            self.uncertainty_propagator = UncertaintyPropagator(coherence_scorer)
        else:
            self.uncertainty_propagator = uncertainty_propagator
        
        self.belief_revision_system = belief_revision_system
        self.scan_interval = scan_interval
        
        # Last full scan timestamp
        self.last_scan_time = datetime(1970, 1, 1)
        
        # Track memory epistemic status
        self.memory_status: Dict[str, KnowledgeState] = {}
        
        # Track memory confidence scores
        self.memory_confidence: Dict[str, float] = {}
        
        # Track known knowledge gaps
        self.knowledge_gaps: List[Dict[str, Any]] = []
        
        # Metrics tracking
        self.metrics: Dict[str, Any] = {
            "status_distribution": Counter(),
            "average_confidence": 0.0,
            "contradiction_rate": 0.0,
            "knowledge_gap_count": 0,
            "stability_score": 0.0,
            "update_history": []
        }
    
    def scan_all_memories(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform a complete scan of all memories to update epistemic status.
        
        Args:
            force: Force scan even if interval hasn't elapsed
            
        Returns:
            Dictionary with scan results and statistics
        """
        now = datetime.now()
        seconds_since_scan = (now - self.last_scan_time).total_seconds()
        
        # Skip if scan interval hasn't elapsed and not forced
        if not force and seconds_since_scan < self.scan_interval:
            logger.debug(f"Skipping scan, last scan was {seconds_since_scan:.1f}s ago")
            return {
                "scanned": False,
                "reason": "interval not elapsed",
                "next_scan_in": self.scan_interval - seconds_since_scan
            }
        
        logger.info("Performing full epistemic status scan")
        start_time = time.time()
        
        # Get all memories
        all_memories = self.memory_store.get_all()
        memory_count = len(all_memories)
        
        # Detect contradictions if detector is available
        contradictions = []
        if self.contradiction_detector is not None:
            contradictions = self.contradiction_detector.find_contradictions(all_memories)
            
            # Update contradiction counts in classifier
            for memory1, memory2, _ in contradictions:
                self.classifier.update_contradiction_count(memory1.id)
                self.classifier.update_contradiction_count(memory2.id)
        
        # Build influence network for uncertainty propagation
        if self.uncertainty_propagator is not None:
            self.uncertainty_propagator.build_influence_network(all_memories)
        
        # Calculate average coherence
        avg_coherence = 0.0
        coherence_count = 0
        if len(all_memories) > 1 and self.coherence_scorer is not None:
            for i, mem1 in enumerate(all_memories[:100]):  # Limit to prevent quadratic explosion
                for mem2 in all_memories[i+1:100]:
                    avg_coherence += self.coherence_scorer.calculate_coherence(mem1, mem2)
                    coherence_count += 1
            
            if coherence_count > 0:
                avg_coherence /= coherence_count
        
        # Reset status distribution
        status_distribution = Counter()
        total_confidence = 0.0
        
        # Process each memory
        for memory in all_memories:
            # Context for classification
            context = {
                "contradictions": sum(1 for m1, m2, _ in contradictions 
                                    if m1.id == memory.id or m2.id == memory.id),
                "coherence": avg_coherence,
                "source_trust": self._get_source_trust(memory.source)
            }
            
            # Classify memory
            status = self.classifier.classify(memory, context)
            
            # Calculate confidence
            confidence = self.classifier._calculate_confidence(memory, context)
            
            # Update tracking
            self.memory_status[memory.id] = status
            self.memory_confidence[memory.id] = confidence
            
            # Update classifier history
            self.classifier.update_state_history(memory.id, status)
            
            # Update metrics
            status_distribution[status] += 1
            total_confidence += confidence
        
        # Update metrics
        elapsed_time = time.time() - start_time
        self.last_scan_time = now
        
        # Calculate and update metrics
        self.metrics["status_distribution"] = dict(status_distribution)
        self.metrics["average_confidence"] = total_confidence / max(1, memory_count)
        self.metrics["contradiction_rate"] = len(contradictions) / max(1, memory_count)
        self.metrics["knowledge_gap_count"] = len(self.knowledge_gaps)
        
        # Calculate stability as percentage of stable vs unstable memories
        stable_states = [KnowledgeState.ESTABLISHED, KnowledgeState.CONFIDENT, KnowledgeState.PROBABLE]
        stable_count = sum(count for state, count in status_distribution.items() if state in stable_states)
        self.metrics["stability_score"] = stable_count / max(1, memory_count)
        
        # Add scan to history
        self.metrics["update_history"].append({
            "timestamp": now,
            "memory_count": memory_count,
            "contradiction_count": len(contradictions),
            "avg_confidence": self.metrics["average_confidence"],
            "stability_score": self.metrics["stability_score"]
        })
        
        # Trim history to last 100 entries
        if len(self.metrics["update_history"]) > 100:
            self.metrics["update_history"] = self.metrics["update_history"][-100:]
        
        logger.info(f"Epistemic scan completed in {elapsed_time:.2f}s, processed {memory_count} memories")
        
        return {
            "scanned": True,
            "memory_count": memory_count,
            "contradictions": len(contradictions),
            "status_distribution": dict(status_distribution),
            "elapsed_time": elapsed_time
        }
    
    def update_memory_status(self, memory: Memory) -> KnowledgeState:
        """
        Update the epistemic status of a single memory.
        
        Args:
            memory: Memory to update
            
        Returns:
            The updated epistemic state
        """
        context = {
            "source_trust": self._get_source_trust(memory.source),
            # Use cached coherence as a simplification
            "coherence": self.metrics.get("average_coherence", 0.5)
        }
        
        # Check for contradictions
        if self.contradiction_detector is not None:
            all_memories = self.memory_store.get_all()
            contradictions = 0
            
            for other_memory in all_memories:
                if memory.id == other_memory.id:
                    continue
                
                # Simple check for contradictions against this specific memory
                if self.contradiction_detector.check_contradiction(memory, other_memory)[0]:
                    contradictions += 1
                    
            context["contradictions"] = contradictions
        
        # Classify memory
        status = self.classifier.classify(memory, context)
        confidence = self.classifier._calculate_confidence(memory, context)
        
        # Update tracking
        self.memory_status[memory.id] = status
        self.memory_confidence[memory.id] = confidence
        
        # Update history
        self.classifier.update_state_history(memory.id, status)
        
        return status
    
    def propagate_status_change(
        self, 
        memory_id: str,
        new_status: KnowledgeState,
        old_status: Optional[KnowledgeState] = None
    ) -> Dict[str, KnowledgeState]:
        """
        Propagate epistemic status changes to related memories.
        
        Args:
            memory_id: ID of the memory whose status changed
            new_status: New status of the memory
            old_status: Previous status of the memory (if known)
            
        Returns:
            Dictionary mapping memory IDs to their new status
        """
        if self.uncertainty_propagator is None:
            return {}
        
        # Skip propagation for stable transitions
        stable_states = [KnowledgeState.ESTABLISHED, KnowledgeState.CONFIDENT]
        if new_status in stable_states and (old_status is None or old_status in stable_states):
            return {}
        
        # Calculate uncertainty based on status
        uncertainty_map = {
            KnowledgeState.ESTABLISHED: 0.1,
            KnowledgeState.CONFIDENT: 0.2,
            KnowledgeState.PROBABLE: 0.4,
            KnowledgeState.TENTATIVE: 0.6,
            KnowledgeState.UNCERTAIN: 0.8,
            KnowledgeState.UNSTABLE: 0.9,
            KnowledgeState.AGNOSTIC: 0.7,
            KnowledgeState.UNKNOWN: 0.8,
        }
        
        uncertainty = uncertainty_map.get(new_status, 0.5)
        
        # Only propagate if uncertainty is significant
        if uncertainty < 0.3:
            return {}
        
        # Propagate uncertainty
        propagated = self.uncertainty_propagator.propagate_uncertainty(memory_id, uncertainty)
        
        # Update status of affected memories
        affected_statuses = {}
        for target_id, propagated_uncertainty in propagated.items():
            # Skip if not in memory store
            if target_id not in self.memory_status:
                continue
                
            current_status = self.memory_status[target_id]
            
            # Only downgrade status based on propagated uncertainty
            if propagated_uncertainty >= 0.7 and current_status in stable_states:
                new_target_status = KnowledgeState.PROBABLE
                self.memory_status[target_id] = new_target_status
                self.classifier.update_state_history(target_id, new_target_status)
                affected_statuses[target_id] = new_target_status
                
            elif propagated_uncertainty >= 0.5 and current_status in [KnowledgeState.ESTABLISHED, 
                                                                     KnowledgeState.CONFIDENT,
                                                                     KnowledgeState.PROBABLE]:
                new_target_status = KnowledgeState.TENTATIVE
                self.memory_status[target_id] = new_target_status
                self.classifier.update_state_history(target_id, new_target_status)
                affected_statuses[target_id] = new_target_status
        
        return affected_statuses
    
    def register_knowledge_gap(
        self, 
        topic: str,
        importance: float = 0.5,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a knowledge gap that the system has identified.
        
        Args:
            topic: Description of the knowledge gap
            importance: Importance of filling this gap (0-1)
            context: Additional context about the gap
            
        Returns:
            ID of the registered knowledge gap
        """
        gap_id = f"gap_{int(time.time())}_{len(self.knowledge_gaps)}"
        
        gap = {
            "id": gap_id,
            "topic": topic,
            "importance": importance,
            "identified_at": datetime.now(),
            "status": "open",
            "context": context or {}
        }
        
        self.knowledge_gaps.append(gap)
        self.metrics["knowledge_gap_count"] = len(self.knowledge_gaps)
        
        logger.info(f"Registered knowledge gap: {topic} (importance: {importance:.2f})")
        return gap_id
    
    def mark_knowledge_gap_resolved(self, gap_id: str, resolution: Optional[str] = None) -> bool:
        """
        Mark a knowledge gap as resolved.
        
        Args:
            gap_id: ID of the knowledge gap
            resolution: Optional description of how it was resolved
            
        Returns:
            Whether the operation was successful
        """
        for gap in self.knowledge_gaps:
            if gap["id"] == gap_id:
                gap["status"] = "resolved"
                gap["resolved_at"] = datetime.now()
                if resolution:
                    gap["resolution"] = resolution
                
                logger.info(f"Marked knowledge gap as resolved: {gap['topic']}")
                return True
        
        return False
    
    def get_memory_status(self, memory_id: str) -> Optional[KnowledgeState]:
        """
        Get the epistemic status of a memory.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Epistemic status if found, None otherwise
        """
        return self.memory_status.get(memory_id)
    
    def get_memory_confidence(self, memory_id: str) -> float:
        """
        Get the confidence score for a memory.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Confidence score (0-1), defaults to 0 if not found
        """
        return self.memory_confidence.get(memory_id, 0.0)
    
    def get_memories_by_status(self, status: KnowledgeState) -> List[str]:
        """
        Get all memory IDs with a specific epistemic status.
        
        Args:
            status: Epistemic status to filter by
            
        Returns:
            List of memory IDs with the specified status
        """
        return [memory_id for memory_id, mem_status in self.memory_status.items()
                if mem_status == status]
    
    def get_open_knowledge_gaps(self, min_importance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get all open knowledge gaps, optionally filtered by importance.
        
        Args:
            min_importance: Minimum importance threshold
            
        Returns:
            List of open knowledge gaps
        """
        return [gap for gap in self.knowledge_gaps 
                if gap["status"] == "open" and gap["importance"] >= min_importance]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current epistemic metrics.
        
        Returns:
            Dictionary with current metrics
        """
        return self.metrics
    
    def _get_source_trust(self, source: str) -> float:
        """
        Get the trust level for a memory source.
        
        Args:
            source: Name of the source
            
        Returns:
            Trust level (0-1)
        """
        if self.belief_revision_system is not None:
            return self.belief_revision_system.get_trust_level(source)
        
        # Default moderate trust
        return 0.5 