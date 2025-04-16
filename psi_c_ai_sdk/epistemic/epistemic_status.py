"""
Epistemic Status Tracking for ΨC-AI SDK

This module provides tools for tracking the epistemic status of memories and beliefs,
classifying them into different states (such as open, closed, unstable, or agnostic),
calculating confidence levels, and propagating uncertainty through the knowledge graph.
"""

import logging
import uuid
from typing import Dict, List, Set, Tuple, Optional, Any, NamedTuple, Enum
from datetime import datetime
import math

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore
from psi_c_ai_sdk.contradiction.detector import ContradictionDetector
from psi_c_ai_sdk.coherence.scorer import CoherenceScorer
from psi_c_ai_sdk.schema.schema import SchemaGraph
from psi_c_ai_sdk.reflection.engine import ReflectionEngine

logger = logging.getLogger(__name__)


class EpistemicState(Enum):
    """Classification of a belief's epistemic state."""
    
    UNSTABLE = "unstable"      # Belief is in flux, contradictory evidence exists
    UNCERTAIN = "uncertain"    # Belief has low confidence but no direct contradictions
    STABLE = "stable"          # Belief is established with moderate confidence
    CONFIDENT = "confident"    # Belief is well-supported with high confidence
    UNKNOWN = "unknown"        # Not enough information to classify


class ReflectionTrigger(Enum):
    """Triggers for reflection based on epistemic conditions."""
    
    CRITICAL_CONTRADICTION = "critical_contradiction"  # Severe contradiction detected
    CERTAINTY_THRESHOLD = "certainty_threshold"        # Certainty below threshold
    STABILITY_BREACH = "stability_breach"              # Stability metric degraded
    COHERENCE_COLLAPSE = "coherence_collapse"          # Coherence dropped significantly
    KNOWLEDGE_GAP = "knowledge_gap"                    # Missing critical connections
    COGNITIVE_DEBT = "cognitive_debt"                  # Too many unresolved issues


class EpistemicMetrics(NamedTuple):
    """Key metrics associated with epistemic status."""
    
    certainty: float           # Overall certainty score (0-1)
    stability: float           # Stability of beliefs over time (0-1)
    coherence: float           # Internal coherence (0-1)
    contradictions: int        # Number of active contradictions
    knowledge_gaps: int        # Number of identified knowledge gaps
    cognitive_debt: float      # Measure of unresolved epistemic issues


class EpistemicStatus:
    """
    System for tracking the epistemic status of memories and beliefs.
    
    This class provides tools for classifying beliefs into different epistemic
    states, calculating confidence levels, detecting knowledge gaps, and
    triggering reflections when necessary.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        coherence_scorer: CoherenceScorer,
        contradiction_detector: ContradictionDetector,
        schema_graph: Optional[SchemaGraph] = None,
        reflection_engine: Optional[ReflectionEngine] = None,
        certainty_threshold: float = 0.3,
        stability_threshold: float = 0.4,
        coherence_threshold: float = 0.5,
        tracking_window: int = 10
    ):
        """
        Initialize the epistemic status tracking system.
        
        Args:
            memory_store: Memory store containing beliefs to track
            coherence_scorer: Scorer for measuring memory coherence
            contradiction_detector: Detector for finding contradictions
            schema_graph: Optional schema graph for structural analysis
            reflection_engine: Optional reflection engine for triggering reflections
            certainty_threshold: Threshold below which certainty is concerning
            stability_threshold: Threshold below which stability is concerning
            coherence_threshold: Threshold below which coherence is concerning
            tracking_window: Number of historical states to track
        """
        self.memory_store = memory_store
        self.coherence_scorer = coherence_scorer
        self.contradiction_detector = contradiction_detector
        self.schema_graph = schema_graph
        self.reflection_engine = reflection_engine
        
        # Thresholds for triggering concerns
        self.certainty_threshold = certainty_threshold
        self.stability_threshold = stability_threshold
        self.coherence_threshold = coherence_threshold
        self.tracking_window = tracking_window
        
        # Current state and history
        self.current_metrics: Optional[EpistemicMetrics] = None
        self.metrics_history: List[Tuple[datetime, EpistemicMetrics]] = []
        self.state_history: List[Tuple[datetime, EpistemicState]] = []
        self.current_state = EpistemicState.UNKNOWN
        
        # Tracking for individual memories
        self.memory_states: Dict[str, EpistemicState] = {}
        self.memory_certainty: Dict[str, float] = {}
        
        # Reflection triggers
        self.active_triggers: Set[ReflectionTrigger] = set()
        
        # Statistics
        self.stats = {
            "state_changes": 0,
            "reflection_triggers": 0,
            "certainty_drops": 0,
            "contradiction_resolutions": 0
        }
    
    def update(self) -> EpistemicMetrics:
        """
        Update the epistemic status based on current memory store state.
        
        This method analyzes the current memories, calculates key metrics,
        updates the epistemic state, and identifies any necessary reflection
        triggers.
        
        Returns:
            Current epistemic metrics
        """
        # Calculate current metrics
        metrics = self._calculate_metrics()
        
        # Update metrics history
        now = datetime.now()
        self.metrics_history.append((now, metrics))
        
        # Maintain fixed history size
        if len(self.metrics_history) > self.tracking_window:
            self.metrics_history = self.metrics_history[-self.tracking_window:]
        
        # Determine current epistemic state
        previous_state = self.current_state
        self.current_state = self._determine_state(metrics)
        
        # Update state history
        self.state_history.append((now, self.current_state))
        if len(self.state_history) > self.tracking_window:
            self.state_history = self.state_history[-self.tracking_window:]
        
        # Check for state change
        if previous_state != self.current_state:
            logger.info(f"Epistemic state changed: {previous_state.value} -> {self.current_state.value}")
            self.stats["state_changes"] += 1
        
        # Update individual memory states
        self._update_memory_states()
        
        # Check for reflection triggers
        self._check_reflection_triggers(metrics)
        
        self.current_metrics = metrics
        return metrics
    
    def _calculate_metrics(self) -> EpistemicMetrics:
        """
        Calculate current epistemic metrics.
        
        Returns:
            EpistemicMetrics object with current values
        """
        memories = self.memory_store.get_all_memories()
        if not memories:
            # No memories, return default metrics
            return EpistemicMetrics(
                certainty=0.0,
                stability=1.0,
                coherence=1.0,
                contradictions=0,
                knowledge_gaps=0,
                cognitive_debt=0.0
            )
        
        # Calculate certainty
        certainty_scores = self._calculate_certainty(memories)
        avg_certainty = sum(certainty_scores.values()) / len(certainty_scores)
        
        # Calculate stability (how consistent the belief system is over time)
        stability = self._calculate_stability()
        
        # Calculate coherence (internal consistency of belief system)
        coherence = self.coherence_scorer.calculate_global_coherence(memories)
        
        # Count contradictions
        contradiction_pairs = self.contradiction_detector.find_contradictions(memories)
        contradiction_count = len(contradiction_pairs)
        
        # Estimate knowledge gaps
        knowledge_gaps = self._identify_knowledge_gaps(memories)
        
        # Calculate cognitive debt (unresolved epistemic issues)
        cognitive_debt = (
            (1.0 - avg_certainty) * 0.3 + 
            (1.0 - stability) * 0.3 + 
            (1.0 - coherence) * 0.2 + 
            min(1.0, contradiction_count / 10.0) * 0.1 +
            min(1.0, knowledge_gaps / 5.0) * 0.1
        )
        
        return EpistemicMetrics(
            certainty=avg_certainty,
            stability=stability,
            coherence=coherence,
            contradictions=contradiction_count,
            knowledge_gaps=knowledge_gaps,
            cognitive_debt=cognitive_debt
        )
    
    def _calculate_certainty(self, memories: List[Memory]) -> Dict[str, float]:
        """
        Calculate certainty scores for each memory.
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            Dictionary mapping memory IDs to certainty scores
        """
        certainty_scores = {}
        
        for memory in memories:
            # Start with base certainty from memory if available
            if hasattr(memory, 'certainty') and memory.certainty is not None:
                base_certainty = memory.certainty
            else:
                base_certainty = 0.5  # Default mid-level certainty
            
            # Adjust based on coherence with other memories
            coherence_scores = [
                self.coherence_scorer.calculate_coherence(memory, other)
                for other in memories
                if other.id != memory.id
            ]
            
            avg_coherence = (
                sum(coherence_scores) / len(coherence_scores)
                if coherence_scores else 0.5
            )
            
            # Adjust based on source trust if available
            source_trust = 0.5  # Default mid-level trust
            if hasattr(memory, 'source_trust') and memory.source_trust is not None:
                source_trust = memory.source_trust
            
            # Combine factors weighted by importance
            certainty = (
                base_certainty * 0.4 + 
                avg_coherence * 0.4 + 
                source_trust * 0.2
            )
            
            certainty_scores[memory.id] = certainty
            
            # Update the memory's certainty attribute if possible
            if hasattr(memory, 'certainty'):
                memory.certainty = certainty
        
        return certainty_scores
    
    def _calculate_stability(self) -> float:
        """
        Calculate stability metric based on historical data.
        
        Returns:
            Stability score between 0 and 1
        """
        # If we don't have enough history, assume high stability
        if len(self.metrics_history) < 2:
            return 1.0
        
        # Calculate volatility of key metrics
        certainty_values = [m[1].certainty for m in self.metrics_history]
        coherence_values = [m[1].coherence for m in self.metrics_history]
        
        # Use standard deviation as a measure of volatility
        if len(certainty_values) > 1:
            certainty_volatility = self._calculate_volatility(certainty_values)
            coherence_volatility = self._calculate_volatility(coherence_values)
            
            # Combine volatilities (lower volatility = higher stability)
            combined_volatility = (certainty_volatility + coherence_volatility) / 2.0
            
            # Convert to stability score (0-1)
            stability = max(0.0, 1.0 - (combined_volatility * 5.0))
            return stability
        
        return 1.0  # Default high stability if not enough data
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """
        Calculate volatility (variation) of a list of values.
        
        Args:
            values: List of values to analyze
            
        Returns:
            Volatility metric (standard deviation)
        """
        if len(values) < 2:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _identify_knowledge_gaps(self, memories: List[Memory]) -> int:
        """
        Identify potential knowledge gaps in the memory system.
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            Number of identified knowledge gaps
        """
        # Simple implementation based on low certainty clusters
        # A more sophisticated implementation would use the schema graph
        # to identify structural gaps in knowledge
        
        if not self.schema_graph:
            # Without a schema graph, estimate based on certainty distribution
            certainty_scores = self._calculate_certainty(memories)
            low_certainty_count = sum(1 for score in certainty_scores.values() if score < 0.3)
            return low_certainty_count
        
        # With a schema graph, identify disconnected components and sparse regions
        gaps = 0
        
        # Check for disconnected components
        if hasattr(self.schema_graph, 'get_disconnected_components'):
            components = self.schema_graph.get_disconnected_components()
            gaps += len(components) - 1  # More than one component indicates gaps
        
        # Check for sparse regions (nodes with few connections)
        if hasattr(self.schema_graph, 'get_node_degrees'):
            degrees = self.schema_graph.get_node_degrees()
            sparse_nodes = sum(1 for degree in degrees.values() if degree < 2)
            gaps += min(5, sparse_nodes // 3)  # Cap at 5 for this factor
        
        return gaps
    
    def _determine_state(self, metrics: EpistemicMetrics) -> EpistemicState:
        """
        Determine the overall epistemic state based on metrics.
        
        Args:
            metrics: Current epistemic metrics
            
        Returns:
            Current epistemic state
        """
        # Check for unstable state first (contradictions or very low coherence)
        if metrics.contradictions > 0 or metrics.coherence < 0.3:
            return EpistemicState.UNSTABLE
        
        # Check for uncertain state (low certainty but not unstable)
        if metrics.certainty < self.certainty_threshold:
            return EpistemicState.UNCERTAIN
        
        # Check for confident state (high certainty and stability)
        if metrics.certainty > 0.7 and metrics.stability > 0.7 and metrics.coherence > 0.7:
            return EpistemicState.CONFIDENT
        
        # Default to stable state (moderate metrics)
        return EpistemicState.STABLE
    
    def _update_memory_states(self) -> None:
        """
        Update epistemic states for individual memories.
        """
        memories = self.memory_store.get_all_memories()
        
        # Skip if no memories
        if not memories:
            return
        
        # Find contradictions to identify unstable memories
        contradiction_pairs = self.contradiction_detector.find_contradictions(memories)
        unstable_memory_ids = set()
        for m1, m2, _ in contradiction_pairs:
            unstable_memory_ids.add(m1.id)
            unstable_memory_ids.add(m2.id)
        
        # Calculate certainty for all memories
        certainty_scores = self._calculate_certainty(memories)
        
        # Update states for each memory
        for memory in memories:
            # Store certainty
            self.memory_certainty[memory.id] = certainty_scores[memory.id]
            
            # Determine state
            if memory.id in unstable_memory_ids:
                state = EpistemicState.UNSTABLE
            elif certainty_scores[memory.id] < self.certainty_threshold:
                state = EpistemicState.UNCERTAIN
            elif certainty_scores[memory.id] > 0.7:
                state = EpistemicState.CONFIDENT
            else:
                state = EpistemicState.STABLE
            
            # Store state
            self.memory_states[memory.id] = state
    
    def _check_reflection_triggers(self, metrics: EpistemicMetrics) -> None:
        """
        Check for conditions that should trigger reflection.
        
        Args:
            metrics: Current epistemic metrics
        """
        # Clear previous triggers
        self.active_triggers.clear()
        
        # Check for critical contradictions
        if metrics.contradictions > 3:
            self.active_triggers.add(ReflectionTrigger.CRITICAL_CONTRADICTION)
        
        # Check for certainty below threshold
        if metrics.certainty < self.certainty_threshold:
            self.active_triggers.add(ReflectionTrigger.CERTAINTY_THRESHOLD)
        
        # Check for stability breach
        if metrics.stability < self.stability_threshold:
            self.active_triggers.add(ReflectionTrigger.STABILITY_BREACH)
        
        # Check for coherence collapse
        if metrics.coherence < self.coherence_threshold:
            self.active_triggers.add(ReflectionTrigger.COHERENCE_COLLAPSE)
        
        # Check for knowledge gaps
        if metrics.knowledge_gaps > 3:
            self.active_triggers.add(ReflectionTrigger.KNOWLEDGE_GAP)
        
        # Check for cognitive debt
        if metrics.cognitive_debt > 0.7:
            self.active_triggers.add(ReflectionTrigger.COGNITIVE_DEBT)
        
        # Log triggers and update stats
        if self.active_triggers:
            trigger_names = [t.value for t in self.active_triggers]
            logger.info(f"Reflection triggers active: {', '.join(trigger_names)}")
            self.stats["reflection_triggers"] += len(self.active_triggers)
            
            # Trigger reflection if engine is available
            if self.reflection_engine and hasattr(self.reflection_engine, 'trigger_reflection'):
                for trigger in self.active_triggers:
                    self.reflection_engine.trigger_reflection(
                        trigger=trigger.value,
                        context={
                            "metrics": metrics._asdict(),
                            "state": self.current_state.value,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
    
    def get_metrics(self) -> Optional[EpistemicMetrics]:
        """
        Get current epistemic metrics.
        
        Returns:
            Current metrics or None if not yet calculated
        """
        return self.current_metrics
    
    def get_state(self) -> EpistemicState:
        """
        Get current overall epistemic state.
        
        Returns:
            Current epistemic state
        """
        return self.current_state
    
    def get_memory_state(self, memory_id: str) -> Tuple[EpistemicState, float]:
        """
        Get epistemic state and certainty for a specific memory.
        
        Args:
            memory_id: ID of the memory to check
            
        Returns:
            Tuple of (epistemic_state, certainty)
        """
        state = self.memory_states.get(memory_id, EpistemicState.UNKNOWN)
        certainty = self.memory_certainty.get(memory_id, 0.0)
        return state, certainty
    
    def get_active_triggers(self) -> Set[ReflectionTrigger]:
        """
        Get currently active reflection triggers.
        
        Returns:
            Set of active triggers
        """
        return self.active_triggers.copy()
    
    def get_metrics_history(self) -> List[Tuple[datetime, Dict[str, Any]]]:
        """
        Get history of epistemic metrics.
        
        Returns:
            List of (timestamp, metrics_dict) pairs
        """
        return [(ts, metrics._asdict()) for ts, metrics in self.metrics_history]
    
    def get_state_history(self) -> List[Tuple[datetime, str]]:
        """
        Get history of epistemic states.
        
        Returns:
            List of (timestamp, state_name) pairs
        """
        return [(ts, state.value) for ts, state in self.state_history]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the epistemic status system.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
    
    def recalculate_uncertainty_propagation(self) -> Dict[str, float]:
        """
        Recalculate uncertainty propagation across the memory graph.
        
        This propagates uncertainty through connected memories in the schema
        graph, causing uncertainty in one node to affect related nodes.
        
        Returns:
            Dictionary mapping memory IDs to updated certainty scores
        """
        # Skip if no schema graph is available
        if not self.schema_graph:
            logger.warning("Cannot propagate uncertainty without a schema graph")
            return self.memory_certainty.copy()
        
        # Get current certainty values
        certainty = self.memory_certainty.copy()
        
        # Get graph connections
        connections = {}
        if hasattr(self.schema_graph, 'get_connections'):
            connections = self.schema_graph.get_connections()
        else:
            logger.warning("Schema graph does not provide connection information")
            return certainty
        
        # Propagate uncertainty through the graph
        # Memories connected to uncertain memories become slightly more uncertain
        updated_certainty = certainty.copy()
        
        for memory_id, connected_ids in connections.items():
            # Skip if memory not in certainty dict
            if memory_id not in certainty:
                continue
                
            # Get connected memories' certainty
            connected_certainty = [
                certainty.get(connected_id, 0.5)
                for connected_id in connected_ids
                if connected_id in certainty
            ]
            
            if not connected_certainty:
                continue
                
            # Calculate influence (weighted average)
            # A memory's certainty is influenced by connected memories
            current_certainty = certainty[memory_id]
            influenced_certainty = sum(connected_certainty) / len(connected_certainty)
            
            # Update with blended certainty (70% current, 30% influence)
            updated_certainty[memory_id] = (
                current_certainty * 0.7 + influenced_certainty * 0.3
            )
        
        # Update memory certainty
        self.memory_certainty = updated_certainty
        
        return updated_certainty
