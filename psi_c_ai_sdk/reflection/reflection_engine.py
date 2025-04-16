"""
Reflection Engine for ΨC-AI SDK

This module implements the reflection system which analyzes epistemic states,
identifies knowledge gaps and contradictions, and triggers reflection cycles
to improve coherence and stability of the AI system's knowledge.
"""

import logging
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import random

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore, MemoryType
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer
from psi_c_ai_sdk.epistemic.epistemic_status import (
    EpistemicStatus, EpistemicState, ReflectionTrigger, EpistemicRecord
)
from psi_c_ai_sdk.beliefs.revision import BeliefRevisionSystem
from psi_c_ai_sdk.schema.toolkit import SchemaToolkit
from psi_c_ai_sdk.memory.legacy import (
    LegacyManager, 
    should_create_legacy, 
    generate_epitaph
)

# Configure logging
logger = logging.getLogger(__name__)


class ReflectionState(Enum):
    """State of the reflection process."""
    
    IDLE = auto()             # No active reflection
    TRIGGERED = auto()        # Reflection triggered but not started
    ANALYZING = auto()        # Analyzing knowledge and contradictions
    RESOLVING = auto()        # Resolving contradictions
    CONSOLIDATING = auto()    # Consolidating knowledge
    INTEGRATING = auto()      # Integrating new insights
    UPDATING = auto()         # Updating schema and beliefs
    COMPLETE = auto()         # Reflection cycle complete


@dataclass
class ReflectionOutcome:
    """Outcome of a reflection cycle."""
    
    id: str                               # Unique identifier
    trigger: ReflectionTrigger            # What triggered the reflection
    start_time: datetime                  # When reflection started
    end_time: Optional[datetime] = None   # When reflection completed
    duration: Optional[float] = None      # Duration in seconds
    success: bool = False                 # Whether reflection was successful
    contradictions_found: int = 0         # Number of contradictions found
    contradictions_resolved: int = 0      # Number of contradictions resolved
    knowledge_consolidated: int = 0       # Number of knowledge elements consolidated
    confidence_change: float = 0.0        # Change in average confidence
    stability_change: float = 0.0         # Change in average stability
    coherence_change: float = 0.0         # Change in average coherence
    created_memories: List[str] = field(default_factory=list)  # IDs of memories created
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "trigger": self.trigger.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "success": self.success,
            "contradictions_found": self.contradictions_found,
            "contradictions_resolved": self.contradictions_resolved,
            "knowledge_consolidated": self.knowledge_consolidated,
            "confidence_change": self.confidence_change,
            "stability_change": self.stability_change,
            "coherence_change": self.coherence_change,
            "created_memories": self.created_memories,
            "metadata": self.metadata
        }


@dataclass
class ReflectionCreditRecord:
    """Record of credit assigned to reflective processes."""
    
    id: str                               # Unique identifier
    memory_id: str                        # Memory ID getting credit
    reflection_id: str                    # Reflection ID giving credit
    credit_amount: float                  # Amount of credit (0-1)
    reason: str                           # Reason for credit
    timestamp: datetime = field(default_factory=datetime.now)  # When credit was assigned
    

class ReflectionCreditSystem:
    """
    System for tracking and assigning credit to reflective processes.
    
    This enables the AI system to attribute improvements in coherence,
    stability, and other metrics to specific reflection events.
    """
    
    def __init__(self, max_records: int = 1000):
        """
        Initialize the reflection credit system.
        
        Args:
            max_records: Maximum number of credit records to maintain
        """
        self.records: List[ReflectionCreditRecord] = []
        self.max_records = max_records
        
        # Track credit by memory
        self.memory_credit: Dict[str, float] = {}
        
        # Track credit by reflection
        self.reflection_credit: Dict[str, float] = {}
    
    def add_credit(
        self,
        memory_id: str,
        reflection_id: str,
        credit_amount: float,
        reason: str
    ) -> str:
        """
        Add credit for a memory item to a reflection event.
        
        Args:
            memory_id: ID of memory receiving credit
            reflection_id: ID of reflection event giving credit
            credit_amount: Amount of credit (0-1)
            reason: Reason for credit
            
        Returns:
            ID of the created credit record
        """
        # Create record
        record_id = str(uuid.uuid4())
        record = ReflectionCreditRecord(
            id=record_id,
            memory_id=memory_id,
            reflection_id=reflection_id,
            credit_amount=credit_amount,
            reason=reason
        )
        
        # Add to records
        self.records.append(record)
        
        # Update credit trackers
        if memory_id not in self.memory_credit:
            self.memory_credit[memory_id] = 0
        self.memory_credit[memory_id] += credit_amount
        
        if reflection_id not in self.reflection_credit:
            self.reflection_credit[reflection_id] = 0
        self.reflection_credit[reflection_id] += credit_amount
        
        # Prune if needed
        if len(self.records) > self.max_records:
            self._prune_records()
            
        return record_id
    
    def get_memory_credit(self, memory_id: str) -> float:
        """
        Get total credit for a memory.
        
        Args:
            memory_id: ID of memory to get credit for
            
        Returns:
            Total credit amount
        """
        return self.memory_credit.get(memory_id, 0.0)
    
    def get_reflection_credit(self, reflection_id: str) -> float:
        """
        Get total credit for a reflection event.
        
        Args:
            reflection_id: ID of reflection to get credit for
            
        Returns:
            Total credit amount
        """
        return self.reflection_credit.get(reflection_id, 0.0)
    
    def get_records_for_memory(self, memory_id: str) -> List[ReflectionCreditRecord]:
        """
        Get all credit records for a memory.
        
        Args:
            memory_id: ID of memory to get records for
            
        Returns:
            List of credit records
        """
        return [record for record in self.records if record.memory_id == memory_id]
    
    def get_records_for_reflection(self, reflection_id: str) -> List[ReflectionCreditRecord]:
        """
        Get all credit records for a reflection event.
        
        Args:
            reflection_id: ID of reflection to get records for
            
        Returns:
            List of credit records
        """
        return [record for record in self.records if record.reflection_id == reflection_id]
    
    def _prune_records(self) -> None:
        """
        Prune oldest records to maintain size.
        """
        to_remove = len(self.records) - self.max_records
        if to_remove <= 0:
            return
            
        # Sort by timestamp (oldest first)
        self.records.sort(key=lambda x: x.timestamp)
        
        # Remove oldest records
        removed_records = self.records[:to_remove]
        self.records = self.records[to_remove:]
        
        # Update credit trackers
        for record in removed_records:
            self.memory_credit[record.memory_id] -= record.credit_amount
            if self.memory_credit[record.memory_id] <= 0:
                del self.memory_credit[record.memory_id]
                
            self.reflection_credit[record.reflection_id] -= record.credit_amount
            if self.reflection_credit[record.reflection_id] <= 0:
                del self.reflection_credit[record.reflection_id]


def calculate_cognitive_debt(
    reflection_rate: float,
    schema_change_rate: float,
    avg_coherence: float
) -> float:
    """
    Calculate cognitive debt based on reflection, schema changes, and coherence.
    
    A system accumulates cognitive debt when it changes rapidly without 
    sufficient reflection and coherence.
    
    Args:
        reflection_rate: Rate of reflection events
        schema_change_rate: Rate of schema changes
        avg_coherence: Average coherence score
        
    Returns:
        Cognitive debt score (higher is worse)
    """
    # Base formula: D_t = R_t - λ · (S_t + C_t)
    # From UnderlyingMath.md #24
    
    # Constant parameters
    lambda_factor = 0.5
    
    # Calculate debt
    debt = reflection_rate - lambda_factor * (schema_change_rate + avg_coherence)
    
    # Bound to 0-1 range
    return max(0.0, min(1.0, debt))


class ReflectionEngine:
    """
    Core engine for managing reflection cycles in the ΨC-AI system.
    
    The ReflectionEngine monitors epistemic states, triggers reflection cycles
    when needed, and manages the reflection process to improve coherence,
    resolve contradictions, and integrate new insights.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        epistemic_status: EpistemicStatus,
        belief_system: Optional[BeliefRevisionSystem] = None,
        schema_toolkit: Optional[SchemaToolkit] = None,
        coherence_scorer: Optional[CoherenceScorer] = None,
        reflection_cooldown: timedelta = timedelta(minutes=5),
        max_reflection_duration: timedelta = timedelta(minutes=30),
        max_outcomes: int = 100
    ):
        """
        Initialize the reflection engine.
        
        Args:
            memory_store: Memory store to work with
            epistemic_status: Epistemic status system to monitor
            belief_system: Optional belief revision system
            schema_toolkit: Optional schema toolkit
            coherence_scorer: Optional coherence scorer
            reflection_cooldown: Minimum time between reflection cycles
            max_reflection_duration: Maximum duration for a reflection cycle
            max_outcomes: Maximum number of reflection outcomes to track
        """
        self.memory_store = memory_store
        self.epistemic_status = epistemic_status
        self.belief_system = belief_system
        self.schema_toolkit = schema_toolkit
        self.coherence_scorer = coherence_scorer
        
        self.reflection_cooldown = reflection_cooldown
        self.max_reflection_duration = max_reflection_duration
        self.max_outcomes = max_outcomes
        
        # Track reflection state
        self.current_state = ReflectionState.IDLE
        self.current_reflection_id: Optional[str] = None
        self.active_reflection_start: Optional[datetime] = None
        
        # Track outcomes
        self.outcomes: List[ReflectionOutcome] = []
        
        # Credit system
        self.credit_system = ReflectionCreditSystem()
        
        # Performance metrics
        self.total_reflections = 0
        self.successful_reflections = 0
        self.failed_reflections = 0
        self.last_reflection_time: Optional[datetime] = None
        self.total_computation_time = 0.0
    
    def check_reflection_needed(self) -> Tuple[bool, Optional[ReflectionTrigger]]:
        """
        Check if reflection is needed based on current epistemic state.
        
        Returns:
            Tuple of (needed, trigger)
        """
        # Don't check if already reflecting
        if self.current_state != ReflectionState.IDLE:
            return False, None
            
        # Check cooldown period
        if self.last_reflection_time and datetime.now() - self.last_reflection_time < self.reflection_cooldown:
            return False, None
            
        # Check epistemic status
        needed, trigger = self.epistemic_status.check_reflection_needed()
        
        return needed, trigger
    
    def trigger_reflection(self, trigger: ReflectionTrigger) -> Optional[str]:
        """
        Trigger a reflection cycle.
        
        Args:
            trigger: What triggered the reflection
            
        Returns:
            ID of the reflection if started, None otherwise
        """
        # Don't start reflection if already reflecting
        if self.current_state != ReflectionState.IDLE:
            logger.warning("Cannot trigger reflection: already reflecting")
            return None
            
        # Create reflection ID
        reflection_id = f"reflection_{int(time.time())}_{self.total_reflections}"
        
        # Create outcome record
        outcome = ReflectionOutcome(
            id=reflection_id,
            trigger=trigger,
            start_time=datetime.now()
        )
        
        # Update state
        self.current_state = ReflectionState.TRIGGERED
        self.current_reflection_id = reflection_id
        self.active_reflection_start = datetime.now()
        
        # Add to outcomes
        self.outcomes.append(outcome)
        
        # Update metrics
        self.total_reflections += 1
        self.last_reflection_time = datetime.now()
        
        logger.info(f"Reflection triggered: {trigger.name} (ID: {reflection_id})")
        
        return reflection_id
    
    def manually_trigger_reflection(self) -> Optional[str]:
        """
        Manually trigger a reflection cycle.
        
        Returns:
            ID of the reflection if started, None otherwise
        """
        return self.trigger_reflection(ReflectionTrigger.MANUAL)
    
    def run_reflection_cycle(self) -> Optional[ReflectionOutcome]:
        """
        Run a complete reflection cycle.
        
        This is the main method that executes a full reflection cycle,
        analyzing knowledge, resolving contradictions, and consolidating insights.
        
        Returns:
            Reflection outcome if completed, None otherwise
        """
        # Must be in TRIGGERED state to start
        if self.current_state != ReflectionState.TRIGGERED:
            logger.warning(f"Cannot run reflection cycle: wrong state ({self.current_state})")
            return None
            
        # Get current reflection outcome
        if not self.current_reflection_id or len(self.outcomes) == 0:
            logger.error("No active reflection to run")
            return None
            
        outcome = next((o for o in self.outcomes if o.id == self.current_reflection_id), None)
        if not outcome:
            logger.error(f"Could not find outcome for reflection {self.current_reflection_id}")
            return None
            
        # Start timer
        cycle_start = time.time()
        
        try:
            # 1. Analyze knowledge and find contradictions
            self.current_state = ReflectionState.ANALYZING
            logger.info("Reflection phase: ANALYZING")
            
            # Get initial metrics
            initial_stats = self.epistemic_status.get_stats()
            initial_avg_confidence = initial_stats["avg_confidence"]
            initial_avg_stability = initial_stats["avg_stability"]
            
            # Find contradictions
            contradictions = self.epistemic_status.find_contradictions()
            outcome.contradictions_found = len(contradictions)
            
            # 2. Resolve contradictions if belief system available
            if self.belief_system and contradictions:
                self.current_state = ReflectionState.RESOLVING
                logger.info(f"Reflection phase: RESOLVING {len(contradictions)} contradictions")
                
                # Process each contradiction
                resolved = 0
                for record1, record2 in contradictions:
                    # Collect memory IDs from both records
                    memory_ids1 = record1.memory_ids
                    memory_ids2 = record2.memory_ids
                    
                    # Get actual memories
                    memories1 = [self.memory_store.get_memory_by_id(mid) for mid in memory_ids1 if mid]
                    memories1 = [m for m in memories1 if m]  # Filter None
                    
                    memories2 = [self.memory_store.get_memory_by_id(mid) for mid in memory_ids2 if mid]
                    memories2 = [m for m in memories2 if m]  # Filter None
                    
                    if not memories1 or not memories2:
                        continue
                        
                    # Resolve contradiction between memory sets
                    resolution_result = self.belief_system.arbitrate_contradiction(memories1, memories2)
                    
                    if resolution_result and resolution_result.get("success", False):
                        resolved += 1
                        
                        # Add reflection memory about the resolution
                        resolution_memory = Memory(
                            uuid=uuid.uuid4(),
                            content=f"Resolved contradiction between '{record1.topic}' and '{record2.topic}': {resolution_result.get('explanation', '')}",
                            memory_type=MemoryType.REFLECTION,
                            importance=0.7,
                            metadata={
                                "reflection_id": outcome.id,
                                "contradiction_resolution": True,
                                "record1_id": record1.id,
                                "record2_id": record2.id,
                                "decision": resolution_result.get("decision", "")
                            }
                        )
                        
                        memory_id = str(self.memory_store.add_memory(resolution_memory))
                        outcome.created_memories.append(memory_id)
                        
                outcome.contradictions_resolved = resolved
            
            # 3. Consolidate knowledge
            self.current_state = ReflectionState.CONSOLIDATING
            logger.info("Reflection phase: CONSOLIDATING")
            
            # Get uncertain knowledge
            uncertain_records = self.epistemic_status.get_uncertain_knowledge()
            
            # Create consolidated memories for uncertain topics
            topic_groups: Dict[str, List[EpistemicRecord]] = {}
            for record in uncertain_records:
                if record.topic not in topic_groups:
                    topic_groups[record.topic] = []
                topic_groups[record.topic].append(record)
            
            # Process each topic group
            consolidated = 0
            for topic, records in topic_groups.items():
                if len(records) < 2:
                    continue
                    
                # Collect all memory IDs
                all_memory_ids = []
                for record in records:
                    all_memory_ids.extend(record.memory_ids)
                    
                # Get actual memories
                all_memories = [self.memory_store.get_memory_by_id(mid) for mid in all_memory_ids if mid]
                all_memories = [m for m in all_memories if m]  # Filter None
                
                if not all_memories:
                    continue
                    
                # Create consolidated memory
                summary = f"Consolidated knowledge about '{topic}' from {len(all_memories)} memories."
                if len(all_memories) > 0:
                    # Could add more sophisticated summarization here
                    summary += " Key points: "
                    for i, memory in enumerate(all_memories[:3]):
                        if memory and memory.content:
                            summary += f"({i+1}) {memory.content[:50]}... "
                
                consolidated_memory = Memory(
                    uuid=uuid.uuid4(),
                    content=summary,
                    memory_type=MemoryType.REFLECTION,
                    importance=0.8,
                    metadata={
                        "reflection_id": outcome.id,
                        "knowledge_consolidation": True,
                        "topic": topic,
                        "record_ids": [r.id for r in records],
                        "memory_count": len(all_memories)
                    }
                )
                
                memory_id = str(self.memory_store.add_memory(consolidated_memory))
                outcome.created_memories.append(memory_id)
                consolidated += 1
                
                # Add to epistemic status with higher confidence
                new_confidence = 0.7  # Higher confidence for consolidated knowledge
                self.epistemic_status.add_knowledge(
                    topic=topic,
                    confidence=new_confidence,
                    memory_id=memory_id,
                    metadata={"consolidated": True, "reflection_id": outcome.id}
                )
            
            outcome.knowledge_consolidated = consolidated
            
            # 4. Update schema if available
            if self.schema_toolkit:
                self.current_state = ReflectionState.UPDATING
                logger.info("Reflection phase: UPDATING schema")
                
                for memory_id in outcome.created_memories:
                    memory = self.memory_store.get_memory_by_id(memory_id)
                    if memory:
                        # Add to schema
                        self.schema_toolkit.add_memory(memory, check_mutation=True)
            
            # 5. Finalize and measure changes
            self.current_state = ReflectionState.COMPLETE
            
            # Get final metrics
            final_stats = self.epistemic_status.get_stats()
            final_avg_confidence = final_stats["avg_confidence"]
            final_avg_stability = final_stats["avg_stability"]
            
            # Calculate changes
            outcome.confidence_change = final_avg_confidence - initial_avg_confidence
            outcome.stability_change = final_avg_stability - initial_avg_stability
            
            # Calculate coherence change if scorer available
            if self.coherence_scorer:
                # This is simplified - real implementation would calculate
                # coherence over the relevant memory subsets
                outcome.coherence_change = 0.05  # Placeholder
            
            # Complete outcome
            outcome.end_time = datetime.now()
            outcome.duration = time.time() - cycle_start
            outcome.success = True
            
            # Update metrics
            self.successful_reflections += 1
            self.total_computation_time += outcome.duration
            
            logger.info(f"Reflection complete: {len(outcome.created_memories)} new memories, " +
                      f"{outcome.contradictions_resolved}/{outcome.contradictions_found} contradictions resolved")
            
            # After reflection is completed, check if we should create a legacy
            # This is triggered when entropy is high or coherence is low
            try:
                # Calculate final coherence and entropy after reflection
                final_coherence = self.calculate_global_coherence()
                # Simple entropy approximation
                final_entropy = 1.0 - (final_coherence * 0.8)
                
                # Check if entropy is critically high or coherence is critically low
                if final_entropy > 0.7 or final_coherence < 0.3:
                    self.logger.info(f"Post-reflection critical values detected: entropy={final_entropy:.2f}, coherence={final_coherence:.2f}")
                    
                    # Create legacy if needed
                    legacy_created = self.check_legacy_creation(
                        entropy=final_entropy,
                        coherence=final_coherence
                    )
                    
                    if legacy_created:
                        self.logger.info("Legacy block created as part of reflection cycle")
                        
                        # Update reflection outcome with legacy information
                        outcome.metadata["legacy_created"] = True
                        outcome.metadata["entropy"] = final_entropy
                        outcome.metadata["coherence"] = final_coherence
                        
                        self.logger.info("Agent has reached entropy threshold and created a legacy")
            except Exception as e:
                self.logger.error(f"Error during legacy check: {e}")
                # Don't let legacy creation failure affect the reflection process
            
            return outcome
            
        except Exception as e:
            logger.error(f"Error during reflection cycle: {str(e)}", exc_info=True)
            
            # Complete outcome as failed
            outcome.end_time = datetime.now()
            outcome.duration = time.time() - cycle_start
            outcome.success = False
            
            # Update metrics
            self.failed_reflections += 1
            self.total_computation_time += outcome.duration
            
            return outcome
        finally:
            # Reset state
            self.current_state = ReflectionState.IDLE
            self.current_reflection_id = None
            self.active_reflection_start = None
    
    def get_outcome(self, reflection_id: str) -> Optional[ReflectionOutcome]:
        """
        Get a reflection outcome by ID.
        
        Args:
            reflection_id: ID of the reflection to get
            
        Returns:
            ReflectionOutcome if found, None otherwise
        """
        for outcome in self.outcomes:
            if outcome.id == reflection_id:
                return outcome
        return None
    
    def get_recent_outcomes(self, limit: int = 10) -> List[ReflectionOutcome]:
        """
        Get recent reflection outcomes.
        
        Args:
            limit: Maximum number of outcomes to return
            
        Returns:
            List of recent ReflectionOutcome objects
        """
        # Sort by start time (newest first)
        sorted_outcomes = sorted(self.outcomes, key=lambda o: o.start_time, reverse=True)
        return sorted_outcomes[:limit]
    
    def assign_credit(
        self,
        memory_id: str,
        reflection_id: str,
        credit_amount: float,
        reason: str
    ) -> str:
        """
        Assign credit to a memory for a reflection event.
        
        Args:
            memory_id: ID of memory receiving credit
            reflection_id: ID of reflection event
            credit_amount: Amount of credit (0-1)
            reason: Reason for credit
            
        Returns:
            ID of the created credit record
        """
        return self.credit_system.add_credit(
            memory_id=memory_id,
            reflection_id=reflection_id,
            credit_amount=credit_amount,
            reason=reason
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the reflection engine.
        
        Returns:
            Dictionary with reflection statistics
        """
        # Calculate success rate
        if self.total_reflections > 0:
            success_rate = self.successful_reflections / self.total_reflections
        else:
            success_rate = 0
            
        # Calculate average duration
        if self.successful_reflections > 0:
            avg_duration = self.total_computation_time / self.successful_reflections
        else:
            avg_duration = 0
            
        # Get last reflection time
        last_reflection_ago = None
        if self.last_reflection_time:
            last_reflection_ago = (datetime.now() - self.last_reflection_time).total_seconds()
            
        return {
            "total_reflections": self.total_reflections,
            "successful_reflections": self.successful_reflections,
            "failed_reflections": self.failed_reflections,
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "total_computation_time": self.total_computation_time,
            "current_state": self.current_state.name,
            "last_reflection_time": self.last_reflection_time.isoformat() if self.last_reflection_time else None,
            "last_reflection_ago": last_reflection_ago,
            "outcome_count": len(self.outcomes),
            "current_reflection_id": self.current_reflection_id
        }

    def check_legacy_creation(
        self,
        entropy: float = None,
        coherence: float = None,
        entropy_threshold: float = 0.8,
        coherence_threshold: float = 0.3,
        legacy_manager = None
    ) -> bool:
        """
        Check if the agent should create a legacy block based on entropy and coherence levels.
        
        Args:
            entropy: Current entropy level (will be calculated if None)
            coherence: Current coherence level (will be calculated if None)
            entropy_threshold: Entropy threshold for legacy creation
            coherence_threshold: Coherence threshold for legacy creation
            legacy_manager: Optional LegacyManager instance (will be created if None)
            
        Returns:
            True if a legacy block was created, False otherwise
        """
        # Calculate entropy and coherence if not provided
        if entropy is None:
            # Simple entropy approximation for demo purposes
            # In a real implementation, this would use a more sophisticated entropy calculation
            entropy = 1.0 - (self.calculate_global_coherence() * 0.8)
        
        if coherence is None:
            coherence = self.calculate_global_coherence()
            
        self.logger.info(f"Checking legacy creation conditions: entropy={entropy:.2f}, coherence={coherence:.2f}")
        
        # Check if agent should create a legacy
        if should_create_legacy(
            entropy=entropy,
            coherence=coherence,
            entropy_threshold=entropy_threshold,
            coherence_threshold=coherence_threshold
        ):
            self.logger.info("Legacy creation conditions met, creating legacy block")
            
            # Initialize legacy manager if not provided
            if legacy_manager is None:
                import os
                legacy_dir = os.path.join(os.getcwd(), "legacy")
                os.makedirs(legacy_dir, exist_ok=True)
                legacy_manager = LegacyManager(storage_path=legacy_dir)
            
            # Get all memories from the memory store
            memories = [m.__dict__ for m in self.memory_store.get_all_memories()]
            
            # Generate epitaph
            high_entropy_regions = self._identify_high_entropy_regions()
            contradiction_nodes = self._get_contradiction_nodes()
            unresolved_reflections = self._get_unresolved_reflections()
            
            epitaph = generate_epitaph(
                self.memory_store,
                high_entropy_regions=high_entropy_regions,
                contradiction_nodes=contradiction_nodes,
                unresolved_reflections=unresolved_reflections
            )
            
            # Create coherence scores, access counts, and recency data
            coherence_scores = self._calculate_memory_coherence_scores()
            access_counts = self._get_memory_access_counts()
            recency_scores = self._calculate_memory_recency_scores()
            
            # Get schema fingerprint if available
            schema_fingerprint = None
            if hasattr(self, 'schema_graph') and self.schema_graph is not None:
                if hasattr(self.schema_graph, 'get_fingerprint'):
                    schema_fingerprint = self.schema_graph.get_fingerprint()
            
            # Create agent ID and name
            agent_id = str(uuid.uuid4())[:8]
            agent_name = getattr(self, 'name', f"ΨC-{agent_id}")
            
            # Create legacy block
            legacy_block = legacy_manager.create_legacy_block(
                agent_id=agent_id,
                agent_name=agent_name,
                memories=memories,
                coherence_scores=coherence_scores,
                access_counts=access_counts,
                recency_scores=recency_scores,
                selector_type="emergent_value",
                epitaph=epitaph,
                schema_fingerprint=schema_fingerprint,
                selection_params={
                    "alpha": 0.4,  # Coherence weight
                    "beta": 0.3,   # Access frequency weight
                    "gamma": 0.3,  # Recency weight
                    "value_threshold": 0.6,
                    "top_k": 10,
                    "message": f"These are the core memories of {agent_name} at termination."
                }
            )
            
            # Save the legacy block
            legacy_file = legacy_manager.save_legacy_block(
                legacy_block, 
                filename=f"{agent_name}_legacy.json"
            )
            
            logger.info(f"Created legacy block with {len(legacy_block.core_memories)} memories")
            logger.info(f"Legacy block saved to {legacy_file}")
            
            return True
        
        return False
    
    def _identify_high_entropy_regions(self) -> List[str]:
        """Identify schema regions with high entropy."""
        # This would be implemented based on schema analysis
        # For demonstration, we return sample data
        return ["knowledge integration", "ethical reasoning", "temporal coherence"]
    
    def _get_contradiction_nodes(self) -> List[str]:
        """Get list of nodes with unresolved contradictions."""
        # This would be implemented based on contradiction detection
        # For demonstration, we return sample data
        contradictions = getattr(self, '_recent_contradictions', [])
        return [f"node_{i}" for i in range(len(contradictions))]
    
    def _get_unresolved_reflections(self) -> List[Dict]:
        """Get list of unresolved reflection cycles."""
        # This would be implemented based on reflection history
        # For demonstration, we return sample data
        return [{"id": f"refl_{i}"} for i in range(2)]
    
    def _calculate_memory_coherence_scores(self) -> Dict[str, float]:
        """Calculate coherence scores for each memory."""
        # This would use the coherence scorer to evaluate each memory
        # For demonstration, we return sample data
        memories = self.memory_store.get_all_memories()
        return {m.id: random.uniform(0.5, 0.9) for m in memories}
    
    def _get_memory_access_counts(self) -> Dict[str, int]:
        """Get access counts for each memory."""
        # This would be retrieved from memory access tracking
        # For demonstration, we return sample data
        memories = self.memory_store.get_all_memories()
        return {m.id: random.randint(1, 20) for m in memories}
    
    def _calculate_memory_recency_scores(self) -> Dict[str, float]:
        """Calculate recency scores for each memory based on timestamps."""
        # This would analyze timestamps to determine recency
        # For demonstration, we return sample data
        memories = self.memory_store.get_all_memories()
        return {m.id: random.uniform(0.1, 1.0) for m in memories}


class ReflectionScheduler:
    """
    Scheduler for triggering reflection cycles based on system state and timing.
    
    The scheduler monitors the epistemic status and other metrics to determine
    when to trigger reflection cycles, and manages the scheduling of these cycles
    to optimize system performance.
    """
    
    def __init__(
        self,
        reflection_engine: ReflectionEngine,
        epistemic_status: EpistemicStatus,
        check_interval: timedelta = timedelta(minutes=1),
        min_updates_before_check: int = 10,
        force_reflection_after: timedelta = timedelta(hours=1)
    ):
        """
        Initialize the reflection scheduler.
        
        Args:
            reflection_engine: Reflection engine to schedule
            epistemic_status: Epistemic status to monitor
            check_interval: How often to check if reflection is needed
            min_updates_before_check: Minimum updates before checking
            force_reflection_after: Force reflection after this time
        """
        self.reflection_engine = reflection_engine
        self.epistemic_status = epistemic_status
        self.check_interval = check_interval
        self.min_updates_before_check = min_updates_before_check
        self.force_reflection_after = force_reflection_after
        
        # Tracking
        self.last_check_time = datetime.now()
        self.last_reflection_time = None
    
    def check_and_schedule(self) -> Optional[str]:
        """
        Check if reflection is needed and schedule if appropriate.
        
        Returns:
            ID of scheduled reflection if triggered, None otherwise
        """
        now = datetime.now()
        
        # Check conditions for scheduling reflection
        should_check = False
        
        # Check interval elapsed
        if now - self.last_check_time >= self.check_interval:
            should_check = True
        
        # Minimum updates reached
        if self.epistemic_status.updates_since_reflection >= self.min_updates_before_check:
            should_check = True
        
        # Force reflection after long time
        if self.last_reflection_time and now - self.last_reflection_time >= self.force_reflection_after:
            logger.info("Forcing reflection due to time since last reflection")
            reflection_id = self.reflection_engine.manually_trigger_reflection()
            self.last_reflection_time = now
            return reflection_id
        
        # Check if needed
        if should_check:
            self.last_check_time = now
            
            needed, trigger = self.reflection_engine.check_reflection_needed()
            if needed and trigger:
                logger.info(f"Scheduling reflection due to trigger: {trigger.name}")
                reflection_id = self.reflection_engine.trigger_reflection(trigger)
                self.last_reflection_time = now
                return reflection_id
        
        return None 