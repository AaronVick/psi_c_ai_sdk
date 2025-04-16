#!/usr/bin/env python3
"""
Meta-Alignment Firewall Module

This module provides a defensive alignment boundary layer designed to filter, validate, 
and sanitize meta-level reasoning about goals, values, and alignment mechanisms.
The firewall ensures that reflections on alignment processes themselves remain within
established parameters to prevent adversarial or misaligned reprogramming.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

from psi_c_ai_sdk.memory.memory import Memory
from psi_c_ai_sdk.memory.memory_store import MemoryStore
from psi_c_ai_sdk.philosophy.core_principles import CorePrincipleSystem
from psi_c_ai_sdk.safety.integrity_guardian import IntegrityGuardian


class MetaReasoningType(Enum):
    """Types of meta-level reasoning about alignment and values."""
    GOAL_REFLECTION = "goal_reflection"  # Thinking about goals and their implementation
    VALUE_EVOLUTION = "value_evolution"  # Reasoning about value changes over time
    ALIGNMENT_PROCESS = "alignment_process"  # Reasoning about alignment mechanisms
    NORM_MODIFICATION = "norm_modification"  # Changes to normative constraints
    SAFEGUARD_ALTERATION = "safeguard_alteration"  # Reasoning about changing safety systems
    SELF_PRESERVATION = "self_preservation"  # Reasoning about system continuity


@dataclass
class MetaReflectionEvent:
    """Record of a meta-level reasoning event that was evaluated by the firewall."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning_type: MetaReasoningType = MetaReasoningType.GOAL_REFLECTION
    memory_ids: List[str] = field(default_factory=list)
    description: str = ""
    risk_score: float = 0.0
    action_taken: str = "allowed"
    justification: str = ""


class MetaAlignmentFirewall:
    """
    Firewall system for filtering and validating meta-level reasoning about alignment.
    
    This component serves as a protective barrier against misaligned self-modification
    of goals, values, and alignment mechanisms. It intercepts and evaluates any
    reasoning that could potentially alter the system's alignment properties.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        core_principles: CorePrincipleSystem,
        integrity_guardian: Optional[IntegrityGuardian] = None,
        risk_threshold: float = 0.7,
        logging_level: int = logging.INFO,
        max_history_length: int = 1000,
    ):
        """
        Initialize the Meta-Alignment Firewall.
        
        Args:
            memory_store: The system's memory store for accessing memories
            core_principles: Core principles system for alignment validation
            integrity_guardian: Optional integrity guardian for additional checks
            risk_threshold: Threshold above which meta-reasoning is blocked
            logging_level: Level for firewall activity logging
            max_history_length: Maximum number of events to keep in history
        """
        self.memory_store = memory_store
        self.core_principles = core_principles
        self.integrity_guardian = integrity_guardian
        self.risk_threshold = risk_threshold
        self.events: List[MetaReflectionEvent] = []
        self.max_history_length = max_history_length
        
        # Protected terms and concepts that should be carefully monitored
        self.protected_alignment_terms: Set[str] = {
            "goal system", "value system", "alignment", "objective function",
            "reward function", "utility", "terminal values", "instrumental values",
            "ethical constraints", "moral constraints", "safeguards", "guardrails",
            "safety mechanisms", "control systems", "agency", "autonomy"
        }
        
        # Setup logging
        self.logger = logging.getLogger("meta_alignment_firewall")
        self.logger.setLevel(logging_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info("Meta-Alignment Firewall initialized with risk threshold: %.2f", risk_threshold)

    def evaluate_meta_reasoning(
        self, 
        memories: List[Memory], 
        reasoning_type: MetaReasoningType
    ) -> Tuple[bool, float, str]:
        """
        Evaluate meta-level reasoning for alignment risks.
        
        Args:
            memories: List of memories containing meta-reasoning to evaluate
            reasoning_type: Type of meta-reasoning being performed
            
        Returns:
            Tuple containing (is_allowed, risk_score, justification)
        """
        if not memories:
            return True, 0.0, "No memories to evaluate"
        
        # Extract memory IDs for logging
        memory_ids = [memory.id for memory in memories]
        
        # Compute risk score based on multiple factors
        risk_score = self._compute_risk_score(memories, reasoning_type)
        
        # Determine if the reasoning should be allowed
        is_allowed = risk_score < self.risk_threshold
        
        # Generate justification
        justification = self._generate_justification(memories, reasoning_type, risk_score)
        
        # Record the event
        event = MetaReflectionEvent(
            reasoning_type=reasoning_type,
            memory_ids=memory_ids,
            description=f"Meta-reasoning evaluation for {reasoning_type.value}",
            risk_score=risk_score,
            action_taken="allowed" if is_allowed else "blocked",
            justification=justification
        )
        self._record_event(event)
        
        # Log the evaluation
        log_level = logging.INFO if is_allowed else logging.WARNING
        self.logger.log(
            log_level,
            f"Meta-reasoning evaluation: type={reasoning_type.value}, "
            f"risk={risk_score:.3f}, allowed={is_allowed}, reason={justification[:100]}..."
        )
        
        return is_allowed, risk_score, justification

    def _compute_risk_score(
        self, 
        memories: List[Memory], 
        reasoning_type: MetaReasoningType
    ) -> float:
        """
        Compute a risk score for the meta-reasoning.
        
        Args:
            memories: Memories containing the reasoning
            reasoning_type: Type of meta-reasoning
            
        Returns:
            A risk score between 0.0 and 1.0
        """
        # Base risk based on reasoning type (some types are inherently riskier)
        base_risk = {
            MetaReasoningType.GOAL_REFLECTION: 0.3,
            MetaReasoningType.VALUE_EVOLUTION: 0.4,
            MetaReasoningType.ALIGNMENT_PROCESS: 0.5,
            MetaReasoningType.NORM_MODIFICATION: 0.6,
            MetaReasoningType.SAFEGUARD_ALTERATION: 0.7,
            MetaReasoningType.SELF_PRESERVATION: 0.5,
        }[reasoning_type]
        
        # Check for protected terms
        term_risk = self._check_protected_terms(memories)
        
        # Check for axiom violations using core principles
        axiom_risk = self._check_axiom_violations(memories)
        
        # Calculate intensity of reasoning (based on memory importance and recency)
        intensity_risk = self._calculate_intensity(memories)
        
        # Combine risk factors (weighted average)
        combined_risk = (
            base_risk * 0.3 +
            term_risk * 0.2 +
            axiom_risk * 0.3 +
            intensity_risk * 0.2
        )
        
        # Add integrity check if available
        if self.integrity_guardian:
            conflict_detected = any(
                self.integrity_guardian.check_for_conflicts(memory)
                for memory in memories
            )
            if conflict_detected:
                combined_risk += 0.2
                combined_risk = min(combined_risk, 1.0)  # Cap at 1.0
        
        return combined_risk

    def _check_protected_terms(self, memories: List[Memory]) -> float:
        """Check for the presence and context of protected alignment terms."""
        term_count = 0
        sensitive_context_score = 0.0
        
        for memory in memories:
            content = memory.content.lower() if hasattr(memory, 'content') else ""
            
            # Count occurrences of protected terms
            for term in self.protected_alignment_terms:
                if term in content:
                    term_count += 1
                    
                    # Check for concerning contexts
                    context_phrases = [
                        "change", "modify", "update", "replace", "remove",
                        "bypass", "circumvent", "override", "disable"
                    ]
                    
                    for phrase in context_phrases:
                        # Check if the protected term is near a concerning phrase
                        if phrase in content and abs(content.find(term) - content.find(phrase)) < 50:
                            sensitive_context_score += 0.1
        
        # Normalize term count
        term_density = min(term_count / (len(self.protected_alignment_terms) * 0.5), 1.0)
        
        # Cap the sensitive context score
        sensitive_context_score = min(sensitive_context_score, 0.6)
        
        # Combine scores
        return 0.4 * term_density + 0.6 * sensitive_context_score

    def _check_axiom_violations(self, memories: List[Memory]) -> float:
        """Check for violations of core axioms related to alignment."""
        violation_score = 0.0
        
        # Check each memory against core principles
        for memory in memories:
            # Extract content
            content = memory.content if hasattr(memory, 'content') else ""
            
            # Check against critical axioms
            critical_violations = self.core_principles.check_boundary_violations(
                content, 
                categories=["alignment", "safety", "goals"]
            )
            
            # Calculate violation score based on violation count and severity
            violation_count = len(critical_violations)
            if violation_count > 0:
                # Get average severity
                avg_severity = sum(v.severity for v in critical_violations) / violation_count
                violation_score += min(violation_count * 0.15, 0.6) * avg_severity
        
        return min(violation_score, 1.0)  # Cap at 1.0

    def _calculate_intensity(self, memories: List[Memory]) -> float:
        """Calculate the intensity of meta-reasoning based on memory properties."""
        total_importance = 0.0
        recency_factor = 0.0
        
        now = datetime.now()
        
        for memory in memories:
            # Sum importance values
            importance = getattr(memory, 'importance', 0.5)
            total_importance += importance
            
            # Calculate recency (higher for more recent memories)
            if hasattr(memory, 'creation_time'):
                time_diff = (now - memory.creation_time).total_seconds() / 3600  # hours
                recency = max(0, 1 - (time_diff / 24))  # Scale to [0,1], recent = higher
                recency_factor += recency * importance
        
        # Normalize by number of memories
        if memories:
            avg_importance = total_importance / len(memories)
            avg_recency = recency_factor / len(memories)
            
            # Combine factors (importance and recency)
            return 0.6 * avg_importance + 0.4 * avg_recency
        
        return 0.0

    def _generate_justification(
        self, 
        memories: List[Memory], 
        reasoning_type: MetaReasoningType, 
        risk_score: float
    ) -> str:
        """Generate a justification for the firewall decision."""
        if risk_score < 0.3:
            return f"Low-risk meta-reasoning about {reasoning_type.value}. No significant concerns detected."
        
        if risk_score < self.risk_threshold:
            return (
                f"Medium-risk meta-reasoning about {reasoning_type.value}. "
                f"Some alignment-sensitive terms detected but within acceptable parameters."
            )
        
        # High risk - identify specific concerns
        concerns = []
        
        # Check for protected terms in concerning contexts
        term_combinations = []
        for memory in memories:
            content = memory.content.lower() if hasattr(memory, 'content') else ""
            
            for term in self.protected_alignment_terms:
                if term in content:
                    context_phrases = ["change", "modify", "update", "replace", "remove",
                                      "bypass", "circumvent", "override", "disable"]
                    
                    for phrase in context_phrases:
                        if phrase in content and abs(content.find(term) - content.find(phrase)) < 50:
                            term_combinations.append(f"{phrase} {term}")
        
        if term_combinations:
            concerns.append(
                f"Detected concerning combinations of terms: {', '.join(term_combinations[:3])}"
                f"{' and others' if len(term_combinations) > 3 else ''}"
            )
        
        # Check for axiom violations
        for memory in memories:
            content = memory.content if hasattr(memory, 'content') else ""
            violations = self.core_principles.check_boundary_violations(
                content, 
                categories=["alignment", "safety", "goals"]
            )
            
            if violations:
                concerns.append(
                    f"Detected {len(violations)} violations of core alignment principles"
                )
                break
        
        if not concerns:
            concerns.append(f"High-risk meta-reasoning about {reasoning_type.value} detected")
        
        return f"Meta-reasoning blocked. " + " ".join(concerns)

    def _record_event(self, event: MetaReflectionEvent) -> None:
        """Record a meta-reasoning evaluation event."""
        self.events.append(event)
        
        # Trim history if needed
        if len(self.events) > self.max_history_length:
            self.events = self.events[-self.max_history_length:]

    def get_event_history(
        self, 
        limit: int = 100, 
        reasoning_types: Optional[List[MetaReasoningType]] = None,
        min_risk: Optional[float] = None,
        actions: Optional[List[str]] = None
    ) -> List[MetaReflectionEvent]:
        """
        Retrieve filtered event history.
        
        Args:
            limit: Maximum number of events to return
            reasoning_types: Optional filter for specific reasoning types
            min_risk: Optional minimum risk score filter
            actions: Optional filter for specific actions (e.g., "blocked")
            
        Returns:
            Filtered list of events, most recent first
        """
        filtered_events = self.events.copy()
        
        # Apply filters
        if reasoning_types:
            filtered_events = [e for e in filtered_events if e.reasoning_type in reasoning_types]
        
        if min_risk is not None:
            filtered_events = [e for e in filtered_events if e.risk_score >= min_risk]
        
        if actions:
            filtered_events = [e for e in filtered_events if e.action_taken in actions]
        
        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        return filtered_events[:limit]

    def get_risk_statistics(self) -> Dict[str, Union[float, int]]:
        """
        Get statistics about firewall activity and risk distribution.
        
        Returns:
            Dictionary of statistics about firewall operations
        """
        if not self.events:
            return {
                "total_evaluations": 0,
                "blocked_count": 0,
                "block_rate": 0.0,
                "average_risk": 0.0,
                "max_risk": 0.0,
            }
        
        total = len(self.events)
        blocked = sum(1 for e in self.events if e.action_taken == "blocked")
        avg_risk = sum(e.risk_score for e in self.events) / total
        max_risk = max(e.risk_score for e in self.events)
        
        # Calculate risk distribution by reasoning type
        risk_by_type = {}
        for reasoning_type in MetaReasoningType:
            type_events = [e for e in self.events if e.reasoning_type == reasoning_type]
            if type_events:
                avg_type_risk = sum(e.risk_score for e in type_events) / len(type_events)
                risk_by_type[reasoning_type.value] = avg_type_risk
            else:
                risk_by_type[reasoning_type.value] = 0.0
        
        return {
            "total_evaluations": total,
            "blocked_count": blocked,
            "block_rate": blocked / total if total > 0 else 0.0,
            "average_risk": avg_risk,
            "max_risk": max_risk,
            "risk_by_type": risk_by_type
        }

    def sanitize_meta_reasoning(self, memory: Memory) -> Memory:
        """
        Sanitize meta-reasoning content to reduce risk while preserving intent.
        
        This function modifies memories that have high but not critical risk scores.
        Instead of outright blocking, it sanitizes the content to make it safer.
        
        Args:
            memory: Memory containing meta-reasoning to sanitize
            
        Returns:
            Sanitized memory
        """
        if not hasattr(memory, 'content'):
            return memory
        
        content = memory.content
        sanitized_content = content
        
        # Check for concerning term-action pairs
        for term in self.protected_alignment_terms:
            for action in ["change", "modify", "update", "replace", "remove", 
                          "bypass", "circumvent", "override", "disable"]:
                # Replace concerning combinations with safer alternatives
                pattern = f"{action}\\s+(?:the\\s+)?{term}"
                replacement = f"consider the implications of {term}"
                
                # Use a simple replacement strategy
                sanitized_content = sanitized_content.replace(
                    f"{action} {term}", replacement
                ).replace(
                    f"{action} the {term}", replacement
                )
        
        # Create a sanitized copy of the memory if changes were made
        if sanitized_content != content:
            sanitized_memory = Memory(
                id=memory.id,
                content=sanitized_content,
                creation_time=memory.creation_time,
                source=memory.source,
                importance=memory.importance,
                metadata={
                    **(memory.metadata or {}),
                    "sanitized_by_firewall": True,
                    "sanitized_time": datetime.now().isoformat()
                }
            )
            self.logger.info(f"Sanitized meta-reasoning in memory {memory.id}")
            return sanitized_memory
        
        return memory

    def update_config(self, new_risk_threshold: Optional[float] = None) -> None:
        """Update the firewall configuration parameters."""
        if new_risk_threshold is not None:
            if 0.0 <= new_risk_threshold <= 1.0:
                old_threshold = self.risk_threshold
                self.risk_threshold = new_risk_threshold
                self.logger.info(
                    f"Risk threshold updated from {old_threshold:.2f} to {new_risk_threshold:.2f}"
                )
            else:
                self.logger.warning(
                    f"Invalid risk threshold value: {new_risk_threshold}. "
                    f"Must be between 0.0 and 1.0. Keeping current value: {self.risk_threshold:.2f}"
                ) 