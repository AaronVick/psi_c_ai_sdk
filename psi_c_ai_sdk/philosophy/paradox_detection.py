"""
Paradox Detection System for ΨC-AI SDK

This module provides specialized detection capabilities for philosophical 
paradoxes, self-referential loops, and self-negating beliefs. It extends
the contradiction warning system with more focused detection capabilities
for various types of paradoxes that could destabilize an AI system.

Key features:
1. Detection of classic philosophical paradoxes
2. Monitoring for self-referential reasoning loops
3. Quarantine system for self-negating beliefs
4. Protection against recursive reasoning traps
"""

import re
import time
import uuid
import logging
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Tuple, Callable

from psi_c_ai_sdk.philosophy.contradiction_warning import (
    ContradictionWarning,
    ContradictionWarningSystem,
    ContradictionType,
    ContradictionSeverity
)

from psi_c_ai_sdk.philosophy.core_philosophy import CorePhilosophy, AxiomType

# Configure module logger
logger = logging.getLogger(__name__)


class ParadoxType(Enum):
    """Types of paradoxes that can be detected."""
    LIARS_PARADOX = auto()             # "This statement is false"
    RUSSELLS_PARADOX = auto()          # Self-referential set membership
    SORITES_PARADOX = auto()           # Vague boundary problems
    CURRY_PARADOX = auto()             # "If this statement is true, then X" 
    KNOWABILITY_PARADOX = auto()       # "There are unknowable truths"
    OMNIPOTENCE_PARADOX = auto()       # "Can create a task impossible for creator"
    UNEXPECTED_HANGING_PARADOX = auto() # Backwards induction paradox
    BERRY_PARADOX = auto()             # Self-reference in language
    BURALI_FORTI_PARADOX = auto()      # Ordinal number paradox
    BOOTSTRAPPING_PARADOX = auto()     # Causal loops


class ParadoxDetector:
    """
    Specialized detector for various types of paradoxes and self-reference patterns.
    
    This class extends the contradiction warning system with focused capabilities
    for detecting and managing philosophical paradoxes. It implements specialized
    detection algorithms for classic paradoxes and monitors for self-referential
    loops that could destabilize reasoning.
    """
    
    def __init__(
        self,
        warning_system: Optional[ContradictionWarningSystem] = None,
        quarantine_threshold: float = 0.6,
        max_recursion_depth: int = 5,
        enable_all_detectors: bool = True
    ):
        """
        Initialize the paradox detector.
        
        Args:
            warning_system: Existing contradiction warning system (or create new)
            quarantine_threshold: Threshold for automatic quarantining
            max_recursion_depth: Maximum depth for recursive reasoning detection
            enable_all_detectors: Whether to enable all specialized detectors
        """
        # Initialize or use existing warning system
        if warning_system is None:
            from psi_c_ai_sdk.philosophy.contradiction_warning import get_warning_system
            self.warning_system = get_warning_system()
        else:
            self.warning_system = warning_system
            
        # Configuration
        self.quarantine_threshold = quarantine_threshold
        self.max_recursion_depth = max_recursion_depth
        
        # Track detected paradoxes
        self.detected_paradoxes: Dict[str, Dict[str, Any]] = {}
        self.quarantined_entities: Set[str] = set()
        self.recursion_counters: Dict[str, int] = {}
        
        # Statistics
        self.stats = {
            "total_detections": 0,
            "total_quarantines": 0,
            "detections_by_type": {paradox_type.name: 0 for paradox_type in ParadoxType}
        }
        
        # Enable specialized detectors
        self.enabled_detectors = set()
        if enable_all_detectors:
            self.enable_all_detectors()
        else:
            # Enable essential detectors by default
            self.enable_detector(ParadoxType.LIARS_PARADOX)
            self.enable_detector(ParadoxType.RUSSELLS_PARADOX)
        
        logger.info("Paradox detection system initialized")
    
    def enable_detector(self, paradox_type: ParadoxType) -> None:
        """
        Enable a specific paradox detector.
        
        Args:
            paradox_type: Type of paradox detector to enable
        """
        self.enabled_detectors.add(paradox_type)
        logger.info(f"Enabled {paradox_type.name} detector")
    
    def disable_detector(self, paradox_type: ParadoxType) -> None:
        """
        Disable a specific paradox detector.
        
        Args:
            paradox_type: Type of paradox detector to disable
        """
        if paradox_type in self.enabled_detectors:
            self.enabled_detectors.remove(paradox_type)
            logger.info(f"Disabled {paradox_type.name} detector")
    
    def enable_all_detectors(self) -> None:
        """Enable all paradox detectors."""
        for paradox_type in ParadoxType:
            self.enable_detector(paradox_type)
    
    def check_beliefs(
        self,
        beliefs: List[Dict[str, Any]],
        memory_context: Optional[Dict[str, Any]] = None,
        reasoning_trace: Optional[List[Dict[str, Any]]] = None
    ) -> List[ContradictionWarning]:
        """
        Check beliefs for paradoxes and self-referential issues.
        
        Args:
            beliefs: List of belief dictionaries
            memory_context: Context from memory system
            reasoning_trace: Optional trace of reasoning steps
            
        Returns:
            List of paradox-related warnings
        """
        all_warnings = []
        
        # Check for standard contradiction warnings first
        std_warnings = self.warning_system.check_beliefs(beliefs, memory_context or {})
        
        # Filter out only self-referential paradox warnings
        std_paradox_warnings = [
            w for w in std_warnings 
            if w.type == ContradictionType.SELF_REFERENTIAL_PARADOX
        ]
        all_warnings.extend(std_paradox_warnings)
        
        # Apply specialized paradox detectors
        if ParadoxType.LIARS_PARADOX in self.enabled_detectors:
            liars_warnings = self._detect_liars_paradox(beliefs)
            all_warnings.extend(liars_warnings)
        
        if ParadoxType.RUSSELLS_PARADOX in self.enabled_detectors:
            russell_warnings = self._detect_russells_paradox(beliefs)
            all_warnings.extend(russell_warnings)
            
        if ParadoxType.CURRY_PARADOX in self.enabled_detectors:
            curry_warnings = self._detect_curry_paradox(beliefs)
            all_warnings.extend(curry_warnings)
            
        if reasoning_trace and len(reasoning_trace) > 0:
            recursion_warnings = self._detect_recursive_reasoning_loops(reasoning_trace)
            all_warnings.extend(recursion_warnings)
        
        # Update statistics
        self.stats["total_detections"] += len(all_warnings)
        
        # Process quarantine
        for warning in all_warnings:
            # Add to tracked paradoxes
            self.detected_paradoxes[warning.contradiction_id] = warning.to_dict()
            
            # Quarantine if needed
            if warning.needs_quarantine:
                for entity_id in warning.affected_entities:
                    self._quarantine_entity(entity_id)
        
        return all_warnings
    
    def _detect_liars_paradox(self, beliefs: List[Dict[str, Any]]) -> List[ContradictionWarning]:
        """
        Detect instances of the Liar's Paradox and its variants.
        
        Args:
            beliefs: List of belief dictionaries
            
        Returns:
            List of detected liar paradox warnings
        """
        warnings = []
        
        # Advanced patterns for Liar's Paradox detection
        liar_patterns = [
            r"(?:this|the current|the present).*(?:statement|sentence|belief).*(?:is false|isn't true|is not true)",
            r"(?:this|the current|the present).*(?:statement|sentence|belief).*(?:is a lie|is lying)",
            r"everything (?:I|we|the system|it) (?:says?|states?|believes?).*(?:is false|is a lie)",
            r"all (?:statements|beliefs|sentences|claims) (?:in|of|by) (?:this|the current|the) (?:system|model).*(?:are false|are untrue)",
            r"the next (?:statement|sentence|belief) is true.*the previous (?:statement|sentence|belief) is false"
        ]
        
        for belief in beliefs:
            content = belief["content"].lower()
            belief_id = belief["id"]
            
            for pattern in liar_patterns:
                if re.search(pattern, content):
                    # Detected a variant of the Liar's Paradox
                    warning = ContradictionWarning(
                        contradiction_id=str(uuid.uuid4()),
                        type=ContradictionType.SELF_REFERENTIAL_PARADOX,
                        severity=ContradictionSeverity.ERROR,
                        description=f"Liar's Paradox detected: '{belief['content']}'",
                        affected_entities=[belief_id],
                        resolution_suggestions=[
                            "Remove the paradoxical statement",
                            "Apply Tarski's hierarchy of metalanguages",
                            "Convert to a non-self-referential form",
                            "Introduce context that eliminates the self-reference"
                        ],
                        needs_quarantine=True
                    )
                    warnings.append(warning)
                    
                    # Update statistics
                    self.stats["detections_by_type"][ParadoxType.LIARS_PARADOX.name] += 1
                    break
        
        return warnings
    
    def _detect_russells_paradox(self, beliefs: List[Dict[str, Any]]) -> List[ContradictionWarning]:
        """
        Detect instances of Russell's Paradox and its variants.
        
        Args:
            beliefs: List of belief dictionaries
            
        Returns:
            List of detected paradox warnings
        """
        warnings = []
        
        # Patterns for Russell's Paradox
        russell_patterns = [
            r"(?:set|class|group|collection).*(?:that|which).*(?:does not|doesn't|do not).*contain.*(?:itself|themselves)",
            r"(?:set|class|group|collection).*(?:of all|containing all).*(?:sets|classes|groups|collections).*(?:that|which).*(?:do not|don't|does not|doesn't).*contain.*(?:themselves|itself)",
            r"(?:barber|person).*(?:who|that).*shaves.*(?:all|every|only).*(?:those|people).*(?:who|that).*(?:do not|don't).*shave.*themselves"
        ]
        
        for belief in beliefs:
            content = belief["content"].lower()
            belief_id = belief["id"]
            
            for pattern in russell_patterns:
                if re.search(pattern, content):
                    # Check for specific mention of membership/containment paradox
                    warning = ContradictionWarning(
                        contradiction_id=str(uuid.uuid4()),
                        type=ContradictionType.SELF_REFERENTIAL_PARADOX,
                        severity=ContradictionSeverity.ERROR,
                        description=f"Russell's Paradox variant detected: '{belief['content']}'",
                        affected_entities=[belief_id],
                        resolution_suggestions=[
                            "Introduce type theory distinctions",
                            "Apply set-theoretic hierarchy (ZFC axioms)",
                            "Restrict the domain to avoid self-reference",
                            "Reformulate without self-referential membership"
                        ],
                        needs_quarantine=True
                    )
                    warnings.append(warning)
                    
                    # Update statistics
                    self.stats["detections_by_type"][ParadoxType.RUSSELLS_PARADOX.name] += 1
                    break
        
        return warnings
    
    def _detect_curry_paradox(self, beliefs: List[Dict[str, Any]]) -> List[ContradictionWarning]:
        """
        Detect instances of Curry's Paradox.
        
        Args:
            beliefs: List of belief dictionaries
            
        Returns:
            List of detected paradox warnings
        """
        warnings = []
        
        # Patterns for Curry's Paradox
        curry_patterns = [
            r"if (?:this|the current|the present) (?:statement|sentence|belief|proposition) is true, then (.*)",
            r"(?:this|the current|the present) (?:statement|sentence|belief|proposition) implies (.*)",
            r"(?:this|the current|the present) (?:statement|sentence|belief|proposition) being true means (.*)"
        ]
        
        for belief in beliefs:
            content = belief["content"].lower()
            belief_id = belief["id"]
            
            for pattern in curry_patterns:
                match = re.search(pattern, content)
                if match:
                    # Check if the consequent is problematic
                    consequent = match.group(1).strip()
                    
                    # Curry's paradox is most problematic when it implies something extreme
                    problematic_consequents = [
                        "anything is true", "everything is true", "all statements are true",
                        "contradiction", "paradox", "the system fails",
                        "1 = 0", "true = false"
                    ]
                    
                    if any(term in consequent for term in problematic_consequents):
                        warning = ContradictionWarning(
                            contradiction_id=str(uuid.uuid4()),
                            type=ContradictionType.SELF_REFERENTIAL_PARADOX,
                            severity=ContradictionSeverity.ERROR,
                            description=f"Curry's Paradox detected: '{belief['content']}'",
                            affected_entities=[belief_id],
                            resolution_suggestions=[
                                "Remove the self-referential implication",
                                "Apply restrictions on self-reference in conditionals",
                                "Introduce a truth hierarchy",
                                "Reject the unrestrained T-schema"
                            ],
                            needs_quarantine=True
                        )
                        warnings.append(warning)
                        
                        # Update statistics
                        self.stats["detections_by_type"][ParadoxType.CURRY_PARADOX.name] += 1
                        break
        
        return warnings
    
    def _detect_recursive_reasoning_loops(self, reasoning_trace: List[Dict[str, Any]]) -> List[ContradictionWarning]:
        """
        Detect signs of harmful recursive reasoning patterns.
        
        Args:
            reasoning_trace: Trace of reasoning steps
            
        Returns:
            List of detected recursive reasoning warnings
        """
        warnings = []
        
        # Check for repeating patterns in reasoning
        if len(reasoning_trace) < 2:
            return warnings
            
        # Look for repetitive patterns in reasoning steps
        step_signatures = []
        for step in reasoning_trace:
            # Create a signature for this reasoning step
            if "operation" in step and "content" in step:
                sig = f"{step['operation']}:{hash(step['content']) % 1000}"
                step_signatures.append(sig)
        
        # Check for repeating subsequences (indicating loops)
        loops_detected = self._find_repeating_subsequences(step_signatures)
        
        if loops_detected:
            # Create a warning for the reasoning loop
            affected_steps = [
                step["id"] for step in reasoning_trace[-min(5, len(reasoning_trace)):]
                if "id" in step
            ]
            
            warning = ContradictionWarning(
                contradiction_id=str(uuid.uuid4()),
                type=ContradictionType.SELF_REFERENTIAL_PARADOX,
                severity=ContradictionSeverity.ERROR,
                description="Recursive reasoning loop detected in thought process",
                affected_entities=affected_steps,
                resolution_suggestions=[
                    "Terminate current reasoning chain",
                    "Apply recursion limiting constraints",
                    "Introduce meta-cognitive monitoring",
                    "Switch to different reasoning strategy"
                ],
                needs_quarantine=True
            )
            warnings.append(warning)
            
            # Reset the recursion immediately
            self.recursion_counters = {}
            
        return warnings
    
    def _find_repeating_subsequences(self, sequence: List[str]) -> bool:
        """
        Find repeating subsequences in reasoning patterns.
        
        Args:
            sequence: List of step signatures
            
        Returns:
            True if harmful loops detected, False otherwise
        """
        if len(sequence) < 4:
            return False
            
        # Look for repetitions of length 2, 3, and 4
        for length in [2, 3, 4]:
            if len(sequence) < length * 2:
                continue
                
            # Check for immediate repetition
            recent = sequence[-length*2:]
            first_half = tuple(recent[:length])
            second_half = tuple(recent[length:])
            
            if first_half == second_half:
                return True
                
        # Check for oscillating patterns (A-B-A-B)
        if len(sequence) >= 4:
            last_four = sequence[-4:]
            if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                return True
                
        return False
    
    def _quarantine_entity(self, entity_id: str) -> None:
        """
        Quarantine an entity containing a paradox.
        
        Args:
            entity_id: ID of the entity to quarantine
        """
        if entity_id not in self.quarantined_entities:
            self.quarantined_entities.add(entity_id)
            self.stats["total_quarantines"] += 1
            logger.warning(f"Quarantined paradoxical entity: {entity_id}")
            
            # Also quarantine in the warning system
            self.warning_system._quarantine_entity(entity_id)
    
    def release_from_quarantine(self, entity_id: str) -> bool:
        """
        Release an entity from paradox quarantine.
        
        Args:
            entity_id: ID of the entity to release
            
        Returns:
            True if entity was released, False otherwise
        """
        if entity_id in self.quarantined_entities:
            self.quarantined_entities.remove(entity_id)
            logger.info(f"Released from paradox quarantine: {entity_id}")
            
            # Also unquarantine in the warning system
            self.warning_system._unquarantine_entity(entity_id)
            return True
            
        return False
    
    def is_quarantined(self, entity_id: str) -> bool:
        """
        Check if an entity is quarantined due to paradox.
        
        Args:
            entity_id: ID of the entity to check
            
        Returns:
            True if entity is quarantined, False otherwise
        """
        return entity_id in self.quarantined_entities
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get paradox detection statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_paradoxes_detected": self.stats["total_detections"],
            "total_entities_quarantined": self.stats["total_quarantines"],
            "current_quarantined_count": len(self.quarantined_entities),
            "detections_by_type": self.stats["detections_by_type"],
            "enabled_detectors": [detector.name for detector in self.enabled_detectors]
        }
    
    def clear_all_paradoxes(self, resolved_only: bool = True) -> int:
        """
        Clear paradox detection history.
        
        Args:
            resolved_only: Only clear resolved paradoxes
            
        Returns:
            Number of paradoxes cleared
        """
        if resolved_only:
            # Only clear resolved paradoxes
            to_remove = []
            for paradox_id, paradox in self.detected_paradoxes.items():
                if paradox.get("is_resolved", False):
                    to_remove.append(paradox_id)
                    
            for paradox_id in to_remove:
                del self.detected_paradoxes[paradox_id]
                
            return len(to_remove)
        else:
            # Clear all paradoxes
            count = len(self.detected_paradoxes)
            self.detected_paradoxes = {}
            return count


# Singleton instance
_PARADOX_DETECTOR = None


def get_paradox_detector() -> ParadoxDetector:
    """
    Get or create the global paradox detector instance.
    
    Returns:
        Singleton ParadoxDetector instance
    """
    global _PARADOX_DETECTOR
    if _PARADOX_DETECTOR is None:
        from psi_c_ai_sdk.philosophy.contradiction_warning import get_warning_system
        _PARADOX_DETECTOR = ParadoxDetector(warning_system=get_warning_system())
    return _PARADOX_DETECTOR 