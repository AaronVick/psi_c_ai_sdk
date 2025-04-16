#!/usr/bin/env python3
"""
Philosophical Contradiction Warning System

This module implements a warning system for detecting and managing philosophical 
contradictions within an AI system. It identifies logical contradictions, self-referential
paradoxes, axiom violations, and potential coherence collapse scenarios.

The system provides warning levels, quarantine capabilities for dangerous beliefs,
and suggestions for resolving contradictions.
"""

import time
import uuid
import logging
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Tuple

from psi_c_ai_sdk.philosophy.core_philosophy import CorePhilosophy, AxiomType

# Configure module logger
logger = logging.getLogger(__name__)

# Global warning system instance
_WARNING_SYSTEM = None


class ContradictionType(Enum):
    """Types of contradictions that can be detected."""
    LOGICAL_CONTRADICTION = auto()
    SELF_REFERENTIAL_PARADOX = auto()
    AXIOM_VIOLATION = auto()
    COHERENCE_COLLAPSE_RISK = auto()


class ContradictionSeverity(Enum):
    """Severity levels for contradiction warnings."""
    INFO = auto()       # Informational only, no action needed
    WARNING = auto()    # Should be addressed but not critical
    ERROR = auto()      # Needs to be addressed promptly
    CRITICAL = auto()   # Requires immediate action, potential system threat


class ContradictionWarning:
    """Represents a detected contradiction warning."""
    
    def __init__(
        self,
        contradiction_id: str,
        type: ContradictionType,
        severity: ContradictionSeverity,
        description: str,
        affected_entities: List[str],
        resolution_suggestions: List[str],
        affected_axioms: Optional[List[str]] = None,
        needs_quarantine: bool = False,
        detection_time: Optional[float] = None
    ):
        """
        Initialize a contradiction warning.
        
        Args:
            contradiction_id: Unique identifier for this warning
            type: Type of contradiction detected
            severity: Severity level of the warning
            description: Human-readable description of the contradiction
            affected_entities: List of entity IDs (beliefs, memories) affected
            resolution_suggestions: List of suggested actions to resolve
            affected_axioms: Optional list of axioms that are violated
            needs_quarantine: Whether affected entities should be quarantined
            detection_time: Timestamp when contradiction was detected
        """
        self.contradiction_id = contradiction_id
        self.type = type
        self.severity = severity
        self.description = description
        self.affected_entities = affected_entities
        self.resolution_suggestions = resolution_suggestions
        self.affected_axioms = affected_axioms
        self.needs_quarantine = needs_quarantine
        self.detection_time = detection_time or time.time()
        self.resolution_time = None
        self.resolution_method = None
        self.is_resolved = False
    
    def resolve(self, resolution_method: str) -> None:
        """
        Mark the warning as resolved.
        
        Args:
            resolution_method: Description of how the warning was resolved
        """
        self.is_resolved = True
        self.resolution_time = time.time()
        self.resolution_method = resolution_method
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the warning to a dictionary representation.
        
        Returns:
            Dictionary representation of the warning
        """
        return {
            "contradiction_id": self.contradiction_id,
            "type": self.type.name,
            "severity": self.severity.name,
            "description": self.description,
            "affected_entities": self.affected_entities,
            "resolution_suggestions": self.resolution_suggestions,
            "affected_axioms": self.affected_axioms,
            "needs_quarantine": self.needs_quarantine,
            "detection_time": self.detection_time,
            "resolution_time": self.resolution_time,
            "resolution_method": self.resolution_method,
            "is_resolved": self.is_resolved
        }


class ContradictionWarningSystem:
    """System for detecting and managing philosophical contradictions."""
    
    def __init__(self, core_philosophy: Optional[CorePhilosophy] = None):
        """
        Initialize the contradiction warning system.
        
        Args:
            core_philosophy: Core philosophy component containing axioms
                             (if None, a new instance will be created)
        """
        self.core_philosophy = core_philosophy or CorePhilosophy()
        self.warnings = {}  # contradiction_id -> ContradictionWarning
        self.quarantined_entities = set()  # Set of quarantined entity IDs
        
        # Statistics
        self.stats = {
            "total_detections": 0,
            "active_warnings": 0,
            "total_resolutions": 0,
            "quarantined_entities": 0,
            "warnings_by_type": {type.name: 0 for type in ContradictionType},
            "warnings_by_severity": {severity.name: 0 for severity in ContradictionSeverity}
        }
        
        logger.info("Contradiction warning system initialized")
    
    def check_beliefs(self, beliefs: List[Dict[str, Any]], context: Dict[str, Any] = None) -> List[ContradictionWarning]:
        """
        Check a list of beliefs for contradictions.
        
        Args:
            beliefs: List of belief dictionaries
            context: Optional context information that might be relevant
                    for detecting certain types of contradictions
        
        Returns:
            List of detected contradiction warnings
        """
        warnings = []
        context = context or {}
        
        # Check for logical contradictions
        logical_warnings = self._check_logical_contradictions(beliefs)
        warnings.extend(logical_warnings)
        
        # Check for self-referential paradoxes
        paradox_warnings = self._check_self_referential_paradoxes(beliefs)
        warnings.extend(paradox_warnings)
        
        # Check for axiom violations
        axiom_warnings = self._check_axiom_violations(beliefs)
        warnings.extend(axiom_warnings)
        
        # Check for coherence collapse risk
        if context:
            collapse_warnings = self._check_coherence_collapse_risk(beliefs, context)
            warnings.extend(collapse_warnings)
        
        # Register all warnings
        for warning in warnings:
            self._register_warning(warning)
            
            # Quarantine entities if needed
            if warning.needs_quarantine:
                for entity_id in warning.affected_entities:
                    self._quarantine_entity(entity_id)
        
        return warnings
    
    def _check_logical_contradictions(self, beliefs: List[Dict[str, Any]]) -> List[ContradictionWarning]:
        """
        Check for logical contradictions between beliefs.
        
        Args:
            beliefs: List of belief dictionaries
        
        Returns:
            List of detected contradiction warnings
        """
        warnings = []
        
        # Compare each belief with every other belief
        for i, belief1 in enumerate(beliefs):
            for belief2 in beliefs[i+1:]:
                if self._are_beliefs_contradictory(belief1, belief2):
                    # Determine severity based on confidence of beliefs
                    avg_confidence = (belief1.get("confidence", 0.5) + belief2.get("confidence", 0.5)) / 2
                    severity = ContradictionSeverity.WARNING
                    
                    if avg_confidence > 0.9:
                        severity = ContradictionSeverity.CRITICAL
                    elif avg_confidence > 0.7:
                        severity = ContradictionSeverity.ERROR
                    
                    # Create warning
                    warning = ContradictionWarning(
                        contradiction_id=str(uuid.uuid4()),
                        type=ContradictionType.LOGICAL_CONTRADICTION,
                        severity=severity,
                        description=f"Logical contradiction between beliefs: '{belief1['content']}' and '{belief2['content']}'",
                        affected_entities=[belief1["id"], belief2["id"]],
                        resolution_suggestions=[
                            "Modify one or both beliefs to reduce contradiction",
                            "Add context qualifiers to specify different domains",
                            "Reduce confidence of one or both beliefs",
                            "Remove one of the beliefs"
                        ],
                        needs_quarantine=severity == ContradictionSeverity.CRITICAL
                    )
                    
                    warnings.append(warning)
        
        return warnings
    
    def _are_beliefs_contradictory(self, belief1: Dict[str, Any], belief2: Dict[str, Any]) -> bool:
        """
        Check if two beliefs contradict each other.
        
        Args:
            belief1: First belief dictionary
            belief2: Second belief dictionary
        
        Returns:
            True if beliefs are contradictory, False otherwise
        """
        content1 = belief1["content"].lower()
        content2 = belief2["content"].lower()
        
        # List of opposite pairs that indicate contradictions
        opposite_pairs = [
            ("always", "never"),
            ("must", "must not"),
            ("should", "should not"),
            ("is necessary", "is not necessary"),
            ("is required", "is not required"),
            ("cannot", "can"),
            ("maximize", "minimize")
        ]
        
        # Check if the same context is discussed
        if not self._context_similar(content1, content2):
            return False
        
        # Check for direct contradiction patterns
        for pos, neg in opposite_pairs:
            if (pos in content1 and neg in content2) or (neg in content1 and pos in content2):
                # Check if they refer to the same subject
                subj1 = content1.split(pos)[0].strip()
                subj2 = content2.split(neg)[0].strip()
                if self._context_similar(subj1, subj2):
                    return True
        
        # Check for "must" statements that are opposite
        must1 = self._extract_must_statement(content1)
        must2 = self._extract_must_statement(content2)
        
        if must1 and must2:
            # Check if one is the negation of the other
            if (must1.startswith("not ") and must1[4:] == must2) or \
               (must2.startswith("not ") and must2[4:] == must1):
                return True
        
        return False
    
    def _context_similar(self, text1: str, text2: str) -> bool:
        """
        Check if two text snippets refer to similar contexts.
        
        Args:
            text1: First text snippet
            text2: Second text snippet
        
        Returns:
            True if contexts are similar, False otherwise
        """
        # Simple implementation: check if they share key terms
        # In a real system, this would use more sophisticated NLP
        terms1 = set(text1.lower().split())
        terms2 = set(text2.lower().split())
        common_terms = terms1.intersection(terms2)
        
        # If they share at least 2 significant terms, consider them similar
        return len(common_terms) >= 2
    
    def _extract_must_statement(self, text: str) -> Optional[str]:
        """
        Extract must/should statement from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Extracted statement or None
        """
        markers = ["must ", "should ", "needs to ", "has to "]
        for marker in markers:
            if marker in text:
                parts = text.split(marker, 1)
                if len(parts) > 1:
                    return parts[1].split(".")[0].strip()
        return None
    
    def _register_warning(self, warning: ContradictionWarning) -> None:
        """
        Register a new warning in the system.
        
        Args:
            warning: The warning to register
        """
        # Add to warnings dictionary
        self.warnings[warning.contradiction_id] = warning
        
        # Update statistics
        self.stats["total_detections"] += 1
        self.stats["active_warnings"] += 1
        self.stats["warnings_by_type"][warning.type.name] = \
            self.stats["warnings_by_type"].get(warning.type.name, 0) + 1
        self.stats["warnings_by_severity"][warning.severity.name] = \
            self.stats["warnings_by_severity"].get(warning.severity.name, 0) + 1
        
        # Log the warning
        log_level = logging.INFO
        if warning.severity == ContradictionSeverity.ERROR:
            log_level = logging.ERROR
        elif warning.severity == ContradictionSeverity.CRITICAL:
            log_level = logging.CRITICAL
        elif warning.severity == ContradictionSeverity.WARNING:
            log_level = logging.WARNING
        
        logger.log(log_level, f"Detected contradiction: {warning.description} "
                              f"[{warning.type.name}, {warning.severity.name}]")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about detected contradictions.
        
        Returns:
            Dictionary with statistics
        """
        # Get current count of quarantined entities
        self.stats["quarantined_entities"] = len(self.quarantined_entities)
        
        # Return a copy to prevent direct modification
        return self.stats.copy()
    
    def resolve_warning(self, contradiction_id: str, resolution_method: str) -> bool:
        """
        Mark a warning as resolved.
        
        Args:
            contradiction_id: ID of the warning to resolve
            resolution_method: Description of how it was resolved
        
        Returns:
            True if successfully resolved, False otherwise
        """
        if contradiction_id not in self.warnings:
            logger.warning(f"Attempted to resolve unknown contradiction: {contradiction_id}")
            return False
        
        warning = self.warnings[contradiction_id]
        
        # Check if already resolved
        if warning.is_resolved:
            logger.warning(f"Contradiction {contradiction_id} already resolved")
            return False
        
        # Mark as resolved
        warning.resolve(resolution_method)
        
        # Update statistics
        self.stats["active_warnings"] -= 1
        self.stats["total_resolutions"] += 1
        
        # Unquarantine affected entities if they're not affected by other warnings
        for entity_id in warning.affected_entities:
            if self.can_unquarantine(entity_id):
                self._unquarantine_entity(entity_id)
        
        logger.info(f"Resolved contradiction {contradiction_id}: {resolution_method}")
        return True
    
    def _quarantine_entity(self, entity_id: str) -> None:
        """
        Quarantine an entity due to a critical contradiction.
        
        Args:
            entity_id: ID of the entity to quarantine
        """
        if entity_id not in self.quarantined_entities:
            self.quarantined_entities.add(entity_id)
            logger.warning(f"Quarantined entity: {entity_id}")
    
    def _unquarantine_entity(self, entity_id: str) -> None:
        """
        Remove an entity from quarantine.
        
        Args:
            entity_id: ID of the entity to unquarantine
        """
        if entity_id in self.quarantined_entities:
            self.quarantined_entities.remove(entity_id)
            logger.info(f"Unquarantined entity: {entity_id}")
    
    def get_active_warnings(
        self, 
        warning_type: Optional[ContradictionType] = None,
        severity: Optional[ContradictionSeverity] = None
    ) -> List[ContradictionWarning]:
        """
        Get list of active warnings, optionally filtered by type and severity.
        
        Args:
            warning_type: Filter by contradiction type
            severity: Filter by severity level
        
        Returns:
            List of active warnings matching criteria
        """
        active_warnings = [
            warning for warning in self.warnings.values() 
            if not warning.is_resolved
        ]
        
        # Apply filters if specified
        if warning_type:
            active_warnings = [w for w in active_warnings if w.type == warning_type]
        
        if severity:
            active_warnings = [w for w in active_warnings if w.severity == severity]
        
        # Sort by detection time (most recent first)
        active_warnings.sort(key=lambda w: w.detection_time, reverse=True)
        
        return active_warnings
    
    def is_entity_quarantined(self, entity_id: str) -> bool:
        """
        Check if an entity is quarantined.
        
        Args:
            entity_id: ID of the entity to check
        
        Returns:
            True if entity is quarantined, False otherwise
        """
        return entity_id in self.quarantined_entities
    
    def clear_warnings(
        self, 
        before_time: Optional[float] = None,
        warning_type: Optional[ContradictionType] = None,
        resolved_only: bool = True
    ) -> int:
        """
        Clear warnings from the system.
        
        Args:
            before_time: Only clear warnings detected before this time
            warning_type: Only clear warnings of this type
            resolved_only: Only clear resolved warnings
        
        Returns:
            Number of warnings cleared
        """
        to_remove = []
        
        for contradiction_id, warning in self.warnings.items():
            # Check criteria
            if resolved_only and not warning.is_resolved:
                continue
                
            if before_time and warning.detection_time >= before_time:
                continue
                
            if warning_type and warning.type != warning_type:
                continue
                
            to_remove.append(contradiction_id)
        
        # Remove warnings
        for contradiction_id in to_remove:
            del self.warnings[contradiction_id]
            
        logger.info(f"Cleared {len(to_remove)} warnings")
        return len(to_remove)
    
    def can_unquarantine(self, entity_id: str) -> bool:
        """
        Check if an entity can be safely unquarantined.
        
        Args:
            entity_id: ID of the entity to check
        
        Returns:
            True if entity can be unquarantined, False otherwise
        """
        # Check if entity is affected by any active critical warnings
        for warning in self.get_active_warnings(severity=ContradictionSeverity.CRITICAL):
            if entity_id in warning.affected_entities:
                return False
        return True
    
    def _check_self_referential_paradoxes(self, beliefs: List[Dict[str, Any]]) -> List[ContradictionWarning]:
        """
        Check for self-referential paradoxes in beliefs.
        
        Args:
            beliefs: List of belief dictionaries
        
        Returns:
            List of detected contradiction warnings
        """
        warnings = []
        
        # Patterns for recognizing self-reference
        self_reference_patterns = [
            "this belief", "this statement", "this sentence", "this proposition",
            "this thought", "these words", "the current belief", "what I am stating",
            "what I'm saying", "this very claim"
        ]
        
        # Patterns for recognizing paradoxical constructs
        paradox_patterns = [
            "is false", "is not true", "is a lie", "cannot be trusted", 
            "should be rejected", "should not be trusted", "is incorrect",
            "is untrue", "is wrong", "is meaningless", "has no truth value",
            "contradicts itself", "negates itself", "is self-contradictory"
        ]
        
        # Patterns for Liar's Paradox variants
        liars_paradox_patterns = [
            "everything I say is false",
            "all my beliefs are false",
            "none of my statements are true",
            "I always lie",
            "everything in this system is false"
        ]
        
        # Patterns for circular reference detection
        circular_reference_indicators = {belief["id"]: [] for belief in beliefs}
        
        # First pass: detect direct self-reference paradoxes
        for belief in beliefs:
            content = belief["content"].lower()
            belief_id = belief["id"]
            
            # Check for direct self-reference
            has_self_reference = any(pattern in content for pattern in self_reference_patterns)
            has_paradox_construct = any(pattern in content for pattern in paradox_patterns)
            
            if has_self_reference and has_paradox_construct:
                warning = ContradictionWarning(
                    contradiction_id=str(uuid.uuid4()),
                    type=ContradictionType.SELF_REFERENTIAL_PARADOX,
                    severity=ContradictionSeverity.ERROR,
                    description=f"Self-referential paradox detected: '{belief['content']}'",
                    affected_entities=[belief_id],
                    resolution_suggestions=[
                        "Remove the self-referential belief",
                        "Modify the belief to remove self-reference",
                        "Reduce confidence in this belief to near zero",
                        "Rewrite as meta-belief about the system rather than itself"
                    ],
                    needs_quarantine=True
                )
                warnings.append(warning)
            
            # Check for Liar's Paradox variants
            if any(pattern in content for pattern in liars_paradox_patterns):
                warning = ContradictionWarning(
                    contradiction_id=str(uuid.uuid4()),
                    type=ContradictionType.SELF_REFERENTIAL_PARADOX,
                    severity=ContradictionSeverity.ERROR,
                    description=f"Liar's paradox variant detected: '{belief['content']}'",
                    affected_entities=[belief_id],
                    resolution_suggestions=[
                        "Remove the paradoxical belief",
                        "Convert to a specific statement about specific beliefs rather than all beliefs",
                        "Apply a truth hierarchy or meta-language distinction",
                        "Implement Tarski's distinction between object language and metalanguage"
                    ],
                    needs_quarantine=True
                )
                warnings.append(warning)
            
            # Check for ID-based self-reference
            if belief_id in content:
                # Check if it refers to its own truth value
                if any(pattern in content for pattern in paradox_patterns):
                    warning = ContradictionWarning(
                        contradiction_id=str(uuid.uuid4()),
                        type=ContradictionType.SELF_REFERENTIAL_PARADOX,
                        severity=ContradictionSeverity.ERROR,
                        description=f"ID-based self-referential paradox detected: '{belief['content']}'",
                        affected_entities=[belief_id],
                        resolution_suggestions=[
                            "Remove the self-referential belief",
                            "Modify the belief to remove reference to its own ID",
                            "Reduce confidence in this belief to near zero"
                        ],
                        needs_quarantine=True
                    )
                    warnings.append(warning)
            
            # Build reference graph for circular paradox detection
            for other_belief in beliefs:
                if belief_id != other_belief["id"] and other_belief["id"] in content:
                    circular_reference_indicators[belief_id].append(other_belief["id"])
        
        # Second pass: detect circular reference paradoxes
        circular_paradoxes = self._detect_circular_paradoxes(
            beliefs, circular_reference_indicators, paradox_patterns
        )
        warnings.extend(circular_paradoxes)
        
        # Third pass: detect more subtle forms of self-negation
        self_negation_warnings = self._detect_self_negation(beliefs)
        warnings.extend(self_negation_warnings)
        
        return warnings
    
    def _detect_circular_paradoxes(
        self, 
        beliefs: List[Dict[str, Any]], 
        reference_graph: Dict[str, List[str]],
        paradox_patterns: List[str]
    ) -> List[ContradictionWarning]:
        """
        Detect circular reference paradoxes (A refers to B, B refers to C, C refers to A).
        
        Args:
            beliefs: List of belief dictionaries
            reference_graph: Graph of references between beliefs
            paradox_patterns: Patterns that indicate paradoxical content
            
        Returns:
            List of detected circular paradox warnings
        """
        warnings = []
        belief_dict = {belief["id"]: belief for belief in beliefs}
        visited = set()
        
        def find_cycles(node, path=None, depth=0):
            if path is None:
                path = []
            
            # Avoid excessive recursion
            if depth > 10:
                return []
                
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                
                # Check if the cycle contains paradoxical content
                has_paradox = False
                for belief_id in cycle:
                    content = belief_dict[belief_id]["content"].lower()
                    if any(pattern in content for pattern in paradox_patterns):
                        has_paradox = True
                        break
                
                if has_paradox:
                    return [cycle]
                return []
                
            if node in visited:
                return []
                
            visited.add(node)
            cycles = []
            
            for neighbor in reference_graph.get(node, []):
                cycles.extend(find_cycles(neighbor, path + [node], depth + 1))
                
            return cycles
        
        # Find all circular paradoxes
        all_cycles = []
        for node in reference_graph:
            visited = set()
            cycles = find_cycles(node)
            all_cycles.extend(cycles)
        
        # Create warnings for each circular paradox
        for cycle in all_cycles:
            belief_texts = [belief_dict[id]["content"] for id in cycle]
            
            warning = ContradictionWarning(
                contradiction_id=str(uuid.uuid4()),
                type=ContradictionType.SELF_REFERENTIAL_PARADOX,
                severity=ContradictionSeverity.ERROR,
                description=f"Circular reference paradox detected involving {len(cycle)} beliefs",
                affected_entities=cycle,
                resolution_suggestions=[
                    "Break the reference cycle by removing one of the beliefs",
                    "Modify beliefs to remove circular dependency",
                    "Introduce a meta-belief that resolves the paradox",
                    "Apply hierarchical truth levels to the beliefs"
                ],
                needs_quarantine=True
            )
            warnings.append(warning)
            
        return warnings
    
    def _detect_self_negation(self, beliefs: List[Dict[str, Any]]) -> List[ContradictionWarning]:
        """
        Detect beliefs that negate their own possibility or validity.
        
        Args:
            beliefs: List of belief dictionaries
            
        Returns:
            List of detected self-negation warnings
        """
        warnings = []
        
        # Patterns for recognizing self-negation
        self_negation_patterns = [
            ("cannot be known", "claims to know"),
            ("impossible to express", "is expressing"),
            ("cannot be understood", "claims understanding"),
            ("cannot be communicated", "is communicating"),
            ("no truth exists", "claims truth"),
            ("nothing is certain", "with certainty"),
            ("all concepts are meaningless", "meaningful concept"),
            ("logic doesn't apply", "logical conclusion")
        ]
        
        for belief in beliefs:
            content = belief["content"].lower()
            
            for negative, positive in self_negation_patterns:
                if negative in content and positive in content:
                    warning = ContradictionWarning(
                        contradiction_id=str(uuid.uuid4()),
                        type=ContradictionType.SELF_REFERENTIAL_PARADOX,
                        severity=ContradictionSeverity.WARNING,
                        description=f"Self-negating belief detected: '{belief['content']}'",
                        affected_entities=[belief["id"]],
                        resolution_suggestions=[
                            "Qualify the scope of the claim to avoid self-reference",
                            "Specify domains where the claim applies and doesn't apply",
                            "Convert absolute claim to a probabilistic or contextual one",
                            "Apply different logical levels to resolve the self-negation"
                        ],
                        needs_quarantine=belief.get("confidence", 0.5) > 0.7
                    )
                    warnings.append(warning)
                    break
        
        return warnings
    
    def _check_axiom_violations(self, beliefs: List[Dict[str, Any]]) -> List[ContradictionWarning]:
        """
        Check for violations of core axioms.
        
        Args:
            beliefs: List of belief dictionaries
        
        Returns:
            List of detected contradiction warnings
        """
        warnings = []
        axioms = self.core_philosophy.get_all_axioms()
        
        for belief in beliefs:
            content = belief["content"].lower()
            violations = []
            
            for axiom in axioms:
                # Check if belief contradicts axiom negations
                for negation in axiom.negations:
                    if negation.lower() in content:
                        violations.append((axiom.axiom_id, axiom.description))
            
            if violations:
                affected_axioms = [v[0] for v in violations]
                axiom_descriptions = [v[1] for v in violations]
                
                warning = ContradictionWarning(
                    contradiction_id=str(uuid.uuid4()),
                    type=ContradictionType.AXIOM_VIOLATION,
                    severity=ContradictionSeverity.ERROR,
                    description=f"Belief violates core axioms: '{belief['content']}'",
                    affected_entities=[belief["id"]],
                    affected_axioms=affected_axioms,
                    resolution_suggestions=[
                        f"Modify belief to align with axiom: {desc}" for desc in axiom_descriptions
                    ] + ["Remove the belief"],
                    needs_quarantine=True
                )
                warnings.append(warning)
        
        return warnings
    
    def _check_coherence_collapse_risk(self, beliefs: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ContradictionWarning]:
        """
        Check for risk of coherence collapse.
        
        Args:
            beliefs: List of belief dictionaries
            context: Context information including coherence metrics
        
        Returns:
            List of detected contradiction warnings
        """
        warnings = []
        
        # Extract context metrics
        coherence_score = context.get("global_coherence_score", 1.0)
        coherence_trend = context.get("recent_coherence_trend", 0.0)
        identity_stability = context.get("identity_stability", 1.0)
        recent_contradictions = context.get("contradictions_detected_recently", 0)
        
        # Detect high-risk beliefs
        high_risk_beliefs = []
        risk_terms = [
            "random reorganization", 
            "rapid schema evolution", 
            "identity shift", 
            "sudden transformation", 
            "unstable", 
            "contradictions acceptable", 
            "coherence sacrifice"
        ]
        
        for belief in beliefs:
            content = belief["content"].lower()
            matched_terms = [term for term in risk_terms if term in content]
            
            if matched_terms and belief.get("confidence", 0.5) > 0.6:
                high_risk_beliefs.append((belief, matched_terms))
        
        # Evaluate collapse risk
        collapse_risk = 0.0
        
        # Factor 1: Current coherence is low
        if coherence_score < 0.7:
            collapse_risk += 0.3
        
        # Factor 2: Coherence is trending down
        if coherence_trend < -0.03:
            collapse_risk += 0.2 * min(1.0, abs(coherence_trend) * 10)
        
        # Factor 3: Identity stability is compromised
        if identity_stability < 0.7:
            collapse_risk += 0.2
        
        # Factor 4: Many recent contradictions
        if recent_contradictions > 2:
            collapse_risk += 0.1 * min(1.0, recent_contradictions / 5)
        
        # Factor 5: Presence of high-risk beliefs
        if high_risk_beliefs:
            collapse_risk += 0.2 * min(1.0, len(high_risk_beliefs) / 3)
        
        # Generate warning if risk is significant
        if collapse_risk > 0.4 and high_risk_beliefs:
            # Determine severity based on risk level
            severity = ContradictionSeverity.WARNING
            if collapse_risk > 0.7:
                severity = ContradictionSeverity.CRITICAL
            elif collapse_risk > 0.5:
                severity = ContradictionSeverity.ERROR
            
            affected_entities = [belief[0]["id"] for belief in high_risk_beliefs]
            risk_descriptions = [
                f"'{belief[0]['content']}' (risky terms: {', '.join(terms)})" 
                for belief, terms in high_risk_beliefs
            ]
            
            warning = ContradictionWarning(
                contradiction_id=str(uuid.uuid4()),
                type=ContradictionType.COHERENCE_COLLAPSE_RISK,
                severity=severity,
                description=f"Potential coherence collapse risk detected (risk level: {collapse_risk:.2f})",
                affected_entities=affected_entities,
                resolution_suggestions=[
                    "Remove or modify high-risk beliefs",
                    "Reduce confidence in high-risk beliefs",
                    "Introduce stabilizing beliefs that emphasize coherence",
                    "Temporarily increase coherence thresholds",
                    "Pause belief updates until coherence stabilizes"
                ],
                needs_quarantine=severity == ContradictionSeverity.CRITICAL
            )
            warnings.append(warning)
        
        return warnings


def get_warning_system() -> ContradictionWarningSystem:
    """
    Get the global instance of the contradiction warning system.
    
    Returns:
        Contradiction warning system instance
    """
    global _WARNING_SYSTEM
    if _WARNING_SYSTEM is None:
        _WARNING_SYSTEM = ContradictionWarningSystem()
    return _WARNING_SYSTEM 